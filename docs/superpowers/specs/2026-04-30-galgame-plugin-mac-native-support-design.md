# Galgame Plugin Mac Native Support Design

Status: Draft approved in brainstorming on 2026-04-30.

## 1. Summary

This design adds macOS native game support to `plugin/plugins/galgame_plugin`
by preserving the existing Windows-oriented top-level protocol and introducing
platform-specific backend implementations underneath it.

The target scope is intentionally narrow:

- Support **native macOS galgame windows only**
- Do **not** support Wine / Whisky / CrossOver / Parallels windows in the
  first implementation
- Preserve the existing UI flow, runtime payload structure, polling loop, and
  session/snapshot/history semantics
- Avoid broad edits to existing Windows logic; prefer additive backend files
  and narrow platform dispatch points

The resulting system should feel like the same plugin on both platforms, with
the same surfaces and status fields, while using a macOS-specific OCR capture
chain under the hood.

## 2. Goals

1. Add macOS native-game support without forking the plugin or introducing a
   second UI.
2. Keep the existing `bridge_tick` / `reader_mode` / `memory_reader_runtime` /
   `ocr_reader_runtime` contract intact.
3. Preserve manual workflow parity with Windows:
   - refresh window list
   - select a game window
   - lock/clear a target
   - run OCR and feed snapshot/history/runtime
4. Keep Windows behavior unchanged by default.
5. Follow the established code style and avoid scattering new hard-coded
   prompts, display strings, or platform-specific ad hoc branches throughout
   the codebase.

## 3. Non-goals

1. Do not implement a macOS equivalent of Textractor or another true
   memory-reader pipeline in the first version.
2. Do not support non-native compatibility-layer windows in the first version.
3. Do not replace existing Windows backends.
4. Do not build a separate macOS-only plugin or duplicate the frontend.
5. Do not redesign the snapshot/history/session model.

## 4. Constraints

### 4.1 Product and scope constraints

- The supported scope is **macOS native Steam games only insofar as they are
  native macOS windows**. Steam itself is not a dependency boundary; native
  window behavior is.
- The plugin must continue to expose the same UI and the same main status
  structures on both platforms.

### 4.2 Engineering constraints

- Avoid modifying unrelated or user-owned code.
- Prefer additive files and small integration points.
- Match existing naming, runtime field shapes, and status/detail patterns.
- Do not hard-code user-facing explanatory text in low-level platform logic.
- Do not silently change Windows defaults in `plugin.toml` just to fit macOS.

## 5. Existing Windows Model To Preserve

The current plugin has a clear layered model that must stay stable:

1. **Top-level scheduler**
   - `GalgamePlugin.bridge_tick()` coordinates polling, OCR/manual advance
     refresh, background bridge polling, and agent updates.

2. **Reader mode contract**
   - `reader_mode` remains `auto`, `memory`, or `ocr`.
   - Upper layers assume those values and derive status/candidate behavior from
     them.

3. **Runtime payload contract**
   - `memory_reader_runtime` and `ocr_reader_runtime` are carried through the
     plugin state and UI regardless of platform.

4. **Candidate/session model**
   - Memory reader, bridge SDK, and OCR contribute candidate sessions.
   - A shared candidate selection path chooses the active session.

5. **Snapshot/history outputs**
   - Once a line is accepted, it flows into current snapshot, line history,
     observed lines, choice state, and related runtime diagnostics.

The macOS design must preserve this model rather than replacing it.

## 6. Proposed Architecture

### 6.1 High-level approach

The design uses a **shared top-level protocol with platform-specific OCR input
backends**.

The plugin keeps one scheduler, one state model, one UI, and one OCR output
pipeline. The only new responsibilities are platform backends that provide:

- macOS window enumeration
- foreground window detection
- macOS-native window capture
- macOS permission inspection

The scheduler and UI still interact with the same plugin state fields and API
shapes, so macOS support appears as another backend path, not another feature
surface.

### 6.2 Why this approach

This approach is preferred because it best satisfies the approved constraints:

- It follows the existing Windows chain instead of inventing a new one.
- It minimizes invasive edits to existing code.
- It isolates platform details into new files.
- It preserves stable contracts for the UI and the rest of the plugin.

## 7. Module Boundaries

### 7.1 Existing files that should remain mostly unchanged

- `plugin/plugins/galgame_plugin/__init__.py`
  - Keep as the top-level orchestrator and state integrator.
  - Only add narrow platform dispatch and backend wiring.

- `plugin/plugins/galgame_plugin/static/main.js`
  - Keep the existing UI structure and workflow.
  - Only extend status rendering and platform-aware explanations where needed.

- `plugin/plugins/galgame_plugin/state.py`
  - Keep the existing state container model.

- `plugin/plugins/galgame_plugin/reader.py`
  - Keep downstream snapshot/history consumption logic unchanged.

### 7.2 Existing files that will need narrow changes

- `plugin/plugins/galgame_plugin/ocr_reader.py`
  - This remains the main OCR pipeline entry point.
  - Replace direct Windows-only platform assumptions with calls into a
    platform capability layer for:
    - listing windows
    - checking foreground state
    - selecting capture backends
    - checking platform permissions
  - Keep OCR normalization, stability filtering, runtime shaping, and snapshot
    emission logic shared.

- `plugin/plugins/galgame_plugin/service.py`
  - Extend config parsing and effective backend resolution so macOS can map to
    supported native capture behavior without changing Windows defaults.

- `plugin/plugins/galgame_plugin/plugin.toml`
  - Do not rewrite Windows defaults.
  - Keep the committed defaults intact; resolve platform-specific effective
    behavior at runtime.

### 7.3 New files to add

- `plugin/plugins/galgame_plugin/mac_window_support.py`
  - Enumerate native macOS candidate windows
  - Read current foreground window data
  - Produce data shaped to match the current window/runtime expectations

- `plugin/plugins/galgame_plugin/mac_capture_support.py`
  - Capture pixel data for a selected macOS window
  - Normalize Retina/scaling behavior into the image sizes expected by the
    existing OCR processing flow

- `plugin/plugins/galgame_plugin/mac_permissions_support.py`
  - Detect Screen Recording permission state
  - Detect Accessibility permission state if needed for reliable foreground or
    window access
  - Return machine-readable permission detail codes

No separate `mac_galgame_plugin.py`, `mac_ocr_reader.py`, or duplicate UI
bundle should be introduced.

## 8. macOS Data Flow

### 8.1 Top-level flow

The macOS flow must reuse the same top-level path:

1. `bridge_tick` runs as usual.
2. Reader-mode gating is computed as usual.
3. `memory_reader_runtime` remains present in state, even if unsupported.
4. `ocr_reader_runtime` becomes the active working data source for macOS.
5. OCR-generated candidates feed the existing candidate/session selection path.
6. Accepted lines flow into the existing snapshot/history pipeline.

### 8.2 Reader mode behavior on macOS

- `reader_mode = memory`
  - The mode remains representable.
  - The runtime must clearly report that the memory reader is unavailable on
    the current platform.
  - The plugin must not fake memory-reader success on macOS.

- `reader_mode = ocr`
  - This is the primary supported mode for macOS native games.

- `reader_mode = auto`
  - This remains `auto` in config and status semantics.
  - On macOS native games, the effective working path should fall through to
    OCR because macOS memory-reader support is unavailable.
  - The code must not silently rewrite the stored config value from `auto` to
    `ocr`.

### 8.3 Runtime compatibility rules

`memory_reader_runtime` must remain present with the same general shape as on
Windows, but report unsupported or inactive platform details.

`ocr_reader_runtime` must remain the real working runtime on macOS and continue
to supply:

- status/detail
- selected target identity
- capture backend diagnostics
- OCR backend diagnostics
- accepted/rejected line metadata
- capture profile status
- foreground/advance-related state where applicable

### 8.4 Window targeting workflow

The current user flow must remain intact:

1. Refresh window list
2. Select a game window
3. Persist a manual lock if chosen
4. Clear the lock if needed
5. Run OCR against the effective target

Only the low-level implementation of:

- `list_windows_snapshot`
- manual target resolution
- current target refresh
- capture backend execution

should change for macOS.

## 9. Permissions and Status Design

### 9.1 Permission model

macOS support will likely depend on:

- **Screen Recording** permission for window capture
- **Accessibility** permission for some combinations of foreground/window
  detection and user-focused diagnostics

If permissions are missing, the plugin must fail explicitly and diagnostically,
not silently.

### 9.2 Status/detail codes

Low-level logic should emit stable machine-readable detail codes instead of
hard-coded user explanations. Examples:

- `screen_recording_permission_denied`
- `accessibility_permission_denied`
- `mac_window_capture_unavailable`
- `no_eligible_window`
- `unsupported_platform`

Display copy should be mapped from these codes at a higher layer so that:

- text can be localized
- copy can be refined later
- platform logic does not accumulate scattered strings

## 10. Capture and OCR Design Considerations

### 10.1 Window enumeration

macOS window enumeration must provide enough data for the existing target
selection heuristics to remain useful:

- process name
- pid
- window title
- geometry
- foreground/active hints

The design should prefer adapting macOS window data into the same structural
shape already consumed by the OCR reader, rather than teaching all downstream
logic about a second window schema.

### 10.2 Screenshot capture

The capture backend must normalize:

- Retina scaling
- logical vs physical coordinate mismatches
- cropped window bounds
- minimized/unavailable window cases

The output should be compatible with the existing OCR preprocessing path so the
same OCR backends and text stabilization logic can be reused.

### 10.3 OCR quality and calibration

macOS support cannot assume native games are easier to OCR than Windows games.
The first implementation must preserve the current OCR workflow features that
already exist to make difficult titles usable:

- manual window selection
- refresh/re-detect
- capture profile matching
- capture-region calibration/fallback
- stability filtering before line acceptance

## 11. Configuration Strategy

The implementation should not rewrite the committed Windows-first defaults in
`plugin.toml` simply because macOS support is being added.

Instead:

1. Parse the same config schema.
2. Compute **effective** platform behavior at runtime.
3. Use explicit runtime diagnostics to show why a backend was chosen or not
   chosen.

This avoids breaking existing Windows expectations and keeps the config surface
stable.

## 12. UX and Localization Rules

The existing frontend flow remains the same. macOS-specific support should be
expressed through status and guidance, not through a parallel UI path.

Rules:

1. Do not hard-code new user-facing platform guidance deep inside backend
   logic.
2. Keep stable detail/reason/status identifiers in backend/runtime code.
3. Map those identifiers to display strings in the UI or a concentrated
   translation/mapping layer.
4. Any new prompt-like or explanatory text must be stored in the established
   configuration/mapping style rather than inline in business branches.

## 13. Testing Strategy

### 13.1 Unit coverage

Add targeted tests for:

- macOS permission-state mapping
- macOS window snapshot normalization
- platform dispatch in OCR reader/backend selection
- `reader_mode=auto` macOS fallback behavior
- retention of `memory_reader_runtime` shape under macOS unsupported flow

### 13.2 Integration coverage

Add plugin-level tests that verify:

- macOS platform path does not crash plugin startup
- macOS OCR runtime can expose a candidate window
- macOS OCR candidate can become the active data source
- snapshot/history writes remain compatible with existing consumers

### 13.3 Regression protection

Windows regression tests must continue to pass unchanged. The macOS work is not
acceptable if it alters default Windows backend selection or breaks existing
Windows runtime/status behavior.

## 14. Acceptance Criteria

The first supported version is acceptable only if all of the following hold:

1. The plugin UI can show a meaningful macOS OCR runtime state.
2. Missing permissions produce explicit status/detail output rather than a
   silent failure.
3. Refreshing windows can surface native macOS game windows as candidates.
4. A user can manually select and lock a native macOS game window.
5. The plugin can capture that window and run OCR on it.
6. OCR output can enter the existing snapshot/history/runtime pipeline.
7. Existing UI workflow and field semantics remain consistent with Windows.
8. Windows behavior and committed defaults remain unchanged.

## 15. Risks

### 15.1 Medium risk

- Retina coordinate mismatches may cause capture-region drift.
- Some native games may expose unstable or low-signal window titles.
- Foreground detection may differ from Windows assumptions and affect advance
  heuristics.

### 15.2 High risk

- Screen Recording permission may be absent or revoked, making the core OCR
  path unusable until the user grants access.
- Some native rendering stacks may produce captures that are OCR-hostile due
  to transparency, animation, or text effects.

## 16. Rollout Phases

Recommended implementation order:

1. **P1**
   - macOS permission detection
   - macOS window enumeration
   - manual window lock path
   - macOS window capture

2. **P2**
   - integrate captured images into the existing OCR runtime/snapshot/history
     path

3. **P3**
   - preserve and tune capture-profile and calibration behavior for macOS

4. **P4**
   - add richer diagnostics, localized copy mapping, and frontend polish

## 17. Design Decision Summary

The approved design is:

- one plugin
- one scheduler
- one UI
- one runtime/state contract
- one shared OCR output pipeline
- platform-specific native backends beneath it

This gives macOS native-game support while preserving the existing Windows
architecture and minimizing risk to unrelated code.
