# TFT Layout Scout Notes

Scope: PC 16:9 screenshots only. The current implementation uses 1920x1080
as the reference canvas and scales region boxes by ratio for other 16:9
resolutions. These notes document the intended screen-state split, rough
regions, priority order, and the real-screenshot calibration pass still needed
before treating recognition output as reliable.

## Screen States

### normal_shop

Preparation phase with the bottom shop open. This is the MVP state and should
produce the richest read: stage, round, gold, level, XP, shop slots, bench,
board, traits, items, and players panel.

Primary regions:

- `stage` / `round`: top-center round marker.
- `gold`: lower-center economy number.
- `level` / `level_exp`: lower-center level and XP strip.
- `shop`: full bottom shop row.
- `shop_slots`: five champion cards in the bottom shop row.
- `shop_odds`: bottom-right shop odds strip.
- `refresh_button` / `buy_xp_button`: lower-left shop controls used as
  normal-shop state sanity checks.
- `bench`: unit bench above the shop row.
- `traits_panel`: left-side traits list.
- `items_area` / `equipment`: lower-left item bench and nearby item holder area.
- `board`: central board, mainly for visible unit and item-holder scouting.
- `players_panel`: right-side player list.

State clues:

- Shop row is visible and contains five card slots.
- Refresh and buy-XP controls are visible on the lower-left shop panel.
- No full-width augment choice overlay is blocking the board.

### combat

Combat phase. The shop can still be visible in some moments, but board state,
item holders, traits, and opponent/player information matter more than shop OCR.
Shop recognition should be considered secondary or skipped if motion blur or
combat effects reduce confidence.

Primary regions:

- `stage` / `round`: top-center round marker.
- `board`: active fight area.
- `items_area` / `equipment`: item bench and holder cues.
- `traits_panel`: active trait counts.
- `players_panel`: health, streak, and opponent context.

State clues:

- Units are fighting or moving on the board.
- Shop row may be hidden, dim, or stale compared with the board.
- Notifications and combat banners can temporarily overlap the top-center area.

### augment_select

Augment selection overlay. This state should be detected before normal shop
analysis because the overlay changes interaction priority and can obscure normal
regions. Current code tracks a right-side `augments` OCR area plus top-center
round context and notifications.

Primary regions:

- `stage` / `round`: keep round context.
- `augments`: augment text/card area.
- `notifications`: top-center overlay and state-detection text.

State clues:

- Augment cards or augment-choice panel is visible.
- Player should choose one option; shop and board recognition are lower value.
- If card layout differs by set, use this state as an overlay bucket first and
  refine exact card boxes after screenshot calibration.

### special

Catch-all for carousel, PvE reward, encounter, portal, set-specific selection,
and other non-standard screens. The first pass should detect this state and avoid
deep normal-shop assumptions. Recognition can still keep stage, round, board,
and notifications for context.

Primary regions:

- `stage` / `round`: top-center round marker.
- `board`: central playable area or special scene.
- `notifications`: prompt, encounter, or reward text.

State clues:

- Shop row is absent or not useful.
- The central board is replaced by a carousel, reward, portal, encounter, or
  modal selection scene.
- State-specific UI is visible but not yet modeled as a dedicated layout.

## Region Semantics

Each `ScreenRegion` carries two layout-related fields:

- `layout`: the default owning layout used for readable debug crop filenames.
- `active_layouts`: every screen state where the region can be useful.

Shared regions such as `stage` and `round` are active in all states. Board,
items, traits, and player-list regions are active in `normal_shop` and `combat`,
with `board` also active in `special` as a broad context crop. Shop-specific
regions stay restricted to `normal_shop`, while `augments` stays restricted to
`augment_select`.

## PC 16:9 Rough Region Map

Reference canvas: 1920x1080. Boxes are approximate and should be validated
against real screenshots from the active TFT set and client scaling.

```text
Y=0
+--------------------------------------------------------------------------------+
|                         stage / round: 870,18 - 1050,64                        |
|                  notifications: 650,96 - 1270,220                               |
| traits_panel               board / combat field                 players_panel  |
| 0,120 - 315,690            360,185 - 1560,735                  1600,160-1918,760|
|                                                                                |
|                                                                                |
| items/equipment      bench: 462,665 - 1458,784                                  |
| 220,610 - 462,824    level/xp: 760,706 - 1048,790                               |
|                      gold: 820,760 - 930,805                                    |
| refresh/buy XP       shop row: 470,800 - 1422,1054        shop odds:1420-1688   |
| 242,824-418,974      five slots, each roughly 176x242 px                         |
+--------------------------------------------------------------------------------+
Y=1080
```

Detailed current boxes:

| Region | Box on 1920x1080 | Purpose |
| --- | --- | --- |
| `stage`, `round` | 870,18 - 1050,64 | Round state OCR |
| `notifications` | 650,96 - 1270,220 | Overlay/state detection OCR |
| `traits_panel` | 0,120 - 315,690 | Trait OCR and icon matching |
| `board` | 360,185 - 1560,735 | Board units and item holders |
| `players_panel` | 1600,160 - 1918,760 | Player health/list context |
| `items_area` | 220,610 - 462,824 | Item component matching |
| `equipment` | 240,620 - 455,815 | Item/equipment matching alias |
| `bench` | 462,665 - 1458,784 | Bench unit matching |
| `level` | 898,708 - 1022,765 | Level OCR |
| `level_exp` | 760,706 - 1048,790 | Level and XP OCR |
| `gold` | 820,760 - 930,805 | Gold OCR |
| `shop` | 470,800 - 1422,1054 | Full shop row |
| `shop_slot_1` | 474,806 - 650,1048 | Unit recognition |
| `shop_slot_2` | 666,806 - 842,1048 | Unit recognition |
| `shop_slot_3` | 858,806 - 1034,1048 | Unit recognition |
| `shop_slot_4` | 1050,806 - 1226,1048 | Unit recognition |
| `shop_slot_5` | 1242,806 - 1418,1048 | Unit recognition |
| `shop_odds` | 1420,798 - 1688,855 | Shop odds OCR |
| `refresh_button` | 242,824 - 418,894 | UI context |
| `buy_xp_button` | 242,904 - 418,974 | UI context |
| `augments` | 1565,105 - 1900,540 | Augment text/card OCR |

## Priority Order

Use priority as a capture and interpretation order, not as a guarantee of final
truth. State detection should decide the active layout first, then run the
regions that matter for that layout.

| Priority | Regions | Why |
| --- | --- | --- |
| 1 | `stage`, `round`, `augments` in augment state | Identify game phase and overlay state before deeper parsing. |
| 2 | `gold` | Economy is central to shop decisions and cheap to OCR. |
| 3 | `level`, `level_exp` | Level/XP changes shop odds and purchase advice. |
| 4 | `shop`, `shop_slots`, `shop_odds` | Main normal-shop recognition path. |
| 5 | `bench` | Needed for upgrade detection and trait gap context. |
| 6 | `traits_panel` | Active trait count and direction. |
| 7 | `items_area`, `equipment` | Item bias and holder information. |
| 8 | `board` | Board unit context; high value but visually noisy. |
| 9 | `players_panel`, `refresh_button`, `buy_xp_button` | Secondary context and state sanity checks. |
| 10 | `notifications` | Overlay prompts and transient state clues. |

Layout-specific priority notes:

- `normal_shop`: run priorities 1-7 first, then board/player context.
- `combat`: run stage/round, board, items, traits, players; defer shop OCR.
- `augment_select`: run stage/round, augment OCR, notifications; suppress normal
  shop conclusions while the overlay is present.
- `special`: run stage/round, notifications, broad board crop; skip deep
  recognition until the special state gets its own calibrated boxes.

## Real Screenshot Calibration Checklist

Collect screenshots at native 1920x1080 first, then repeat at at least one other
16:9 resolution such as 2560x1440 or 1366x768 to verify ratio scaling.

- Capture at least three `normal_shop` screenshots: early game, mid game, and
  late game with different shop costs and bench occupancy.
- Capture at least two `combat` screenshots: one with effects near the board
  center and one with the shop visible or partially visible.
- Capture at least three `augment_select` screenshots: silver, gold, and prismatic
  choices if available; include reroll state if the current set supports it.
- Capture at least three `special` screenshots: carousel, PvE/reward, and one
  set-specific encounter or portal/selection screen.
- For each screenshot, save debug crops from `save_debug_crops()` and check that
  every crop contains the intended UI element with 4-12 px of safe padding where
  possible.
- Verify `stage` / `round` OCR through every state; top-center banners should not
  clip the round number in normal cases.
- Verify `gold`, `level`, and `level_exp` against manual labels; adjust boxes if
  digits are clipped by client scaling or language settings.
- Verify all five shop slots separately; the crop should include champion art,
  name/cost area as needed, and should not bleed into adjacent slots enough to
  confuse template hashing.
- Verify `bench` with empty, partially filled, and full benches; make sure units
  at both ends are not clipped.
- Verify `traits_panel` in collapsed and expanded/scroll states if the client or
  set UI supports them.
- Verify `items_area` and `equipment` with loose components, completed items, and
  item-holder overlap near the lower-left board edge.
- Verify `players_panel` with different scoreboard states, including hover or
  expanded views if those can appear in captured screenshots.
- Record any UI scale, language, spectator/replay mode, or set-specific skin that
  changes coordinates; keep the base profile tied to standard live-game PC 16:9.
- Treat a state as calibrated only after at least 90% of its required crops are
  visually correct across the collected screenshots and the remaining misses are
  documented with a concrete follow-up box adjustment.

`game_companion_summarize_layout_calibration` exposes this as
`crop_acceptance`: all checks must be reviewed, at least 90% must be marked
`pass`, and any `fail` / `needs_adjustment` checks must include a note. Only then
can `ready_for_recognition` become true, assuming sample coverage and screenshot
readiness also pass.

The final gate is intentionally stricter than the global 90% number:

- `layout_acceptance`: every expected layout state represented in the batch must
  meet the same review and pass-rate requirements.
- `critical_acceptance`: `stage_round_clean`, `gold_clean`, `level_exp_clean`,
  and `shop_slots_complete` must all pass. These regions feed phase-5 OCR and
  shop recognition, so they are not allowed to be accepted as known misses.

Structured calibration batches can pass `samples` instead of plain
`image_paths`. Each sample may include:

```json
{
  "image_path": "D:/captures/tft_normal_shop_01.png",
  "expected_layout": "normal_shop",
  "tags": ["shop_open", "shop_five_units", "bench_units"],
  "label": "early-game shop with bench",
  "note": "1920x1080, standard UI scale"
}
```

The report coverage summary tracks whether the batch includes all four layout
states and the recommended tags: `shop_open`, `shop_five_units`, `bench_units`,
`traits_panel_expanded`, and `items_visible`.

For a local screenshot folder, run
`game_companion_init_layout_calibration_workspace` first to create the ignored
local workspace folders. Then run `game_companion_prepare_layout_calibration_manifest`.
It scans common image extensions, writes an editable `samples_manifest.json`, and
infers `expected_layout` / `tags` from filenames when possible. Manually fix any
missing labels in the manifest, then pass `samples_manifest_path` into
`game_companion_calibrate_layout`.

Use `game_companion_layout_calibration_status` as a read-only preflight check
before each step. It can inspect an input screenshot directory, an editable
samples manifest, or a generated calibration report and return the next action
without writing crops or mutating files.

Manual crop checks can be edited directly in `calibration_report.json` or updated
one at a time with `game_companion_update_layout_calibration_check`, which also
refreshes the report summary and HTML review page.
For reviewing many crops at once, use
`game_companion_update_layout_calibration_checks` with a list of updates so the
report and HTML are regenerated once after the batch is applied.

Open calibration questions:

- Whether `augment_select` should split the right-side `augments` region into
  exact card boxes after real screenshots confirm the active set layout.
- Whether `combat` needs a separate opponent-board crop when scouting transitions
  differ from the player's board framing.
- Whether `special` should remain a catch-all or split into carousel, PvE reward,
  encounter, and portal layouts once enough screenshots exist.
