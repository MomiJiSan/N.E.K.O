import json

import re

import shutil

import subprocess

import textwrap

from pathlib import Path

import pytest

from tests.node_harness import run_node_script

APP_WEBSOCKET_PATH = Path(__file__).resolve().parents[2] / "static" / "app" / "app-websocket.js"

APP_STATE_PATH = Path(__file__).resolve().parents[2] / "static" / "app" / "app-state.js"

APP_SETTINGS_PATH = Path(__file__).resolve().parents[2] / "static" / "app" / "app-settings.js"

APP_AUDIO_CAPTURE_PATH = Path(__file__).resolve().parents[2] / "static" / "app" / "app-audio-capture.js"

APP_BUTTONS_PATH = Path(__file__).resolve().parents[2] / "static" / "app" / "app-buttons.js"

TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "templates"

APP_GAME_VOICE_CONTROL_PATH = (
    Path(__file__).resolve().parents[2] / "static" / "app" / "app-game-voice-control.js"
)

def _block_after(js: str, opener: str) -> str:
    """Return the brace-balanced body that follows ``opener``.

    CodeRabbit: ``split("}", 1)[0]`` truncates at the FIRST closing brace in the
    body -- a nested ``if {...}``, an object literal, even a ``}`` inside a
    string -- so the slice can shrink to a line or two and the assertions then
    pass by accident, or miss a real regression. Count braces instead, skipping
    those inside string literals and line comments.

    Two opener shapes are supported: one ending in ``{`` (scope = that block),
    and a plain statement (scope = the rest of its enclosing block). Both leave
    ``depth`` at 1. A TRUNCATED opener is neither -- ``"function foo("`` stops
    before the body brace, so the body's own ``{`` pushes depth to 2 and the
    scan runs past the function into everything that follows it (CodeRabbit
    caught two of these scoped to 1131 lines instead of 29, where the
    assertions could match an unrelated function). An opener with unbalanced
    parentheses is exactly that mistake, so reject it here rather than let a
    later reader rediscover it.
    """

    if opener.count("(") != opener.count(")"):
        raise AssertionError(
            f"opener has unbalanced parentheses, so it stops mid-signature "
            f"and the scan would overrun the block: {opener!r}"
        )
    rest = js.split(opener, 1)[1]
    depth = 1
    out = []
    quote = None
    i = 0
    while i < len(rest):
        ch = rest[i]
        if quote:
            if ch == "\\":
                out.append(rest[i : i + 2])
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in "\"'`":
            quote = ch
        elif ch == "/" and rest[i : i + 2] == "//":
            end = rest.find("\n", i)
            end = len(rest) if end == -1 else end
            out.append(rest[i:end])
            i = end
            continue
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out)
        out.append(ch)
        i += 1
    raise AssertionError(f"unbalanced block after {opener!r}")

def _code_only(js: str) -> str:
    """Strip // line comments so 'does not do X' assertions test code, not prose.

    Several pins in this file assert that a block does NOT call something; a
    comment explaining why it must not would otherwise trip them.
    """

    return "\n".join(line.split("//", 1)[0] for line in js.splitlines())

LOCALES_PATH = Path(__file__).resolve().parents[2] / "static" / "locales"

WEBSOCKET_ROUTER_PATH = Path(__file__).resolve().parents[2] / "main_routers" / "websocket_router.py"

ASR_REGISTRY_META_PATH = Path(__file__).resolve().parents[2] / "main_logic" / "asr_client" / "_registry_meta.py"

def _run_settings_node_harness(script: str) -> subprocess.CompletedProcess[str]:
    node_path = shutil.which("node")
    if not node_path:
        pytest.skip("node is not installed; skipping app-settings harness test")
    # run_node_script writes the script to a temp file: node -e would put the
    # whole harness on the command line, which Windows refuses past 32767
    # characters and which encodes under the locale codec rather than UTF-8.
    return run_node_script(
        node_path,
        script,
        cwd=str(Path(__file__).resolve().parents[2]),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

def test_game_route_close_events_require_matching_generation_when_one_is_active():
    source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")

    assert re.search(
        r"\(endedRouteInstanceId \|\| currentRouteInstanceId\)\s*"
        r"&&\s*"
        r"endedRouteInstanceId !== currentRouteInstanceId",
        source,
    )
    assert re.search(
        r"\(incomingGameRouteInstanceId \|\| currentGameRouteInstanceId\)\s*"
        r"&&\s*"
        r"incomingGameRouteInstanceId !== currentGameRouteInstanceId",
        source,
    )

def test_rejected_close_events_still_tombstone_their_own_identity():
    """A close event this window rejects still names a dead route.

    ``GAME_ROUTE_ENDED`` and ``game_window_state_change: closed`` are emitted
    only from route finalize, so the identity in the payload is provably dead
    even when it does not match what this window currently holds. Dropping it
    without a tombstone lets a late STT gate for that identity re-activate
    ``S.gameRouteActive`` once the current route also ends -- which suppresses
    proactive chat and auto-goodbye until a full open/close cycle or a reload.

    The tombstone must use the payload's OWN identity: falling back to the
    current one would tombstone the live route and permanently reject its real
    gate, which is the fail-closed direction.
    """
    source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")
    rejected_branches = [
        ("忽略过期的 GAME_ROUTE_ENDED | ended_session=", "return;"),
        ("忽略过期的 GAME_ROUTE_ENDED | ended_route=", "return;"),
        ("[GameWindow] 忽略过期窗口事件", "} else if (detail.action === 'opened')"),
    ]
    for marker, terminator in rejected_branches:
        assert source.count(marker) == 1, marker
        # Start after the guard's own console.log, which legitimately prints the
        # live identity it is comparing against.
        start = source.index(");", source.index(marker)) + 2
        end = source.index(terminator, start)
        block = source[start:end]
        assert "rememberEndedGameRouteIdentity(" in block, (
            f"a rejected close event ({marker}) forgot the identity it just refused"
        )
        for live_identity in ("currentSessionId", "currentGameSessionId",
                              "currentRouteInstanceId", "currentGameRouteInstanceId",
                              "S.gameRouteGameType", "S.gameRouteSessionId"):
            assert live_identity not in block, (
                f"a rejected close event ({marker}) tombstoned the live route via "
                f"{live_identity}"
            )

def test_disabled_independent_asr_is_a_normal_native_status_without_failure_toast():
    source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")

    status_block = source.split(
        "if (statusCode && statusCode.indexOf('ASR_INDEPENDENT_') === 0)",
        1,
    )[1].split("if (statusCode === 'TTS_CONNECTION_FAILED')", 1)[0]
    disabled_branch = status_block.split(
        "if (statusCode === 'ASR_INDEPENDENT_DISABLED')",
        1,
    )[1].split("if (statusCode === 'ASR_INDEPENDENT_INJECTION_FAILED')", 1)[0]

    assert "S.independentAsrActive = false;" in disabled_branch
    assert "return;" in disabled_branch
    assert "independentAsrFallback" not in disabled_branch

def test_provider_unavailable_status_names_provider_and_denies_silent_switch():
    source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")

    assert "ASR_INDEPENDENT_PROVIDER_UNAVAILABLE" in source
    assert "microphone.independentAsrProviderUnavailable" in source
    assert "{ providerKey: asrProvider || 'unknown' }" in source
    assert "It did not switch to another speech recognition service" in source

def test_lifecycle_blocked_clears_independent_asr_and_shows_failure_toast():
    # runtime.py _handle_independent_asr_error always broadcasts lifecycle
    # BLOCKED before the fatal status code, and most fatal codes
    # (ASR_ENDPOINTING_FAILED, ASR_BLOCKED_ENDPOINTING,
    # ASR_AUDIO_ORDERING_FAILED, ASR_PROVIDER_FINAL_TIMEOUT, provider codes)
    # do NOT carry the ASR_INDEPENDENT_ prefix. The failure teardown must
    # therefore hang off the BLOCKED lifecycle notification, not off a
    # fatal-code enumeration.
    source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")

    lifecycle_block = source.split("if (statusCode === 'ASR_LIFECYCLE_STATE')", 1)[1].split(
        "if (statusCode === 'VOICE_INPUT_LEASE_RESYNC_REQUIRED')",
        1,
    )[0]

    # BLOCKED teardown lives inside the validated-state branch.
    assert "if (lifecycleState === 'blocked')" in lifecycle_block
    assert lifecycle_block.index("allowedLifecycleStates.indexOf(lifecycleState)") < lifecycle_block.index(
        "if (lifecycleState === 'blocked')"
    )

    blocked_branch = lifecycle_block.split("if (lifecycleState === 'blocked')", 1)[1]
    # Performed by the shared teardown helper the branch calls.
    assert "tearDownBlockedVoiceRoute();" in blocked_branch
    teardown_fn = source.split("function tearDownBlockedVoiceRoute() {", 1)[1].split(
        "\n    }", 1
    )[0]
    assert "removeExternalAsrPreview();" in teardown_fn
    assert "S.independentAsrActive = false;" in teardown_fn
    assert teardown_fn.index("removeExternalAsrPreview();") < teardown_fn.index(
        "S.independentAsrActive = false;"
    )
    # The teardown runs before the toast, so the failure message is what stays
    # on screen.
    assert blocked_branch.index("tearDownBlockedVoiceRoute();") < blocked_branch.index(
        "microphone.independentAsrFallback"
    )

    # Cross-reference comment so backend changes to the failure path get
    # traced back here.
    assert "_handle_independent_asr_error" in lifecycle_block

    # Start-path failures never emit BLOCKED; the per-code toasts in the
    # ASR_INDEPENDENT_ prefix branch must survive.
    prefix_block = source.split(
        "if (statusCode && statusCode.indexOf('ASR_INDEPENDENT_') === 0)",
        1,
    )[1].split("if (statusCode === 'TTS_CONNECTION_FAILED')", 1)[0]
    assert "microphone.independentAsrProviderUnavailable" in prefix_block
    assert "microphone.independentAsrFallback" in prefix_block

def test_provider_names_cover_asr_registry_keys_in_all_locales():
    registry_source = ASR_REGISTRY_META_PATH.read_text(encoding="utf-8")
    registry_keys = set(re.findall(r'provider_key="([a-z0-9_]+)"', registry_source))
    assert registry_keys, "provider_key extraction regex no longer matches _registry_meta.py"
    required_keys = registry_keys | {"unknown"}

    locale_names = sorted(path.name for path in LOCALES_PATH.glob("*.json"))
    assert len(locale_names) == 8

    key_sets = {}
    for locale_name in locale_names:
        locale = json.loads((LOCALES_PATH / locale_name).read_text(encoding="utf-8"))
        provider_names = locale["api"]["providerNames"]
        key_sets[locale_name] = set(provider_names)
        missing = required_keys - set(provider_names)
        assert not missing, f"{locale_name} providerNames missing: {sorted(missing)}"
        for key in required_keys:
            value = provider_names[key]
            assert isinstance(value, str) and value.strip(), f"{locale_name} providerNames[{key}] is empty"

    reference_locale = locale_names[0]
    for locale_name in locale_names[1:]:
        assert key_sets[locale_name] == key_sets[reference_locale], (
            f"providerNames key set of {locale_name} diverges from {reference_locale}"
        )

def test_websocket_has_no_widget_mode_capability_or_lifecycle_protocol():
    frontend_source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")
    router_source = WEBSOCKET_ROUTER_PATH.read_text(encoding="utf-8")

    assert "widget_mode_capable" not in frontend_source
    assert "widget_mode_capable" not in router_source
    assert "response.type.startsWith('widget_mode_')" not in frontend_source
    assert "neko:widget-mode-message" not in frontend_source

def test_external_asr_preview_clears_only_on_current_session_terminals():
    source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")

    final_block = source.split("// -------- user_transcript --------", 1)[1].split(
        "// --------", 1
    )[0]
    session_ended_block = source.split(
        "// -------- session_ended_by_server --------", 1
    )[1].split("// -------- reload_page --------", 1)[0]
    onclose_block = source.split("// ---- onclose ----", 1)[1].split(
        "// ---- onerror ----", 1
    )[0]
    stale_guard, current_close = onclose_block.split(
        "console.log(window.t('console.websocketClosed'));", 1
    )
    onerror_block = source.split("// ---- onerror ----", 1)[1].split(
        "mod.connectWebSocket = connectWebSocket;", 1
    )[0]

    assert "removeExternalAsrPreview();" in final_block
    assert "removeExternalAsrPreview();" in session_ended_block
    assert "if (S.socket !== _thisSocket)" in stale_guard
    assert "removeExternalAsrPreview();" not in stale_guard
    assert "removeExternalAsrPreview();" in current_close
    assert "removeExternalAsrPreview();" not in onerror_block

def test_noise_reduction_toggle_uses_conversation_settings_cas_client():
    capture_source = APP_AUDIO_CAPTURE_PATH.read_text(encoding="utf-8")
    save_block = capture_source.split(
        "function saveNoiseReductionSetting() {",
        1,
    )[1].split("function loadNoiseReductionSetting()", 1)[0]
    settings_source = APP_SETTINGS_PATH.read_text(encoding="utf-8")

    assert "window.appSettings.saveSettings();" in save_block
    assert "fetch('/api/config/conversation-settings'" not in save_block
    assert "'noiseReductionEnabled'," in settings_source
    assert "noiseReductionEnabled: S.noiseReductionEnabled" in settings_source

def test_stale_unsupported_capability_does_not_override_paid_core_preference_harness():
    source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")
    start = source.index("function attachStartSessionHandshake(ws)")
    end = source.index("function connectWebSocket()", start)
    attach_source = source[start:end]
    harness = (
        """
        const S = {
          coreApiSupportsIndependentAsr: false,
          settingsHydrated: true,
          independentAsrAuthoritative: true,
          independentAsrEnabled: true,
          voiceInputResourceOptimizationAuthoritative: false,
        };
        """
        + attach_source
        + """
        const frames = [];
        const ws = { send(data) { frames.push(data); } };
        attachStartSessionHandshake(ws);
        ws.send(JSON.stringify({ action: 'start_session', input_type: 'audio' }));
        const sent = JSON.parse(frames[0]);
        if (sent.independent_asr_enabled !== true) {
          throw new Error('stale capability must not replace the authoritative preference');
        }
        console.log('ok');
        """
    )
    result = _run_settings_node_harness(harness)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"

def test_start_session_handshake_omitted_until_settings_hydrated():
    # On a fresh browser profile — or while the async conversation-settings
    # GET is still pending — S.independentAsrEnabled is only the boot default
    # false. Stamping that onto an early start_session would override the
    # backend's persisted true. The stamp must therefore be gated on
    # S.settingsHydrated; when the field is omitted the backend falls back to
    # its persisted setting (websocket_router forwards the absent field as
    # None; pinned by
    # test_start_session_handshake_missing_falls_back_to_persisted). A
    # permanently failing GET keeps the field omitted — persisted value
    # governs, which is the correct fallback.
    websocket_source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")
    state_source = APP_STATE_PATH.read_text(encoding="utf-8")

    wrapper = websocket_source.split(
        "function attachStartSessionHandshake(ws)",
        1,
    )[1].split("function connectWebSocket()", 1)[0]

    # The stamp exists exactly once and only inside the hydration-gated
    # branch: no second, unconditional assignment path.
    assert wrapper.count("msg.independent_asr_enabled") == 1
    assert (
        "msg.action === 'start_session' && S.settingsHydrated === true" in wrapper
    ), "independent_asr_enabled stamp must be gated on S.settingsHydrated"
    # Codex P2: settingsHydrated alone is not enough — it also flips on an
    # unrelated user preference change while independentAsrEnabled is still the
    # boot default. The stamp needs the per-key authority flag as well.
    assert "S.independentAsrAuthoritative === true" in wrapper, (
        "independent_asr_enabled stamp must also require per-key ASR authority"
    )

    # Both flags start false so a pre-hydration start_session omits the field.
    assert "settingsHydrated: false," in state_source
    assert "independentAsrAuthoritative: false," in state_source

def test_user_toggle_during_get_failure_marks_hydration_posts_and_stamps():
    # Round-10 semantics must survive the userInitiated split: an explicit
    # user change is an authoritative hydration source even while the settings
    # GET keeps failing. The independent-ASR toggle marks S.settingsHydrated
    # synchronously (before its POST awaits) and publishes the POST; the
    # start_session handshake then stamps the user's choice.
    capture_source = APP_AUDIO_CAPTURE_PATH.read_text(encoding="utf-8")
    settings_source = APP_SETTINGS_PATH.read_text(encoding="utf-8")
    websocket_source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")

    # The toggle's direct sync call is user-initiated and still POSTs.
    toggle_block = capture_source.split(
        "var asrToggle = createVoiceSettingToggle(", 1
    )[1].split("asrRow.appendChild(asrCopy);", 1)[0]
    persist_block = capture_source.split(
        "function persistVoiceSettingChange() {", 1
    )[1].split("function markVoiceSettingsPending", 1)[0]
    assert "persistVoiceSettingChange();" in toggle_block
    assert "window.appSettings.syncSettingsToServer({ userInitiated: true })" in persist_block

    # saveSettings' full (non-skipServerSync) path is the other user seam —
    # the settings popup, subtitle toggles and chat-window toggles all route
    # through it — so it must pass userInitiated too.
    save_fn = settings_source.split("function saveSettings(options)", 1)[1].split(
        "function loadSettings()",
        1,
    )[0]
    assert "syncSettingsToServer({ userInitiated: true });" in save_fn
    # ... while the first-launch boot save keeps skipping the sync entirely,
    # so boot defaults still never mark hydration.
    assert "saveSettings({ skipServerSync: true });" in settings_source

    # And the handshake stamp keys off exactly that flag, so the toggle's
    # pre-hydration change reaches the backend on the next start_session.
    wrapper = websocket_source.split(
        "function attachStartSessionHandshake(ws)",
        1,
    )[1].split("function connectWebSocket()", 1)[0]
    assert "msg.action === 'start_session' && S.settingsHydrated === true" in wrapper
    assert "msg.independent_asr_enabled = S.independentAsrEnabled === true;" in wrapper

def test_settings_posts_serialize_so_a_stale_body_cannot_win_persistence():
    # Codex P2 (round 13): flipping the ASR toggle twice before the first POST
    # completed used to start two independent syncSettingsToServer calls with
    # their own snapshots; the backend saves each in a separate
    # asyncio.to_thread, so the OLDER request could finish LAST and persist the
    # earlier toggle value. Pin the serialization fix: every sync queues behind
    # a module-level chain tail and builds its settings snapshot at SEND time
    # (inside the queued runSync), so at most one POST is in flight and the
    # last-issued request always carries the final local state.
    settings_source = APP_SETTINGS_PATH.read_text(encoding="utf-8")

    # The chain tail starts resolved and is module-scoped (shared by the
    # toggle path AND the periodic tick, so those cannot race each other
    # either).
    assert "let _syncChainTail = Promise.resolve();" in settings_source

    sync_fn = settings_source.split(
        "async function syncSettingsToServer(options)", 1
    )[1].split("function startPeriodicSync()", 1)[0]

    # Chaining structure: the queued body is attached on BOTH fulfillment and
    # rejection so one failed sync cannot stall the tail, the tail advances to
    # the newly chained promise, and the caller gets that promise back (the
    # toggle handler publishes it as S.pendingSettingsSyncPromise for the
    # ensureWebSocketOpen gate).
    assert "const chained = _syncChainTail.then(runSync, runSync);" in sync_fn
    assert "_syncChainTail = chained;" in sync_fn
    assert "return chained;" in sync_fn

    # The settings snapshot is built inside the queued runSync — at send time,
    # after the predecessor completed — not at call time.
    run_sync_index = sync_fn.index("const runSync = async () =>")
    snapshot_index = sync_fn.index("const settings = getConversationSettings();")
    fetch_index = sync_fn.index(
        "await _fetchConversationSettingsJsonWithTimeout("
    )
    assert run_sync_index < snapshot_index < fetch_index

    # Negative: the synchronous hydration mark and dirty-key recording stay
    # at call time, BEFORE the queued body — deferring them would reopen the
    # stale-GET-merge window and the pre-hydration handshake gap.
    assert sync_fn.index("S.settingsHydrated = true;") < run_sync_index
    assert sync_fn.index("_markUserDirtySettings();") < run_sync_index

def test_rapid_asr_toggle_double_flip_persists_final_state_harness():
    # Behavioral pin for the Codex P2 fix: drive the real syncSettingsToServer
    # with a controllable fetch and simulate the double-flip race. Before the
    # fix both POSTs were in flight together and completing them in reverse
    # order let the stale body be the backend's last save; now the second POST
    # must not even be issued until the first settles, and its body must carry
    # the final toggle state.
    harness = textwrap.dedent(
        """
        const fs = require('node:fs');
        const vm = require('node:vm');

        const source = fs.readFileSync(__APP_SETTINGS_PATH__, 'utf8');

        function assert(cond, msg) {
          if (!cond) throw new Error('ASSERT: ' + msg);
        }

        function makeContext() {
          // Module load runs loadSettings(), which issues a boot GET; the
          // harness settles it (as a failure) so the bounded settings-POST
          // gate opens without waiting for its timeout, and assertions below
          // look only at the POST calls.
          const postCalls = [];
          const getCalls = [];
          const timeoutCallbacks = [];
          const sandbox = {
            console: { log() {}, warn() {}, error() {} },
            setInterval() { return 0; },
            clearInterval() {},
            setTimeout(fn, ms) {
              // Unref'd so a pending gate timer cannot hold the process open;
              // the harness then exits naturally and stdout always flushes.
              const t = setTimeout(fn, ms);
              if (t && typeof t.unref === 'function') t.unref();
              timeoutCallbacks.push({ fn, ms, timer: t });
              return t;
            },
            clearTimeout,
            localStorage: { getItem() { return null; }, setItem() {}, removeItem() {} },
            document: { getElementById() { return null; } },
            fetch(url, opts) {
              return new Promise((resolve, reject) => {
                if (opts && opts.method === 'POST') {
                  postCalls.push({ url, body: opts.body, opts, resolve, reject });
                } else {
                  getCalls.push({ url, resolve, reject });
                }
              });
            },
          };
          sandbox.window = {
            appState: {
              independentAsrEnabled: false,
              slopFilterEnabled: false,
              focusModeEnabled: false,
              settingsHydrated: false,
            },
            appConst: {},
            appUtils: { mapRenderQualityToFollowPerf() { return 'medium'; } },
            slopFilterEnabled: false,
            focusModeEnabled: false,
            addEventListener() {},
            removeEventListener() {},
          };
          vm.createContext(sandbox);
          vm.runInContext(source, sandbox);
          // The harness drives hydration and the toggle state explicitly from
          // a clean baseline.
          sandbox.window.appState.settingsHydrated = false;
          return {
            postCalls,
            getCalls,
            timeoutCallbacks,
            S: sandbox.window.appState,
            mod: sandbox.window.appSettings,
          };
        }

        const okResponse = { ok: true, json: async () => ({ success: true }) };
        const tick = () => new Promise((resolve) => setImmediate(resolve));

        async function settleBootGet(ctx) {
          // Fail the boot GET (null result: no merge, no hydration) so the
          // settings-POST gate settles deterministically.
          assert(ctx.getCalls.length === 1, 'boot must issue the settings GET');
          ctx.getCalls[0].resolve({ ok: false });
          await tick();
          await tick();
        }

        async function main() {
          const ctx = makeContext();
          const { postCalls, S, mod } = ctx;
          await settleBootGet(ctx);
          assert(S.settingsHydrated === false, 'a failed boot GET must not mark hydration');

          // First flip: POST issued with the pre-second-flip snapshot.
          S.independentAsrEnabled = true;
          const p1 = mod.syncSettingsToServer({ userInitiated: true });
          assert(S.settingsHydrated === true, 'hydration must be marked synchronously at call time');
          await tick();
          assert(postCalls.length === 1, 'first sync must POST immediately');
          assert(JSON.parse(postCalls[0].body).independentAsrEnabled === true, 'first body snapshots true');

          // Second flip while the first POST is still in flight.
          S.independentAsrEnabled = false;
          const p2 = mod.syncSettingsToServer({ userInitiated: true });
          await tick();
          assert(postCalls.length === 1, 'second POST must be queued, not concurrent (reordered completions impossible)');

          // Only when the first settles does the second go out — carrying the
          // FINAL state because the snapshot is taken at send time.
          postCalls[0].resolve(okResponse);
          await tick();
          assert(postCalls.length === 2, 'queued sync must run after the predecessor completed');
          assert(JSON.parse(postCalls[1].body).independentAsrEnabled === false, 'last-issued body must carry the final toggle state');
          postCalls[1].resolve(okResponse);
          await p1;
          await p2;

          // Negative: a predecessor that fails (network reject) must neither
          // stall the chain nor reject the published promises.
          S.independentAsrEnabled = true;
          const p3 = mod.syncSettingsToServer({ userInitiated: true });
          await tick();
          S.independentAsrEnabled = false;
          const p4 = mod.syncSettingsToServer({ userInitiated: true });
          await tick();
          assert(postCalls.length === 3, 'third sync in flight, fourth queued');
          postCalls[2].reject(new Error('network down'));
          await tick();
          assert(postCalls.length === 4, 'a failed predecessor must not stall the queued sync');
          assert(JSON.parse(postCalls[3].body).independentAsrEnabled === false, 'post-failure sync still carries the final state');
          postCalls[3].resolve(okResponse);
          await p3;
          await p4;

          // A transport that never settles must time out and release the
          // serialization tail so the later user state can still be sent.
          const timeoutCountBeforeStall = ctx.timeoutCallbacks.length;
          S.independentAsrEnabled = true;
          const p5 = mod.syncSettingsToServer({ userInitiated: true });
          await tick();
          S.independentAsrEnabled = false;
          const p6 = mod.syncSettingsToServer({ userInitiated: true });
          await tick();
          assert(postCalls.length === 5, 'fifth sync in flight, sixth queued');
          const requestTimeout = ctx.timeoutCallbacks
            .slice(timeoutCountBeforeStall)
            .find(
              (timer) => timer.ms === 15000
          );
          assert(requestTimeout, 'the in-flight request must have a bounded timeout');
          requestTimeout.fn();
          await tick();
          await tick();
          assert(
            postCalls.length === 6,
            'a timed-out predecessor must release the queued sync'
          );
          assert(
            JSON.parse(postCalls[5].body).independentAsrEnabled === false,
            'the post-timeout sync must rebuild from the final state'
          );
          postCalls[5].resolve(okResponse);
          await p5;
          await p6;

          // Negative: a non-userInitiated (periodic-style) call never marks
          // hydration — and after a FAILED boot GET, with nothing the user
          // touched, it writes nothing at all (round 16: an attempt that
          // merged no server value must not license a full snapshot). Its
          // promise still resolves.
          const fresh = makeContext();
          await settleBootGet(fresh);
          const pp = fresh.mod.syncSettingsToServer();
          await tick();
          await tick();
          assert(fresh.S.settingsHydrated === false, 'periodic-style sync must not mark hydration');
          assert(
            fresh.postCalls.length === 0,
            'no merged server value and no dirty key means there is nothing safe to write'
          );
          await pp;

          // ... but once a real merge licensed full snapshots, the
          // periodic-style call still POSTs and still serializes behind an
          // in-flight sync (the chain itself is unchanged).
          const merged = makeContext();
          assert(merged.getCalls.length === 1, 'boot must issue the settings GET');
          merged.getCalls[0].resolve({
            ok: true,
            json: async () => ({
              success: true,
              settings: { independentAsrEnabled: true },
              telemetryBranch: null,
            }),
          });
          await tick();
          await tick();
          assert(merged.postCalls.length === 1, 'the merge writeback POST goes out first');
          const mp = merged.mod.syncSettingsToServer();
          await tick();
          assert(merged.postCalls.length === 1, 'the periodic-style sync queues behind it');
          merged.postCalls[0].resolve(okResponse);
          await tick();
          await tick();
          assert(merged.postCalls.length === 2, 'it goes out once the predecessor settled');
          assert(
            JSON.parse(merged.postCalls[1].body).independentAsrEnabled === true,
            'and carries the merged full snapshot'
          );
          merged.postCalls[1].resolve(okResponse);
          await mp;

          console.log('HARNESS_OK');
          // Timers in the sandbox are unref'd, so the process exits naturally
          // once main() returns and piped stdout is fully flushed.
          process.exitCode = 0;
        }

        main().catch((err) => {
          console.error(err && err.stack ? err.stack : String(err));
          process.exitCode = 1;
        });
        """
    ).replace("__APP_SETTINGS_PATH__", json.dumps(str(APP_SETTINGS_PATH)))

    result = _run_settings_node_harness(harness)
    assert result.returncode == 0, (
        "settings sync harness failed\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "HARNESS_OK" in result.stdout

def test_boot_get_converges_on_a_peer_asr_flip_it_did_not_witness_harness():
    # Issue #2540 (residual 2 of #2345): window A's boot GET merged the stale
    # server value onto independentAsrEnabled just because the key was not in
    # _dirtySettingsKeys, and A then stayed on that value. The sibling test
    # above only covers the flip arriving DURING the in-flight GET, where the
    # flip gate marks the key dirty. Pin the two interleavings it does not
    # reach -- the flip already sitting in the boot snapshot, and the flip
    # arriving after the merge already landed -- so the decision-tuple ordering
    # that makes both converge cannot silently regress into dirty-mark-only
    # gating again.
    harness = textwrap.dedent(
        """
        const fs = require('node:fs');
        const vm = require('node:vm');

        const source = fs.readFileSync(__APP_SETTINGS_PATH__, 'utf8');

        function assert(cond, msg) {
          if (!cond) throw new Error('ASSERT: ' + msg);
        }

        function makeContext(bootSnapshot) {
          const postCalls = [];
          const getCalls = [];
          const listeners = [];
          const store = Object.create(null);
          if (bootSnapshot) {
            store['project_neko_settings'] = JSON.stringify(bootSnapshot);
          }
          const sandbox = {
            console: { log() {}, warn() {}, error() {} },
            CustomEvent: class {
              constructor(type) { this.type = type; }
            },
            setInterval() { return 0; },
            clearInterval() {},
            setTimeout(fn, ms) {
              const t = setTimeout(fn, ms);
              if (t && typeof t.unref === 'function') t.unref();
              return t;
            },
            clearTimeout,
            localStorage: {
              getItem(key) {
                return Object.prototype.hasOwnProperty.call(store, key)
                  ? store[key]
                  : null;
              },
              setItem(key, value) { store[key] = value; },
              removeItem(key) { delete store[key]; },
            },
            document: { getElementById() { return null; } },
            fetch(url, opts) {
              return new Promise((resolve, reject) => {
                if (opts && opts.method === 'POST') {
                  postCalls.push({ url, body: opts.body, resolve, reject });
                } else {
                  getCalls.push({ url, resolve, reject });
                }
              });
            },
          };
          sandbox.window = {
            appState: {
              independentAsrEnabled: false,
              independentAsrActive: true,
              voiceChatActive: true,
              voiceInputLifecycleState: 'active',
              voiceSessionStartEpoch: 10,
              voiceSettingsPendingUntilEpoch: null,
              pendingVoiceRouteIndependentAsr: null,
              settingsHydrated: false,
            },
            appConst: {},
            appUtils: { mapRenderQualityToFollowPerf() { return 'medium'; } },
            addEventListener(type, fn) { listeners.push({ type, fn }); },
            removeEventListener() {},
            dispatchEvent() {},
          };
          vm.createContext(sandbox);
          vm.runInContext(source, sandbox);
          const storage = listeners.filter((entry) => entry.type === 'storage');
          assert(storage.length === 1, 'module must register exactly one storage listener');
          return {
            postCalls,
            getCalls,
            S: sandbox.window.appState,
            fireStorage(value) {
              storage[0].fn({
                key: 'project_neko_settings',
                newValue: JSON.stringify(value),
              });
            },
            postedAsrValues() {
              return postCalls.map((call) => {
                try { return JSON.parse(call.body).independentAsrEnabled; }
                catch (_) { return undefined; }
              });
            },
          };
        }

        const tick = () => new Promise((resolve) => setImmediate(resolve));
        const NOW = Date.now();
        // The peer toggled a second ago; the server snapshot this window is
        // about to read still carries the decision from a minute ago.
        const PEER_WRITE_ID = NOW - 1000;
        const SERVER_DECISION_ID = NOW - 60000;

        function peerFlipSnapshot(writeId, value) {
          return {
            independentAsrEnabled: value,
            _sharedWriteMeta: {
              writeId,
              writerId: 'peerwindow',
              changedKeys: ['independentAsrEnabled'],
              hydrated: true,
              asrAuthoritative: true,
              pendingRecovery: false,
              asrDecision: { writeId, writerId: 'peerwindow', value },
            },
          };
        }

        function resolveStaleGet(getCall, options) {
          const withDecision = !(options && options.withoutDecisions);
          const body = {
            success: true,
            revision: 7,
            settings: { independentAsrEnabled: false },
            telemetryBranch: null,
          };
          if (withDecision) {
            body.decisions = {
              independentAsrEnabled: {
                writeId: SERVER_DECISION_ID,
                writerId: 'serverside',
                value: false,
              },
            };
          }
          getCall.resolve({
            ok: true,
            headers: {
              get(header) {
                return header === 'ETag' ? '"conversation-settings-7"' : null;
              },
            },
            json: async () => body,
          });
        }

        async function main() {
          // Scenario 1: the peer's flip reached localStorage before this window
          // loaded, so no storage event ever announces it and the key is not
          // dirty here. The restored decision tuple must still outrank the
          // server's older one.
          const ctx = makeContext(peerFlipSnapshot(PEER_WRITE_ID, true));
          assert(ctx.S.independentAsrEnabled === true, 'boot must load the peer flip');
          assert(ctx.getCalls.length === 1, 'boot must issue the settings GET');
          resolveStaleGet(ctx.getCalls[0]);
          await tick();
          await tick();
          await tick();
          assert(
            ctx.S.independentAsrEnabled === true,
            'the stale boot GET must not overwrite a non-dirty key the peer already decided'
          );
          assert(
            ctx.postedAsrValues().every((value) => value !== false),
            'the stale value must not be POSTed back over the peer decision'
          );

          // Scenario 1 negative: the SAME shape with a decision OLDER than the
          // server's is a genuine hydration, not a conflict -- the merge must
          // still apply the server value, or scenario 1 would pass vacuously.
          const stale = makeContext(
            peerFlipSnapshot(SERVER_DECISION_ID - 1000, true)
          );
          assert(stale.S.independentAsrEnabled === true, 'boot must load the local value');
          resolveStaleGet(stale.getCalls[0]);
          await tick();
          await tick();
          await tick();
          assert(
            stale.S.independentAsrEnabled === false,
            'a local decision older than the server tuple must still hydrate from the server'
          );

          // Scenario 2: the merge lands first and the peer's flip only arrives
          // afterwards. Adopting the server tuple must not pin this window.
          const late = makeContext(null);
          resolveStaleGet(late.getCalls[0]);
          await tick();
          await tick();
          await tick();
          assert(
            late.S.independentAsrEnabled === false,
            'the clean boot must merge the server value'
          );
          late.fireStorage(peerFlipSnapshot(NOW, true));
          assert(
            late.S.independentAsrEnabled === true,
            'a peer flip newer than the adopted server tuple must win after the merge'
          );

          // Scenario 3: a server with no decisions block at all carries no
          // ordering evidence, so it must never displace a restored local one.
          const legacyServer = makeContext(peerFlipSnapshot(PEER_WRITE_ID, true));
          resolveStaleGet(legacyServer.getCalls[0], { withoutDecisions: true });
          await tick();
          await tick();
          await tick();
          assert(
            legacyServer.S.independentAsrEnabled === true,
            'a decision-less server snapshot must not overwrite a restored local decision'
          );

          console.log('HARNESS_OK');
          process.exitCode = 0;
        }

        main().catch((err) => {
          console.error(err && err.stack ? err.stack : String(err));
          process.exitCode = 1;
        });
        """
    ).replace("__APP_SETTINGS_PATH__", json.dumps(str(APP_SETTINGS_PATH)))

    result = _run_settings_node_harness(harness)
    assert result.returncode == 0, (
        "boot-GET convergence harness failed\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "HARNESS_OK" in result.stdout

def test_unrelated_save_from_unhydrated_window_is_not_an_asr_toggle_harness():
    # Behavioral pin for the Codex P2 follow-up, driven end-to-end across two
    # real module instances: the WRITER's actual localStorage payload is fed to
    # the RECEIVER's storage listener, so the metadata contract is exercised,
    # not mocked.
    #
    # Scenario 1 (the bug): an unhydrated window saves one unrelated preference.
    # Its snapshot carries the boot-default independentAsrEnabled, which the
    # receiving window had already merged as `true` from the server. On the
    # pre-fix code the value difference alone read as an explicit toggle: the
    # receiver adopted `false`. Now the writer's metadata says only the
    # unrelated key changed, so the ASR value is ignored.
    # Scenario 2 (negative, dirty marking): same stale snapshot delivered to a
    # window whose boot GET is still unsettled — the ASR key must NOT enter the
    # dirty set, observable because unsettled POST bodies carry dirty keys only.
    # Scenario 3: a genuine cross-window toggle stays authoritative (hydration
    # marked, key dirtied so a stale merge cannot revert it).
    # Scenario 4: an already-superseded (older write id) snapshot is ignored.
    # Scenario 5: a metadata-less legacy payload keeps today's behaviour.
    harness = textwrap.dedent(
        """
        const fs = require('node:fs');
        const vm = require('node:vm');

        const source = fs.readFileSync(__APP_SETTINGS_PATH__, 'utf8');

        function assert(cond, msg) {
          if (!cond) throw new Error('ASSERT: ' + msg);
        }

        function makeContext(initialSettings = null) {
          const postCalls = [];
          const getCalls = [];
          const listeners = [];
          const timers = [];
          const writes = [];
          let savedShared = initialSettings ? JSON.stringify(initialSettings) : null;
          const sandbox = {
            console: { log() {}, warn() {}, error() {} },
            setInterval() { return 0; },
            clearInterval() {},
            setTimeout(fn, ms) {
              // Fully controllable: no pending timer can hold the process open.
              timers.push({ fn, ms });
              return { unref() {} };
            },
            clearTimeout() {},
            localStorage: {
              getItem(key) {
                return key === 'project_neko_settings' ? savedShared : null;
              },
              setItem(key, value) {
                if (key === 'project_neko_settings') savedShared = value;
                writes.push({ key, value });
              },
              removeItem() {},
            },
            document: { getElementById() { return null; } },
            fetch(url, opts) {
              return new Promise((resolve, reject) => {
                if (opts && opts.method === 'POST') {
                  postCalls.push({ url, body: opts.body, resolve, reject });
                } else {
                  getCalls.push({ url, resolve, reject });
                }
              });
            },
          };
          sandbox.window = {
            appState: { independentAsrEnabled: false, settingsHydrated: false },
            appConst: {},
            appUtils: { mapRenderQualityToFollowPerf() { return 'medium'; } },
            addEventListener(type, fn) { listeners.push({ type, fn }); },
            removeEventListener() {},
            dispatchEvent() {},
          };
          vm.createContext(sandbox);
          vm.runInContext(source, sandbox);
          const storage = listeners.filter((entry) => entry.type === 'storage');
          assert(storage.length === 1, 'module must register exactly one storage listener');
          return {
            postCalls,
            getCalls,
            S: sandbox.window.appState,
            win: sandbox.window,
            mod: sandbox.window.appSettings,
            sharedWrites() {
              return writes.filter((w) => w.key === 'project_neko_settings');
            },
            lastSharedWrite() {
              const shared = writes.filter((w) => w.key === 'project_neko_settings');
              assert(shared.length > 0, 'the module must persist the shared settings snapshot');
              return shared[shared.length - 1].value;
            },
            fireStorage(newValue) {
              storage[0].fn({ key: 'project_neko_settings', newValue });
            },
            fireGateTimeout() {
              assert(timers.length >= 1, 'a bounded gate timer must be armed');
              timers.shift().fn();
            },
          };
        }

        const okPost = { ok: true, json: async () => ({ success: true }) };
        const tick = () => new Promise((resolve) => setImmediate(resolve));

        async function hydrateFromServer(ctx, settings, decisions = null) {
          assert(ctx.getCalls.length === 1, 'boot must issue the settings GET');
          ctx.getCalls[0].resolve({
            ok: true,
            json: async () => ({
              success: true,
              settings,
              decisions: decisions || {},
              telemetryBranch: null,
            }),
          });
          await tick();
          await tick();
          assert(ctx.S.settingsHydrated === true, 'a successful GET must hydrate the window');
          while (ctx.postCalls.length) {
            ctx.postCalls.shift().resolve(okPost);
            await tick();
          }
        }

        async function main() {
          // ---- Scenario 1: unrelated save from an UNHYDRATED window ----
          const receiver = makeContext();
          await hydrateFromServer(receiver, {
            independentAsrEnabled: true,
            voiceInputResourceOptimizationEnabled: false,
          });
          assert(receiver.S.independentAsrEnabled === true, 'receiver merged the server ASR value');
          assert(
            receiver.S.voiceInputResourceOptimizationEnabled === false,
            'receiver merged the server optimization value'
          );

          const writer = makeContext();      // boot GET left pending -> unhydrated
          writer.win.mergeMessagesEnabled = true;
          writer.mod.saveSettings();         // an UNRELATED preference
          const stalePayload = writer.lastSharedWrite();
          const staleParsed = JSON.parse(stalePayload);
          assert(
            staleParsed.independentAsrEnabled === false,
            'saveSettings still copies the ASR key into every snapshot (that is the trap)'
          );
          assert(
            staleParsed.voiceInputResourceOptimizationEnabled === true,
            'saveSettings still copies the optimization key into every snapshot (that is the trap)'
          );
          const receiverPostsBefore = receiver.postCalls.length;
          receiver.fireStorage(stalePayload);
          assert(
            receiver.S.independentAsrEnabled === true,
            'the hydrated ASR value must survive an unrelated save from an unhydrated window'
          );
          assert(
            receiver.S.voiceInputResourceOptimizationEnabled === false,
            'the hydrated optimization value must survive an unrelated save from an unhydrated window'
          );
          assert(
            receiver.S.mergeMessagesEnabled === true,
            'every other shared key must still sync across windows'
          );
          assert(
            receiver.postCalls.length === receiverPostsBefore,
            'the receiving window must never POST from the storage listener'
          );

          // The metadata that made the decision possible.
          const staleMeta = staleParsed._sharedWriteMeta;
          assert(staleMeta && typeof staleMeta.writeId === 'number', 'the write must carry metadata');
          assert(
            staleMeta.changedKeys.indexOf('mergeMessagesEnabled') !== -1,
            'the explicitly changed key must be declared'
          );
          assert(
            staleMeta.changedKeys.indexOf('independentAsrEnabled') === -1,
            'an unrelated save must NOT declare the ASR key as user-changed'
          );
          assert(
            staleMeta.changedKeys.indexOf('voiceInputResourceOptimizationEnabled') === -1,
            'an unrelated save must NOT declare the optimization key as user-changed'
          );
          assert(staleMeta.hydrated === false, 'the writer had not merged the server settings yet');

          // ---- Scenario 2 (negative): the ASR key must not be dirtied ----
          // Observability: while the boot GET is unsettled the POST body is
          // restricted to the user-dirty keys, so a wrongly dirtied ASR key
          // would show up there.
          const pending = makeContext();     // boot GET stays pending
          pending.win.mergeMessagesEnabled = true;
          pending.mod.saveSettings();        // hydrates this window, dirties ONE key
          assert(pending.S.settingsHydrated === true, 'a user change hydrates synchronously');
          pending.fireGateTimeout();
          await tick();
          await tick();
          assert(pending.postCalls.length === 1, 'the bounded gate must release the POST');
          const dirtyBody1 = JSON.parse(pending.postCalls[0].body);
          assert(
            !('independentAsrEnabled' in dirtyBody1),
            'baseline: the untouched ASR key is not dirty yet'
          );
          pending.postCalls[0].resolve(okPost);
          await tick();
          // This window already holds the authoritative ASR value; its own GET
          // has not landed, so set the state the listener reads directly.
          pending.S.independentAsrEnabled = true;

          pending.fireStorage(stalePayload);
          assert(
            pending.S.independentAsrEnabled === true,
            'the stale snapshot must not overwrite the authoritative value here either'
          );
          const pp = pending.mod.syncSettingsToServer();  // periodic-style: no dirty marking
          await tick();
          await tick();
          assert(
            pending.postCalls.length === 1,
            'the acknowledged local key and incidental ASR copy leave no pending POST'
          );
          await pp;

          // ---- Scenario 3: a genuine cross-window toggle stays authoritative ----
          const toggler = makeContext();
          await hydrateFromServer(toggler, { independentAsrEnabled: false });
          // Mirror app-audio-capture.js: local persist first, then the POST.
          toggler.S.independentAsrEnabled = true;
          toggler.mod.saveSettings({ skipServerSync: true });
          const togglePayload = toggler.lastSharedWrite();
          const toggleMeta = JSON.parse(togglePayload)._sharedWriteMeta;
          assert(
            toggleMeta.changedKeys.indexOf('independentAsrEnabled') !== -1,
            'a real toggle must declare the ASR key as explicitly changed'
          );

          const receiver2 = makeContext();   // boot GET still pending
          receiver2.fireStorage(togglePayload);
          assert(receiver2.S.independentAsrEnabled === true, 'a real toggle must be applied');
          assert(
            receiver2.S.settingsHydrated === true,
            'a real toggle must arm the start_session handshake stamp'
          );
          assert(receiver2.postCalls.length === 0, 'still no POST from the receiving window');
          // The key must be dirty: the stale server merge cannot revert it.
          receiver2.getCalls[0].resolve({
            ok: true,
            json: async () => ({
              success: true,
              settings: { independentAsrEnabled: false },
              telemetryBranch: null,
            }),
          });
          await tick();
          await tick();
          assert(
            receiver2.S.independentAsrEnabled === true,
            'the flipped key must be dirty so the stale merge preserves it'
          );

          // ---- Scenario 4: a superseded (older) write is ignored ----
          const replay = JSON.parse(togglePayload);
          replay.independentAsrEnabled = false;
          replay._sharedWriteMeta = {
            writeId: toggleMeta.writeId - 1,
            changedKeys: ['independentAsrEnabled'],
            hydrated: true,
          };
          receiver2.fireStorage(JSON.stringify(replay));
          assert(
            receiver2.S.independentAsrEnabled === true,
            'an already-superseded write must not re-flip the route'
          );

          // ---- Scenario 5: metadata-less legacy payload keeps today's behaviour ----
          const legacy = makeContext();
          legacy.fireStorage(JSON.stringify({ independentAsrEnabled: true }));
          assert(legacy.S.independentAsrEnabled === true, 'legacy payloads still apply the value');
          assert(
            legacy.S.settingsHydrated === true,
            'legacy payloads keep the value-difference authority fallback'
          );
          assert(legacy.postCalls.length === 0, 'legacy fallback still never POSTs from the listener');

          // ---- Scenario 6: an authoritative server-merge broadcast updates
          // the decision tuple even though changedKeys is intentionally empty.
          const sibling = makeContext();
          await hydrateFromServer(sibling, { independentAsrEnabled: false });
          sibling.S.independentAsrEnabled = true;
          sibling.mod.saveSettings({ skipServerSync: true });
          const localToggleMeta = JSON.parse(sibling.lastSharedWrite())._sharedWriteMeta;
          const serverDecision = {
            writeId: localToggleMeta.asrDecision.writeId + 10,
            writerId: 'server-legacy',
            value: false,
          };
          sibling.fireStorage(JSON.stringify({
            independentAsrEnabled: false,
            _sharedWriteMeta: {
              writeId: serverDecision.writeId,
              writerId: 'server-window',
              changedKeys: [],
              hydrated: true,
              asrAuthoritative: true,
              asrDecision: serverDecision,
            },
          }));
          assert(
            sibling.S.independentAsrEnabled === false,
            'the authoritative server broadcast must replace the old local toggle'
          );
          sibling.win.mergeMessagesEnabled = true;
          sibling.mod.saveSettings({ skipServerSync: true });
          const resharedMeta = JSON.parse(sibling.lastSharedWrite())._sharedWriteMeta;
          assert(
            JSON.stringify(resharedMeta.asrDecision) === JSON.stringify(serverDecision),
            'an unrelated save must retain the accepted server decision tuple'
          );

          // ---- Scenario 7: a persisted server-merge tuple survives reload.
          // changedKeys is intentionally empty because this is authority from
          // the server, not a new browser user action.
          const bootDecision = {
            writeId: Date.now() + 1000,
            writerId: 'server-ahead',
            value: false,
          };
          const reloaded = makeContext({
            independentAsrEnabled: false,
            _sharedWriteMeta: {
              writeId: bootDecision.writeId - 1,
              writerId: 'merging-window',
              changedKeys: [],
              hydrated: true,
              asrAuthoritative: true,
              asrDecision: bootDecision,
            },
          });
          reloaded.win.mergeMessagesEnabled = true;
          reloaded.mod.saveSettings({ skipServerSync: true });
          const afterReloadMeta = JSON.parse(reloaded.lastSharedWrite())._sharedWriteMeta;
          assert(
            JSON.stringify(afterReloadMeta.asrDecision) === JSON.stringify(bootDecision),
            'reload must restore and retain a matching server-merge ASR tuple'
          );
          reloaded.S.independentAsrEnabled = true;
          reloaded.mod.saveSettings({ skipServerSync: true });
          const nextLocalDecision = JSON.parse(
            reloaded.lastSharedWrite()
          )._sharedWriteMeta.asrDecision;
          assert(
            nextLocalDecision.writeId > bootDecision.writeId,
            'the first local toggle after reload must supersede the persisted server tuple'
          );

          // ---- Scenario 8: pending recovery writes terminate instead of
          // bouncing between two windows with different pending keys.
          const pendingA = makeContext();
          pendingA.win.focusModeEnabled = true;
          pendingA.mod.saveSettings();
          const pendingAPayload = pendingA.lastSharedWrite();

          const pendingB = makeContext({
            independentAsrEnabled: false,
            focusModeEnabled: false,
          });
          pendingB.win.mergeMessagesEnabled = true;
          pendingB.mod.saveSettings();
          const pendingBPayload = pendingB.lastSharedWrite();

          pendingA.fireStorage(pendingBPayload);
          const recoveryPayload = pendingA.lastSharedWrite();
          const recoveryMeta = JSON.parse(recoveryPayload)._sharedWriteMeta;
          assert(
            recoveryMeta.pendingRecovery === true,
            'the reasserted full snapshot must be marked as pending recovery'
          );
          const writesBeforeRecovery = pendingB.sharedWrites().length;
          pendingB.fireStorage(recoveryPayload);
          assert(
            pendingB.sharedWrites().length === writesBeforeRecovery,
            'a pending window must not answer a recovery with another recovery write'
          );
          assert(
            pendingB.S.mergeMessagesEnabled === true,
            'the receiving window must still preserve its pending runtime value'
          );
          assert(
            JSON.parse(pendingAPayload)._sharedWriteMeta.pendingRecovery !== true,
            'ordinary user writes must not be marked as recovery'
          );

          // ---- Scenario 9: a merge envelope minted later does not make a
          // stale field newer than an explicit edit the sender never observed.
          const editor = makeContext({
            independentAsrEnabled: false,
            focusModeEnabled: false,
          });
          editor.win.focusModeEnabled = true;
          editor.mod.saveSettings({ skipServerSync: true });
          const explicitPayload = editor.lastSharedWrite();
          const explicitMeta = JSON.parse(explicitPayload)._sharedWriteMeta;

          const observer = makeContext({
            independentAsrEnabled: false,
            focusModeEnabled: false,
          });
          observer.fireStorage(explicitPayload);
          assert(observer.S.focusModeEnabled === true, 'observer accepts the explicit edit');

          const staleMerger = makeContext({
            independentAsrEnabled: false,
            focusModeEnabled: false,
          });
          staleMerger.mod.saveSettings({
            skipServerSync: true,
            serverMerged: true,
          });
          const staleMerge = JSON.parse(staleMerger.lastSharedWrite());
          staleMerge._sharedWriteMeta.writeId = explicitMeta.writeId + 100;
          assert(
            Object.keys(staleMerge._sharedWriteMeta.knownKeyWrites).length === 0,
            'the stale sender must declare that it never observed the edit'
          );
          observer.fireStorage(JSON.stringify(staleMerge));
          assert(
            observer.S.focusModeEnabled === true,
            'a high envelope id must not roll back a newer per-key edit'
          );

          const mergeAfterEdit = makeContext({
            independentAsrEnabled: false,
            focusModeEnabled: false,
          });
          mergeAfterEdit.fireStorage(explicitPayload);
          mergeAfterEdit.getCalls[0].resolve({
            ok: true,
            json: async () => ({
              success: true,
              settings: { independentAsrEnabled: false, focusModeEnabled: false },
              telemetryBranch: null,
            }),
          });
          await tick();
          await tick();
          assert(
            mergeAfterEdit.S.focusModeEnabled === true,
            'a boot GET started before the cross-window edit must preserve that edit'
          );

          // ---- Scenario 10: a late recovery envelope cannot roll back a
          // newer explicit decision for the same key.
          const newerEditor = makeContext(JSON.parse(explicitPayload));
          newerEditor.win.focusModeEnabled = false;
          newerEditor.mod.saveSettings({ skipServerSync: true });
          const newerPayload = newerEditor.lastSharedWrite();
          const newerMeta = JSON.parse(newerPayload)._sharedWriteMeta;
          observer.fireStorage(newerPayload);
          assert(observer.S.focusModeEnabled === false, 'observer accepts the newer edit');

          const lateRecovery = JSON.parse(explicitPayload);
          lateRecovery._sharedWriteMeta = {
            ...lateRecovery._sharedWriteMeta,
            writeId: newerMeta.writeId + 100,
            changedKeys: [],
            pendingRecovery: true,
          };
          observer.fireStorage(JSON.stringify(lateRecovery));
          assert(
            observer.S.focusModeEnabled === false,
            'a fresh recovery envelope must not outrank its older per-key provenance'
          );

          // ---- Scenario 11: a genuine optimization toggle still propagates ----
          const optimizationWriter = makeContext();
          await hydrateFromServer(optimizationWriter, {
            independentAsrEnabled: false,
            voiceInputResourceOptimizationEnabled: true,
          });
          optimizationWriter.S.voiceInputResourceOptimizationEnabled = false;
          optimizationWriter.mod.saveSettings({ skipServerSync: true });
          const optimizationPayload = optimizationWriter.lastSharedWrite();
          const optimizationMeta = JSON.parse(optimizationPayload)._sharedWriteMeta;
          assert(
            optimizationMeta.changedKeys.indexOf(
              'voiceInputResourceOptimizationEnabled'
            ) !== -1,
            'a real optimization toggle must be declared explicitly'
          );

          const optimizationReceiver = makeContext();
          await hydrateFromServer(optimizationReceiver, {
            independentAsrEnabled: false,
            voiceInputResourceOptimizationEnabled: true,
          });
          optimizationReceiver.fireStorage(optimizationPayload);
          assert(
            optimizationReceiver.S.voiceInputResourceOptimizationEnabled === false,
            'a real optimization toggle must apply across windows'
          );

          // ---- Scenario 7: concurrent optimization toggles converge ----
          // Each window writes before observing the other. Freshness against
          // received writes cannot order either window's own pending choice;
          // the per-key decision tuple must select the same winner on both.
          const optimizationA = makeContext();
          await hydrateFromServer(optimizationA, {
            independentAsrEnabled: false,
            voiceInputResourceOptimizationEnabled: true,
          });
          optimizationA.S.voiceInputResourceOptimizationEnabled = false;
          optimizationA.mod.saveSettings({ skipServerSync: true });
          const optimizationPayloadA = optimizationA.lastSharedWrite();

          const optimizationB = makeContext();
          await hydrateFromServer(optimizationB, {
            independentAsrEnabled: false,
            voiceInputResourceOptimizationEnabled: false,
          });
          optimizationB.S.voiceInputResourceOptimizationEnabled = true;
          optimizationB.mod.saveSettings({ skipServerSync: true });
          const optimizationPayloadB = optimizationB.lastSharedWrite();

          const parsedA = JSON.parse(optimizationPayloadA);
          const parsedB = JSON.parse(optimizationPayloadB);
          const decisionA = parsedA._sharedWriteMeta.optimizationDecision;
          const decisionB = parsedB._sharedWriteMeta.optimizationDecision;
          assert(decisionA && decisionB, 'each explicit optimization write must carry its decision tuple');
          const aWins = decisionA.writeId > decisionB.writeId
            || (
              decisionA.writeId === decisionB.writeId
              && decisionA.writerId > decisionB.writerId
            );
          const winningValue = aWins
            ? parsedA.voiceInputResourceOptimizationEnabled
            : parsedB.voiceInputResourceOptimizationEnabled;

          optimizationA.fireStorage(optimizationPayloadB);
          optimizationB.fireStorage(optimizationPayloadA);
          assert(
            optimizationA.S.voiceInputResourceOptimizationEnabled === winningValue,
            'window A must converge on the winning optimization choice'
          );
          assert(
            optimizationB.S.voiceInputResourceOptimizationEnabled === winningValue,
            'window B must converge on the winning optimization choice'
          );

          // ---- Scenario 8: a real return to the restored value is fresh ----
          const restoredDecision = {
            writeId: 1,
            writerId: 'restored-writer',
            value: false,
          };
          const rebound = makeContext({
            independentAsrEnabled: false,
            voiceInputResourceOptimizationEnabled: false,
            _sharedWriteMeta: {
              writeId: 1,
              writerId: 'restored-writer',
              changedKeys: [
                'independentAsrEnabled',
                'voiceInputResourceOptimizationEnabled',
              ],
              asrDecision: restoredDecision,
              optimizationDecision: restoredDecision,
              optimizationDecisionPendingSync: false,
            },
          });
          await hydrateFromServer(rebound, {
            independentAsrEnabled: true,
            voiceInputResourceOptimizationEnabled: true,
          }, {
            independentAsrEnabled: {
              writeId: 2,
              writerId: 'server-winner',
              value: true,
            },
          });
          assert(
            rebound.S.independentAsrEnabled === true
              && rebound.S.voiceInputResourceOptimizationEnabled === true,
            'server decisions must apply before testing a return to the restored value'
          );
          rebound.S.independentAsrEnabled = false;
          rebound.S.voiceInputResourceOptimizationEnabled = false;
          rebound.mod.saveSettings({ skipServerSync: true });
          const reboundMeta = JSON.parse(
            rebound.lastSharedWrite()
          )._sharedWriteMeta;
          assert(
            reboundMeta.asrDecision.writeId === reboundMeta.writeId
              && reboundMeta.asrDecision.writerId === reboundMeta.writerId,
            'returning to a restored ASR value is a fresh user decision'
          );
          assert(
            reboundMeta.optimizationDecision.writeId === reboundMeta.writeId
              && reboundMeta.optimizationDecision.writerId === reboundMeta.writerId,
            'returning to a restored optimization value is a fresh user decision'
          );

          // ---- Scenario 9: an invalid optimization decision id cannot poison
          // later local choices by permanently outranking the browser clock.
          const poisonedOptimization = makeContext({
            voiceInputResourceOptimizationEnabled: false,
            _sharedWriteMeta: {
              writeId: 1,
              writerId: 'poisoned-writer',
              changedKeys: ['voiceInputResourceOptimizationEnabled'],
              optimizationDecision: {
                writeId: Number.MAX_SAFE_INTEGER,
                writerId: 'poisoned-writer',
                value: false,
              },
              optimizationDecisionPendingSync: false,
            },
          });
          poisonedOptimization.S.voiceInputResourceOptimizationEnabled = true;
          poisonedOptimization.mod.saveSettings({ skipServerSync: true });
          const recoveredOptimizationMeta = JSON.parse(
            poisonedOptimization.lastSharedWrite()
          )._sharedWriteMeta;
          assert(
            recoveredOptimizationMeta.optimizationDecision
              && recoveredOptimizationMeta.optimizationDecision.value === true
              && recoveredOptimizationMeta.optimizationDecision.writeId
                === recoveredOptimizationMeta.writeId,
            'invalid optimization decision ids must not block a fresh local choice'
          );

          console.log('HARNESS_OK');
          // Every sandbox timer is harness-controlled, so the process exits
          // naturally once main() returns and piped stdout is fully flushed.
          process.exitCode = 0;
        }

        main().catch((err) => {
          console.error(err && err.stack ? err.stack : String(err));
          process.exitCode = 1;
        });
        """
    ).replace("__APP_SETTINGS_PATH__", json.dumps(str(APP_SETTINGS_PATH)))

    result = _run_settings_node_harness(harness)
    assert result.returncode == 0, (
        "unrelated-save-from-unhydrated-window harness failed\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "HARNESS_OK" in result.stdout

def test_never_settling_get_posts_only_dirty_keys_harness():
    # Behavioral pin for the round-15 fix, driving the real module with a
    # controllable fetch AND a controllable gate timer. Scenario 1: the boot GET
    # never settles, the user changes ONE unrelated preference, and the bound
    # elapses — the POST must still go out (liveness) but must carry only the
    # changed key, so the server-persisted preferences this client never read
    # survive; when the slow GET finally lands, its (intact) values hydrate the
    # untouched keys and the writeback converges. On the pre-fix code the body
    # was the full boot snapshot and independentAsrEnabled=false clobbered the
    # persisted true. Scenario 2 (negative): no dirty keys -> no POST at all.
    # Scenario 3: the normal fast-GET flow still posts the full snapshot.
    # Scenario 4: the ASR toggle flow still persists the user's choice even
    # when the bound elapses.
    harness = textwrap.dedent(
        """
        const fs = require('node:fs');
        const vm = require('node:vm');

        const source = fs.readFileSync(__APP_SETTINGS_PATH__, 'utf8');

        function assert(cond, msg) {
          if (!cond) throw new Error('ASSERT: ' + msg);
        }

        function makeContext() {
          const postCalls = [];
          const getCalls = [];
          const timers = [];
          const sandbox = {
            console: { log() {}, warn() {}, error() {} },
            setInterval() { return 0; },
            clearInterval() {},
            setTimeout(fn, ms) {
              // Fully controllable: the bound never elapses on its own, so the
              // harness decides when the timeout wins the gate race (and no
              // pending timer can hold the process open).
              timers.push({ fn, ms });
              return { unref() {} };
            },
            clearTimeout() {},
            localStorage: { getItem() { return null; }, setItem() {}, removeItem() {} },
            document: { getElementById() { return null; } },
            fetch(url, opts) {
              return new Promise((resolve, reject) => {
                if (opts && opts.method === 'POST') {
                  postCalls.push({ url, body: opts.body, resolve, reject });
                } else {
                  getCalls.push({ url, resolve, reject });
                }
              });
            },
          };
          sandbox.window = {
            appState: { independentAsrEnabled: false, settingsHydrated: false },
            appConst: {},
            appUtils: { mapRenderQualityToFollowPerf() { return 'medium'; } },
            addEventListener() {},
            removeEventListener() {},
            dispatchEvent() {},
          };
          vm.createContext(sandbox);
          vm.runInContext(source, sandbox);
          return {
            postCalls,
            getCalls,
            S: sandbox.window.appState,
            win: sandbox.window,
            mod: sandbox.window.appSettings,
            fireGateTimeout() {
              assert(timers.length === 1, 'exactly one bounded gate timer must be armed');
              assert(timers[0].ms === 3000, 'the gate bound must stay the 3s constant');
              timers.shift().fn();
            },
          };
        }

        const okPost = { ok: true, json: async () => ({ success: true }) };
        const tick = () => new Promise((resolve) => setImmediate(resolve));

        async function main() {
          // Scenario 1: slow (never-settling) GET + one unrelated user change.
          const ctx = makeContext();
          assert(ctx.getCalls.length === 1, 'boot must issue the settings GET');

          ctx.win.mergeMessagesEnabled = true; // the settings-popup mirror
          ctx.mod.saveSettings();              // full user path -> userInitiated POST
          assert(ctx.S.settingsHydrated === true, 'a user change still hydrates synchronously');
          await tick();
          assert(ctx.postCalls.length === 0, 'the POST waits for the gate while the GET is pending');

          ctx.fireGateTimeout();
          await tick();
          await tick();
          assert(ctx.postCalls.length === 1, 'the bound must still release the POST (liveness)');
          const body1 = JSON.parse(ctx.postCalls[0].body);
          assert(body1.mergeMessagesEnabled === true, 'the dirty key must be persisted');
          assert(
            Object.keys(body1).length === 1,
            'a timed-out gate must post ONLY the dirty keys, got: ' + JSON.stringify(body1)
          );
          assert(
            !('independentAsrEnabled' in body1),
            'the untouched ASR preference must not be overwritten by this boot default'
          );
          assert(
            !('proactiveChatEnabled' in body1),
            'no untouched preference may ride along in the timed-out body'
          );

          // The slow GET now lands. Because the partial POST left them alone,
          // the server values for untouched keys are still the persisted ones.
          ctx.postCalls[0].resolve(okPost);
          ctx.getCalls[0].resolve({
            ok: true,
            json: async () => ({
              success: true,
              settings: { independentAsrEnabled: true, mergeMessagesEnabled: false },
              telemetryBranch: null,
            }),
          });
          await tick();
          await tick();
          assert(ctx.S.independentAsrEnabled === true, 'the untouched key hydrates from the surviving server value');
          assert(ctx.S.mergeMessagesEnabled === true, 'the dirty key survives the late merge');
          assert(ctx.postCalls.length === 2, 'the merge writeback POST follows');
          const body2 = JSON.parse(ctx.postCalls[1].body);
          assert(
            Object.keys(body2).length > 1,
            'once the GET settled, full snapshots resume and converge the server'
          );
          assert(body2.independentAsrEnabled === true, 'the writeback carries the server ASR value');
          assert(body2.mergeMessagesEnabled === true, 'the writeback carries the user change');
          ctx.postCalls[1].resolve(okPost);
          await tick();

          // Scenario 2 (negative): nothing dirty while the GET is unsettled —
          // a periodic-style sync must not write pre-merge values at all, and
          // its promise must still resolve (never-rejecting sync contract).
          const ctx2 = makeContext();
          const pp = ctx2.mod.syncSettingsToServer();
          await tick();
          ctx2.fireGateTimeout();
          await tick();
          await tick();
          assert(ctx2.postCalls.length === 0, 'no dirty key means no POST while the GET is unsettled');
          await pp;

          // Scenario 3: the normal fast-GET flow is unchanged — the merge
          // settles before the bound, so the user POST carries the FULL
          // snapshot (server truth for untouched keys included).
          const ctx3 = makeContext();
          ctx3.getCalls[0].resolve({
            ok: true,
            json: async () => ({
              success: true,
              settings: { independentAsrEnabled: true },
              telemetryBranch: null,
            }),
          });
          await tick();
          await tick();
          assert(ctx3.postCalls.length === 1, 'the merge writeback POST goes out first');
          ctx3.postCalls[0].resolve(okPost);
          await tick();
          await tick();
          ctx3.win.mergeMessagesEnabled = true;
          ctx3.mod.saveSettings();
          await tick();
          await tick();
          assert(ctx3.postCalls.length === 2, 'the fast-GET user POST goes out');
          const body3 = JSON.parse(ctx3.postCalls[1].body);
          assert(Object.keys(body3).length > 1, 'a settled GET keeps posting the full snapshot');
          assert(body3.independentAsrEnabled === true, 'the full snapshot carries the merged server value');
          assert(body3.mergeMessagesEnabled === true, 'the full snapshot carries the user change');

          // Scenario 4: the ASR toggle flow still persists the user's choice
          // when the bound elapses (the toggled key is dirty).
          const ctx4 = makeContext();
          ctx4.S.independentAsrEnabled = true;
          ctx4.mod.saveSettings({ skipServerSync: true });
          const p4 = ctx4.mod.syncSettingsToServer({ userInitiated: true });
          assert(ctx4.S.settingsHydrated === true, 'the toggle hydrates synchronously at call time');
          await tick();
          assert(ctx4.postCalls.length === 0, 'the toggle POST is gated behind the pending GET');
          ctx4.fireGateTimeout();
          await tick();
          await tick();
          assert(ctx4.postCalls.length === 1, 'the toggle POST goes out on the bound');
          const body4 = JSON.parse(ctx4.postCalls[0].body);
          assert(body4.independentAsrEnabled === true, 'the toggle POST carries the user choice');
          assert(
            Object.keys(body4).length === 1,
            'the toggle POST carries nothing else, got: ' + JSON.stringify(body4)
          );
          ctx4.postCalls[0].resolve(okPost);
          await p4;

          console.log('HARNESS_OK');
          // No live timers remain (the harness owns setTimeout), so the process
          // exits naturally once main() returns and piped stdout is flushed.
          process.exitCode = 0;
        }

        main().catch((err) => {
          console.error(err && err.stack ? err.stack : String(err));
          process.exitCode = 1;
        });
        """
    ).replace("__APP_SETTINGS_PATH__", json.dumps(str(APP_SETTINGS_PATH)))

    result = _run_settings_node_harness(harness)
    assert result.returncode == 0, (
        "never-settling-GET dirty-only harness failed\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "HARNESS_OK" in result.stdout

def test_normal_teardown_paths_reset_independent_asr_route_flags():
    # ASR_INDEPENDENT_READY sets S.independentAsrActive; ordinary user stop,
    # server-side session end, and socket close must reset it (and the
    # provider) too, or the mic settings hint keeps claiming independent ASR
    # is active until some later route status arrives.
    websocket_source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")
    capture_source = APP_AUDIO_CAPTURE_PATH.read_text(encoding="utf-8")

    stop_block = capture_source.split("function stopRecording(options)", 1)[1].split(
        "function startMicVolumeVisualization",
        1,
    )[0]
    pre_early_return = stop_block.split("if (!S.isRecording) return;", 1)[0]
    assert "S.independentAsrActive = false;" in pre_early_return
    assert "S.independentAsrProvider = '';" in pre_early_return
    assert pre_early_return.index("window.removeExternalAsrPreview();") < pre_early_return.index(
        "S.independentAsrActive = false;"
    )

    session_ended_block = websocket_source.split(
        "// -------- session_ended_by_server --------",
        1,
    )[1].split("// -------- reload_page --------", 1)[0]
    assert "S.independentAsrActive = false;" in session_ended_block
    assert "S.independentAsrProvider = '';" in session_ended_block
    # Reset must not hide behind the isRecording branch: a paused mic keeps
    # S.isRecording false while the flags are still set.
    assert session_ended_block.index("S.independentAsrActive = false;") < session_ended_block.index(
        "if (S.isRecording)"
    )

    onclose_block = websocket_source.split("// ---- onclose ----", 1)[1].split(
        "// ---- onerror ----",
        1,
    )[0]
    stale_guard, current_close = onclose_block.split(
        "console.log(window.t('console.websocketClosed'));", 1
    )
    # Negative: a stale socket's onclose must not touch the live session flags.
    assert "S.independentAsrActive = false;" not in stale_guard
    assert "S.independentAsrActive = false;" in current_close
    assert "S.independentAsrProvider = '';" in current_close
    assert current_close.index("S.independentAsrActive = false;") < current_close.index(
        "if (S.isRecording || window.isMicStarting)"
    )

def test_blocked_greeting_check_retries_without_home_tutorial_state():
    source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")

    blocked_branch = source.split("if (_isGreetingCheckBlocked()) {", 1)[1].split(
        "try {",
        1,
    )[0]
    assert "sendHomeTutorialState(" not in blocked_branch
    assert "_scheduleGreetingCheckRetry();" in blocked_branch

def test_goodbye_blocks_stale_audio_session_started():
    source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")

    stale_audio_guard = source.split("// -------- session_started --------", 1)[1].split(
        "console.log(window.t('console.sessionStartedReceived')",
        1,
    )[0]

    assert "response.input_mode !== 'text'" in stale_audio_guard
    assert "window.isNekoGoodbyeModeActive()" in stale_audio_guard
    assert "window.cancelPendingSessionStart('Voice start cancelled by goodbye');" in stale_audio_guard
    assert "S.socket.send(JSON.stringify({ action: 'end_session' }));" in stale_audio_guard
    assert "return;" in stale_audio_guard

def test_asr_authority_is_per_key_not_granted_by_unrelated_setting_change():
    # Codex P2. syncSettingsToServer({userInitiated:true}) marks the GLOBAL
    # S.settingsHydrated for every user action, including ones that never touch
    # the ASR key (settings popup toggles, subtitle toggles, the chat-window
    # translate toggle). With a pending or permanently failing boot GET,
    # S.independentAsrEnabled is still the boot default false at that moment, so
    # a global-only gate would let the next start_session stamp false over the
    # backend's persisted true. Authority for that one key must therefore be
    # tracked separately and granted only by explicit ASR edits/cross-window
    # choices or an authoritative server snapshot.
    settings_source = APP_SETTINGS_PATH.read_text(encoding="utf-8")

    user_gate = _block_after(settings_source, "if (userInitiated) {")
    assert "S.settingsHydrated = true;" in user_gate
    # The per-key mark inside the userInitiated gate must be conditional on the
    # ASR key actually being dirty — an unconditional mark here is the bug.
    assert (
        "if (_dirtySettingsKeys.has('independentAsrEnabled')) "
        "S.independentAsrAuthoritative = true;" in user_gate
    ), "ASR authority must be granted only when the user change touched that key"

    # (1) A merged server GET grants authority.
    merge_block = settings_source.split(
        "const mergeSettled = loadSettingsFromServer().then(serverResult => {",
        1,
    )[1]
    assert "S.independentAsrAuthoritative = true;" in merge_block.split(
        "startPeriodicSync();", 1
    )[0]

    # (2) A full snapshot from a successful partial POST or 412 grants the same
    # server authority when the boot GET was unavailable.
    snapshot_merge = _block_after(
        settings_source, "function _mergeConversationSettingsSnapshot(data, preservedKeys) {"
    )
    assert "S.independentAsrAuthoritative = true;" in snapshot_merge

    # (3) A cross-window ASR flip grants authority, next to the dirty-key add.
    cross_window = _block_after(
        settings_source, "_dirtySettingsKeys.add('independentAsrEnabled');"
    )
    assert "S.independentAsrAuthoritative = true;" in cross_window

    # No unrelated path grants it: exactly these four assignment sites (the
    # conditional local-user gate plus the three authoritative sources above).
    assert settings_source.count("S.independentAsrAuthoritative = true;") == 4

def test_blocked_route_latch_blocks_game_exit_microphone_resume():
    # The teardown above is skipped while the game STT gate holds the
    # microphone, and BLOCKED is never re-sent, so the game-exit resume path
    # would reopen the mic onto a still-fail-closed route. A sticky latch
    # closes that, and is cleared wherever a fresh route can exist again.
    websocket_source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")
    state_source = APP_STATE_PATH.read_text(encoding="utf-8")

    assert "voiceInputRouteBlocked: false," in state_source

    teardown = websocket_source.split(
        "function tearDownBlockedVoiceRoute() {", 1
    )[1].split("\n    }", 1)[0]
    assert "S.voiceInputRouteBlocked = true;" in teardown

    resume = websocket_source.split("if (shouldResumeAudio && wasRecording", 1)[1].split(
        ")", 1
    )[0]
    assert "S.voiceInputRouteBlocked !== true" in resume

    # Cleared only where a fresh or healthy route really exists: a provider
    # that came READY, the DISABLED (native) route, and user intent to start a
    # new voice session. Deliberately NOT in the session_started handler --
    # lifecycle.py runs the route decision BEFORE sending that ack, so clearing
    # there would wipe the current session's own verdict.
    assert websocket_source.count("S.voiceInputRouteBlocked = false;") == 3
    started_handler = websocket_source.split(
        "S.isTextSessionActive = response.input_mode === 'text';", 1
    )[1].split("var _tiaStarted", 1)[0]
    assert "S.voiceInputRouteBlocked = false;" not in started_handler

def test_session_started_only_settles_the_start_it_answers():
    # Codex P2. The cross-mode guard cannot catch a SAME-mode ack meant for
    # another window, and that is the load-bearing case: the window that claims
    # the microphone mid-start becomes the lease holder, so the in-flight start's
    # ack is fanned out to it. Without this guard it clears its own timeout,
    # resolves, reads the blocked route that ack carries and aborts its
    # microphone flow -- and its real ack, carrying the re-decided route, lands
    # on a flow that already gave up.
    websocket_source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")

    started_handler = websocket_source.split(
        "S.isTextSessionActive = response.input_mode === 'text';", 1
    )[0].rsplit("} else if (response.type === 'session_started') {", 1)[1]
    guard = started_handler.split("var _ackAnswersThisWindow =", 1)[1].split(";", 1)[0]
    assert "response.request_id === S._pendingSessionStartRequestId" in guard
    # Anchored on the resolver first (Codex P2): a dozen places clear the
    # resolver, and expecting every one of them to also clear the id is exactly
    # the checklist that goes stale. A window with no start pending must treat
    # any ack as its own, or a leaked id silently disables the latch forever.
    assert "!S.sessionStartedResolver" in guard
    # An ack with no id counts as ours: the internal starts (proactive,
    # greeting, disconnect recovery) carry no request, and the cross-mode guard
    # already covers them.
    assert "!response.request_id" in guard
    assert "!S._pendingSessionStartRequestId" in guard

    # Settling is what the guard gates -- the timeout clear and the deferred
    # resolve. The UI sync below it is deliberately NOT gated: the backend did
    # start a session, so composer visibility and the microphone teardown still
    # apply to this window.
    tail = websocket_source.split("var _ackAnswersThisWindow =", 1)[1]
    timeout_clear = tail.split("clearTimeout(window.sessionTimeoutId);", 1)[0]
    assert "_ackAnswersThisWindow && S.sessionStartedResolver" in timeout_clear
    capture = next(l for l in tail.splitlines() if "var _ackedResolver =" in l)
    assert "_ackAnswersThisWindow ?" in capture
    # voiceStartPending is a start-lifecycle flag, not a session fact:
    # app-auto-goodbye.js reads it as "a voice start is in flight", so clearing
    # it on somebody else's ack lets goodbye/idle run through a legitimate mic
    # start that is still waiting for its own ack (Codex P2).
    pending_clear = next(
        l for l in tail.splitlines() if "S.voiceStartPending = false;" in l
    )
    assert "_ackAnswersThisWindow" in pending_clear
    # Session facts stay ungated -- the backend really did start a session.
    session_facts = next(
        l for l in tail.splitlines() if "S.voiceChatActive = response.input_mode" in l
    )
    assert "_ackAnswersThisWindow" not in session_facts

def test_cross_window_adopted_values_roll_the_dirty_baseline():
    # Without rolling the baseline, a value this window merely RECEIVED looks
    # like a local user edit on the next unrelated save: the key gets marked
    # dirty, that grants S.independentAsrAuthoritative, it rides out in
    # changedKeys as an explicit toggle other windows trust, and the pending
    # settings GET skips it as user-owned. That launders an adopted value into
    # user intent with no clock race at all.
    settings_source = APP_SETTINGS_PATH.read_text(encoding="utf-8")

    listener_block = settings_source.split(
        "window.addEventListener('storage', function (event) {", 1
    )[1].split("});", 1)[0]

    apply_index = listener_block.index("const changed = applySharedRuntimeSettings(incoming);")
    roll_index = listener_block.index("_settingsBaseline[key] = S[key];")
    assert apply_index < roll_index, "the baseline roll must observe the applied values"
    # Keys this window really did touch keep their authority.
    assert "if (_dirtySettingsKeys.has(key)) continue;" in listener_block

def test_cross_mode_session_started_still_stops_the_microphone():
    # The cross-mode ack guard returns early when this window has its own start
    # in flight. In the multi-window sequence "user clicks the mic in A while B
    # sends text", A receives the text session_started with an audio start
    # pending and would return before the teardown -- leaving the hardware mic
    # open and uploading into a route the text session pinned to blocked.
    websocket_source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")

    guard = websocket_source.split(
        "console.log('[App] ignore cross-mode session_started', response.input_mode,", 1
    )[1].split("return;", 1)[0]

    assert "response.input_mode === 'text'" in guard
    assert "S.isRecording === true" in guard
    # Same notifyServer:false reasoning as the main branch: pause_session would
    # end the text session that this very ack just announced.
    assert "window.stopRecording({ notifyServer: false });" in guard

def test_startup_failure_runs_the_same_teardown_as_a_runtime_failure():
    # Codex P2. A startup failure (provider connect, credentials, config)
    # leaves the route blocked but can NEVER emit a BLOCKED lifecycle event --
    # IndependentAsrRuntime.start cannot reach _handle_independent_asr_error,
    # the only emitter. So the terminal ASR_INDEPENDENT_* codes used to show a
    # toast and nothing else, while the browser kept the hardware microphone
    # open for the rest of the session.
    source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")

    status_block = source.split(
        "if (statusCode && statusCode.indexOf('ASR_INDEPENDENT_') === 0)", 1
    )[1].split("if (statusCode === 'TTS_CONNECTION_FAILED')", 1)[0]
    terminal = status_block.split(
        "if (statusCode === 'ASR_INDEPENDENT_INJECTION_FAILED')", 1
    )[1]

    # Both failure kinds go through one teardown, so they cannot drift.
    assert "tearDownBlockedVoiceRoute();" in terminal
    lifecycle_block = source.split("if (statusCode === 'ASR_LIFECYCLE_STATE')", 1)[
        1
    ].split("if (statusCode === 'VOICE_INPUT_LEASE_RESYNC_REQUIRED')", 1)[0]
    assert "tearDownBlockedVoiceRoute();" in lifecycle_block
    assert source.count("function tearDownBlockedVoiceRoute()") == 1

def test_audio_preprocessing_failure_tears_down_the_voice_route():
    # Codex P2. ASR_AUDIO_PREPROCESSING_FAILED rides neither the BLOCKED
    # lifecycle channel nor the ASR_INDEPENDENT_ prefix, so it was the one
    # status that announced a dead route while the microphone kept running.
    websocket_source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")

    branch = _block_after(
        websocket_source, "if (statusCode === 'ASR_AUDIO_PREPROCESSING_FAILED') {"
    )
    assert "tearDownBlockedVoiceRoute();" in branch
    assert "microphone.audioPreprocessingFailed" in branch
    # It must be reached before the ASR_INDEPENDENT_ prefix test, which would
    # not match this code anyway but makes the ordering explicit.
    assert websocket_source.index(
        "if (statusCode === 'ASR_AUDIO_PREPROCESSING_FAILED') {"
    ) < websocket_source.index("if (statusCode && statusCode.indexOf('ASR_INDEPENDENT_') === 0) {")

def test_auto_restart_unwinds_a_cancelled_microphone_start():
    # startMicCapture returns false for an ownership cancellation. The restart's
    # backend session has already been accepted by then, so it must enter the
    # common teardown without showing a generic failure toast or continuing to
    # the floating-control/restartComplete success path.
    websocket_source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")
    restart = websocket_source.split("await sessionStartPromise;", 1)[1].split(
        "} catch (error) {", 1
    )[0]
    restart_code = _code_only(restart)
    await_marker = "microphoneStarted = await window.startMicCapture();"
    cancellation_marker = "if (microphoneStarted !== true) {"
    success_marker = "window.syncFloatingMicButtonState(true)"
    assert await_marker in restart_code
    assert cancellation_marker in restart_code
    assert restart_code.index(await_marker) < restart_code.index(cancellation_marker)
    assert restart_code.index(cancellation_marker) < restart_code.index(success_marker)
    assert "microphoneStartCancelled.microphoneStartCancelled = true;" in restart_code
    assert "throw microphoneStartCancelled;" in restart_code

    catch = websocket_source.split("} catch (error) {", 1)[1].split(
        "}, 7500);", 1
    )[0]
    catch_code = _code_only(catch)
    assert "error && error.microphoneStartCancelled" in catch_code
    assert "if (!isMicrophoneStartCancelled" in catch_code
    assert "S.socket.send(JSON.stringify({ action: 'end_session' }));" in catch_code
    assert "window.syncFloatingMicButtonState(false)" in catch_code

def test_text_takeover_cancels_a_pending_microphone_start():
    # Both text-session branches stop an ALREADY-recording mic; neither could
    # reach a start still inside its await window.
    websocket_source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")

    # Count the GUARDED form, not the bare call: `if (false) window.invalid...`
    # keeps the bare substring and would satisfy a looser count.
    guarded = (
        "if (response.input_mode === 'text' "
        "&& typeof window.invalidatePendingMicStart === 'function') "
        "window.invalidatePendingMicStart();"
    )
    assert websocket_source.count(guarded) == 2
    # Each sits with, and before, its paired stopRecording teardown.
    for branch_opener in (
        "console.log('[App] text session installed; stopping the microphone (cross-mode)');",
        "console.log('[App] text session installed; stopping the microphone');",
    ):
        before = websocket_source.split(branch_opener, 1)[0]
        assert "window.invalidatePendingMicStart();" in before

    capture_source = APP_AUDIO_CAPTURE_PATH.read_text(encoding="utf-8")
    assert "window.invalidatePendingMicStart = invalidatePendingMicStart;" in capture_source

def test_reconnect_reconciliation_tombstones_the_route_the_server_finalized():
    """The reconnect snapshot is the compensation path for a missed ``closed``.

    Precisely because the websocket event was lost, no tombstone exists for the
    route this branch clears, so a late ``GAME_VOICE_STT_GATE_ACTIVE`` for it
    would re-activate a dead route on the page -- which suppresses proactive
    chat and auto-goodbye until a full open/close cycle or a reload.

    The identity recorded must come from the server's own snapshot. This read
    can disagree with the socket (character resolution drift), and tombstoning
    the identity the page currently holds would permanently reject the live
    route's real gate -- in browser_fallback mode that means the game never
    receives a transcript.
    """
    source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")
    reconnect_block = _block_after(
        source,
        "function syncGameWindowStateOnWsConnect() {",
    )
    closed_branch = reconnect_block[reconnect_block.index("var reconciledWasActive"):]
    assert "advanceGameRouteStateRevision();" in closed_branch, (
        "the reconnect snapshot cleared route state without advancing the "
        "revision, so a snapshot already in flight cannot be recognised as stale"
    )
    assert "rememberEndedGameRouteIdentity(" in closed_branch, (
        "the reconnect snapshot cleared a route without tombstoning it, so a "
        "late STT gate can re-activate it"
    )
    tombstone_call = closed_branch[
        closed_branch.index("rememberEndedGameRouteIdentity("):
    ]
    tombstone_call = tombstone_call[: tombstone_call.index(");")]
    for page_identity in (
        "S.gameRouteGameType",
        "S.gameRouteSessionId",
        "S.gameRouteInstanceId",
    ):
        assert page_identity not in tombstone_call, (
            f"the reconnect tombstone fell back to {page_identity}; only the "
            "server's own finalized identity is safe to record here"
        )
    assert "ended_route" in closed_branch, (
        "the reconnect tombstone must read the identity the server finalized"
    )
