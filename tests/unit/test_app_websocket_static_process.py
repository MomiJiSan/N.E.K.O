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

def test_reconnect_route_snapshot_cannot_overwrite_a_newer_websocket_route_event():
    source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")
    state_source = APP_STATE_PATH.read_text(encoding="utf-8")
    reconnect_block = _block_after(
        source,
        "function syncGameWindowStateOnWsConnect() {",
    )

    assert "gameRouteStateRevision: 0" in state_source
    assert "var reconciliationGeneration = _gameRouteReconciliationGeneration;" in reconnect_block
    assert "var routeRevisionAtRequest = gameRouteStateRevision();" in reconnect_block
    assert (
        "reconciliationGeneration !== _gameRouteReconciliationGeneration"
        in reconnect_block
    )
    assert (
        "gameRouteStateRevision() !== routeRevisionAtRequest"
        in reconnect_block
    )
    assert reconnect_block.index(
        "gameRouteStateRevision() !== routeRevisionAtRequest"
    ) < reconnect_block.index(
        "window.dispatchEvent(new CustomEvent('neko-game-window-state-change'"
    )
    stt_gate_block = _block_after(
        source,
        "if (statusCode === 'GAME_VOICE_STT_GATE_ACTIVE') {",
    )
    assert "incomingSttSessionId !== currentSttSessionId" in stt_gate_block
    assert re.search(
        r"\(incomingSttRouteInstanceId \|\| currentSttRouteInstanceId\)\s*"
        r"&&\s*"
        r"incomingSttRouteInstanceId !== currentSttRouteInstanceId",
        stt_gate_block,
    )
    assert stt_gate_block.index("if (staleSttGate) {") < stt_gate_block.index(
        "advanceGameRouteStateRevision();"
    )
    assert stt_gate_block.index(
        "advanceGameRouteStateRevision();"
    ) < stt_gate_block.index(
        "S.gameRouteActive = true;"
    )
    assert source.count("advanceGameRouteStateRevision();") >= 4

def test_independent_asr_injection_failure_does_not_show_fallback_toast():
    source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")

    status_block = source.split(
        "if (statusCode && statusCode.indexOf('ASR_INDEPENDENT_') === 0)",
        1,
    )[1].split("if (statusCode === 'TTS_CONNECTION_FAILED')", 1)[0]
    injection_branch = status_block.split(
        "if (statusCode === 'ASR_INDEPENDENT_INJECTION_FAILED')",
        1,
    )[1].split("tearDownBlockedVoiceRoute();", 1)[0]

    assert "return;" in injection_branch
    assert "independentAsrFallback" not in injection_branch

def test_independent_asr_terminal_status_reports_stopped_voice_input():
    source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")

    assert "Independent ASR unavailable; using Omni native recognition" not in source
    assert "Voice input has stopped for this session" in source

def test_voice_session_activation_status_is_validated_and_exposed_to_ui():
    source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")

    assert "statusCode === 'VOICE_SESSION_ACTIVATION_STATE'" in source
    assert "data-voice-session-activation-state" in source
    assert "voice-session-activation-changed" in source
    assert "voiceSessionActivationRevision" in source
    assert "activationRevision <=" in source
    assert "voiceIdentity.sessionWaiting" in source
    assert "voiceIdentity.sessionActive" in source
    assert "voiceIdentity.sessionUnavailable" in source

def test_independent_asr_provider_copy_resolves_via_provider_names():
    source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")
    capture_source = APP_AUDIO_CAPTURE_PATH.read_text(encoding="utf-8")

    assert "{ provider: asrProvider }" not in source
    ready_branch = source.split("if (statusCode === 'ASR_INDEPENDENT_READY')", 1)[1].split(
        "if (statusCode === 'ASR_INDEPENDENT_DISABLED')",
        1,
    )[0]
    assert "window.t('microphone.independentAsrActive', { providerKey: asrProvider || 'unknown' })" in ready_branch
    assert "window.t('microphone.independentAsrProviderUnavailable', { providerKey: asrProvider || 'unknown' })" in source

    # The shared popover now owns the summary. It resolves the registry key to a
    # display name, renders that value through the locale template, and refreshes
    # from the toggle handler so the visible route never lags the user's choice.
    summary_block = capture_source.split(
        "function updateVoiceRecognitionUi() {", 1
    )[1].split("function onVoiceLifecycleChanged()", 1)[0]
    assert "'microphone.independentAsrSummary'" in summary_block
    assert "{ provider: provider }" in summary_block
    assert "'microphone.voiceRecognitionDisabled'" in summary_block
    assert "provider: S.independentAsrProvider" not in summary_block

    change_handler = capture_source.split(
        "var asrToggle = createVoiceSettingToggle(", 1
    )[1].split("var voicePanelId", 1)[0]
    assert "updateVoiceRecognitionUi();" in change_handler

def test_response_discarded_visible_in_react_chat():
    source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")

    assert "function appendAssistantStatusMessage(text)" in source
    assert "window.reactChatWindowHost.appendMessage({" in source
    assert "appendAssistantStatusMessage(translatedDiscardMsg);" in source

    helper_block = source.split("function appendAssistantStatusMessage(text)", 1)[1].split(
        "function websocketTraceEnabled()",
        1,
    )[0]
    assert helper_block.index("window.reactChatWindowHost.appendMessage({") < helper_block.index(
        "document.createElement('div')"
    )
    assert "status: 'failed'" in helper_block
    assert "window.currentGeminiMessage" not in helper_block

    response_discarded_block = source.split("// -------- response_discarded --------", 1)[1].split(
        "// -------- user_transcript --------",
        1,
    )[0]
    assert "document.createElement('div')" not in response_discarded_block
    assert "appendChild(messageDiv)" not in response_discarded_block

def test_external_asr_preview_uses_owned_react_message_id():
    source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")

    preview_helper = source.split("function upsertExternalAsrPreview(text)", 1)[1].split(
        "function removeExternalAsrPreview()", 1
    )[0]
    remove_helper = source.split("function removeExternalAsrPreview()", 1)[1].split(
        "function websocketTraceEnabled()", 1
    )[0]
    event_block = source.split("// -------- user_transcript_preview", 1)[1].split(
        "// -------- user_transcript --------", 1
    )[0]
    final_block = source.split("// -------- user_transcript --------", 1)[1].split(
        "// --------", 1
    )[0]

    assert "reactChatWindowHost" in preview_helper
    assert "host.appendMessage({" in preview_helper
    assert "host.updateMessage(existingId" in preview_helper
    assert "querySelectorAll" not in event_block
    assert "window.appendMessage" not in event_block
    assert "host.removeMessage(messageId)" in remove_helper
    assert "removeExternalAsrPreview();" in final_block

def test_independent_asr_toggle_awaits_server_sync_before_next_session():
    # Session start reads the SERVER-persisted independentAsrEnabled value
    # (asr_runtime.py _start_independent_asr_if_enabled), so the toggle must
    # not rely on the fire-and-forget POST inside saveSettings(): it persists
    # locally, runs the POST itself, and publishes the in-flight promise for
    # the session-start path to await.
    capture_source = APP_AUDIO_CAPTURE_PATH.read_text(encoding="utf-8")
    websocket_source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")

    persist_block = capture_source.split(
        "function persistVoiceSettingChange() {", 1
    )[1].split("function markVoiceSettingsPending", 1)[0]
    toggle_block = capture_source.split(
        "var asrToggle = createVoiceSettingToggle(", 1
    )[1].split("asrRow.appendChild(asrCopy);", 1)[0]
    assert "persistVoiceSettingChange();" in toggle_block
    assert "window.appSettings.saveSettings({ skipServerSync: true });" in persist_block
    assert "window.appSettings.syncSettingsToServer({ userInitiated: true })" in persist_block
    assert "S.pendingSettingsSyncPromise = syncPromise;" in persist_block
    # Completion clears the gate only when it still owns it (a newer toggle
    # may have replaced the pending promise meanwhile).
    assert "if (S.pendingSettingsSyncPromise === syncPromise)" in persist_block
    assert "S.pendingSettingsSyncPromise = null;" in persist_block
    # Fallback when the settings module does not expose syncSettingsToServer.
    assert "window.appSettings.saveSettings();" in persist_block

    gate_block = websocket_source.split(
        "function ensureWebSocketOpen(timeoutMs = 5000)",
        1,
    )[1].split("function ensureWebSocketOpenNow(timeoutMs)", 1)[0]
    assert "S.pendingSettingsSyncPromise" in gate_block
    # Negative: only thenables gate; anything else falls through immediately.
    assert "typeof pendingSync.then === 'function'" in gate_block
    # The wait is bounded and never rejects, so a hung or failed POST cannot
    # block session starts or socket-dependent flows.
    assert "Promise.race([" in gate_block
    assert "SETTINGS_SYNC_GATE_TIMEOUT_MS" in gate_block
    assert "pendingSync.catch(" in gate_block
    assert "return ensureWebSocketOpenNow(timeoutMs);" in gate_block
    assert "var SETTINGS_SYNC_GATE_TIMEOUT_MS = 3000;" in websocket_source

def test_start_session_payload_carries_independent_asr_handshake():
    # The bounded settings-sync gate is best-effort: when the settings POST
    # fails or outlives the bound, the backend would read a stale persisted
    # independentAsrEnabled. The send() wrapper stamps the frontend's
    # authoritative toggle onto every start_session payload so the backend can
    # override that read (websocket_router -> set_independent_asr_handshake).
    websocket_source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")

    wrapper = websocket_source.split(
        "function attachStartSessionHandshake(ws)",
        1,
    )[1].split("function connectWebSocket()", 1)[0]
    # Strict-bool stamp taken from live S state at send time — but only once
    # settings are hydrated (see
    # test_start_session_handshake_omitted_until_settings_hydrated).
    assert "msg.independent_asr_enabled = S.independentAsrEnabled === true;" in wrapper
    # Only start_session text frames are rewritten; binary audio frames and
    # other messages pass through untouched.
    assert "typeof data === 'string'" in wrapper
    assert "msg.action === 'start_session'" in wrapper
    assert "coreApiSupportsIndependentAsr" not in wrapper

    # The wrapper is attached at the single socket-creation seam, so every
    # start_session send site (including the ones in app-buttons.js) carries
    # the field.
    creation_index = websocket_source.index("S.socket = new WebSocket(wsUrl);")
    attach_index = websocket_source.index("attachStartSessionHandshake(S.socket);")
    assert 0 < attach_index - creation_index < 200

def test_start_session_payload_carries_resource_optimization_handshake():
    websocket_source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")
    state_source = APP_STATE_PATH.read_text(encoding="utf-8")
    settings_source = APP_SETTINGS_PATH.read_text(encoding="utf-8")
    wrapper = websocket_source.split(
        "function attachStartSessionHandshake(ws)",
        1,
    )[1].split("function connectWebSocket()", 1)[0]

    assert "voiceInputResourceOptimizationAuthoritative: false," in state_source
    assert "S.voiceInputResourceOptimizationAuthoritative === true" in wrapper
    assert (
        "msg.voice_input_resource_optimization_enabled = "
        "S.voiceInputResourceOptimizationEnabled !== false;"
    ) in wrapper
    assert (
        "_dirtySettingsKeys.has('voiceInputResourceOptimizationEnabled')"
        in settings_source
    )
    assert "S.voiceInputResourceOptimizationAuthoritative = true;" in settings_source

def test_periodic_sync_skips_post_and_never_marks_hydration_while_unhydrated():
    # Persistent GET failure: loadSettingsFromServer resolves null (or the
    # whole chain throws), yet BOTH failure paths still start the periodic
    # task (the .finally() after the merge callback, and the outer catch).
    # Before the userInitiated split, syncSettingsToServer's entry marked
    # S.settingsHydrated unconditionally, so the 60s tick (a) uploaded the
    # boot default independentAsrEnabled=false over the server-persisted
    # true and (b) falsely armed the start_session handshake with that
    # default. Pin the two-part fix.
    settings_source = APP_SETTINGS_PATH.read_text(encoding="utf-8")

    # Both GET-failure paths do start the periodic task — that is exactly why
    # the tick itself must carry the guard.
    load_fn = settings_source.split("function loadSettings()", 1)[1]
    finally_block = load_fn.split("}).finally(() => {", 1)[1].split(
        "});", 1
    )[0]
    assert "startPeriodicSync();" in finally_block
    load_catch_block = _block_after(
        settings_source, "console.error('服务器设置同步启动失败:', error);"
    )
    assert "startPeriodicSync();" in load_catch_block

    # (1) The tick refuses to POST while settings were never hydrated (no
    # successful GET and no user change), logging the skip only once.
    tick_body = settings_source.split("_syncTimerId = setInterval(() => {", 1)[1].split(
        "}, SYNC_INTERVAL_MS);",
        1,
    )[0]
    hydration_guard_index = tick_body.index("if (S.settingsHydrated !== true) {")
    sync_call_index = tick_body.index("syncSettingsToServer();")
    assert hydration_guard_index < sync_call_index, (
        "the unhydrated guard must run before the periodic POST"
    )
    guard_block = tick_body[hydration_guard_index:sync_call_index]
    assert "return;" in guard_block
    assert "_periodicSyncSkippedUnhydratedLogged" in guard_block

    # (2) The periodic caller passes no options, so even a tick that does run
    # (post-hydration, or if the guard ever regressed) can never be the event
    # that marks hydration — only userInitiated callers mark (pinned in
    # test_settings_hydration_marked_on_server_merge_and_user_change).
    assert "userInitiated" not in tick_body

def test_settings_post_snapshot_waits_bounded_for_boot_get_merge():
    # Codex P2 (merge-before-post): a POST issued while the boot GET is still
    # pending used to snapshot pure local state, so unchanged fields carried
    # boot defaults. The queued runSync now awaits a bounded, never-rejecting
    # gate that settles when the GET's merge settled — the send-time snapshot
    # is therefore assembled AFTER the merge whenever the GET has resolved,
    # and unchanged fields carry server truth. If the GET outlives the bound
    # the POST proceeds with local state and the merge's writeback
    # saveSettings() converges the server afterwards.
    settings_source = APP_SETTINGS_PATH.read_text(encoding="utf-8")

    assert "let _settingsGetGate = Promise.resolve();" in settings_source
    assert "const SETTINGS_GET_GATE_TIMEOUT_MS = 3000;" in settings_source

    # The gate await sits inside the queued runSync, before the send-time
    # snapshot and the fetch.
    sync_fn = settings_source.split(
        "async function syncSettingsToServer(options)", 1
    )[1].split("function startPeriodicSync()", 1)[0]
    run_sync_body = sync_fn.split("const runSync = async () =>", 1)[1]
    gate_index = run_sync_body.index("await _settingsGetGate;")
    snapshot_index = run_sync_body.index("const settings = getConversationSettings();")
    fetch_index = run_sync_body.index(
        "await _fetchConversationSettingsJsonWithTimeout("
    )
    assert gate_index < snapshot_index < fetch_index

    # The gate is armed at GET issue time as a race between the settled merge
    # chain (with a catch so it can never reject) and the bounded timeout.
    load_fn = settings_source.split("function loadSettings()", 1)[1]
    assert "const mergeSettled = loadSettingsFromServer().then(serverResult => {" in load_fn
    gate_assign_index = load_fn.index("_settingsGetGate = Promise.race([")
    assert load_fn.index("startPeriodicSync();") < gate_assign_index
    gate_block = load_fn[gate_assign_index:].split("]);", 1)[0]
    assert "mergeSettled.catch(() => { })" in gate_block
    assert "setTimeout(resolve, SETTINGS_GET_GATE_TIMEOUT_MS)" in gate_block

    # Negative: the synchronous hydration/dirty marks stay at call time —
    # only the POST body waits for the merge, never the authority marks.
    assert sync_fn.index("S.settingsHydrated = true;") < sync_fn.index(
        "const runSync = async () =>"
    )
    assert sync_fn.index("_markUserDirtySettings();") < sync_fn.index(
        "const runSync = async () =>"
    )

def test_cross_window_asr_flip_marks_hydration_and_asr_dirty():
    # Codex P2: a cross-window independent-ASR toggle arrives via the
    # 'storage' listener, which used to copy the value into S without marking
    # S.settingsHydrated or the key's authority. In the receiving
    # window that meant (a) the next start_session omitted the handshake field
    # (the stamp is gated on S.settingsHydrated, pinned by
    # test_start_session_handshake_omitted_until_settings_hydrated), so the
    # backend read the OLD persisted value while the originating window's POST
    # was still in flight, and (b) a still-pending settings GET later merged
    # the stale server snapshot over the flip and POSTed it back via
    # saveSettings(). Pin the fix: the flip is detected before the apply and
    # treated as an authoritative hydration event that marks the ASR key
    # dirty (so the field-level merge preserves it), with no POST from the
    # receiving window.
    settings_source = APP_SETTINGS_PATH.read_text(encoding="utf-8")

    listener_block = settings_source.split(
        "window.addEventListener('storage', function (event) {", 1
    )[1].split("});", 1)[0]

    # Flip detection: own-property guard plus strict inequality against S,
    # computed BEFORE applySharedRuntimeSettings mutates S.
    assert (
        "Object.prototype.hasOwnProperty.call(settings, 'independentAsrEnabled')"
        in listener_block
    )
    assert "S.independentAsrEnabled !== settings.independentAsrEnabled" in listener_block
    assert listener_block.index("const asrChangedByOtherWindow") < listener_block.index(
        "applySharedRuntimeSettings(incoming)"
    )

    # Hydration mark + ASR dirty mark sit inside the ASR-flip gate only.
    flip_gate = listener_block.split("if (asrChangedByOtherWindow) {", 1)[1].split(
        "}", 1
    )[0]
    assert "S.settingsHydrated = true;" in flip_gate
    assert "_dirtySettingsKeys.add('independentAsrEnabled');" in flip_gate
    optimization_gate = listener_block.split(
        "if (optimizationChangedByOtherWindow) {",
        1,
    )[1].split("}", 1)[0]
    assert "S.settingsHydrated = true;" in optimization_gate
    assert listener_block.count("S.settingsHydrated = true;") == 2
    assert listener_block.count("_dirtySettingsKeys.add('independentAsrEnabled');") == 1

    # No POST from the receiving window: the originating window owns
    # persistence, and a receiving-window POST would duplicate writes and
    # loop storage events between windows. (Assert on code lines only — the
    # in-source comment legitimately names saveSettings.)
    listener_code = "\n".join(
        line
        for line in listener_block.splitlines()
        if not line.strip().startswith("//")
    )
    assert "syncSettingsToServer" not in listener_code
    assert "saveSettings();" not in listener_code
    if "saveSettings({" in listener_code:
        assert "skipServerSync: true" in listener_code
    assert "syncSettingsToServer" not in listener_code
    assert "fetch(" not in listener_code

    # Negative: applySharedRuntimeSettings itself must stay authority-neutral —
    # other shared keys (and non-flip events) keep syncing values across
    # windows without marking hydration or dirtying keys, so a
    # first-launch boot-defaults write in another window can never arm this
    # window's periodic sync or handshake.
    apply_fn = settings_source.split(
        "function applySharedRuntimeSettings(settings) {", 1
    )[1].split("function isManualScreenShareActive()", 1)[0]
    assert "settingsHydrated" not in apply_fn
    assert "_dirtySettingsKeys" not in apply_fn
    assert "_markUserDirtySettings" not in apply_fn

def test_cross_window_asr_flip_authoritative_over_pending_get_harness():
    # Behavioral pin for the cross-window Codex P2 fix: drive the real module
    # in a vm sandbox, deliver a 'storage' event carrying another window's
    # independent-ASR flip while this window's boot GET is still pending, then
    # resolve that GET with the stale pre-flip server value. The flip must mark
    # hydration (arming the start_session handshake stamp), the field-level
    # merge must preserve the flipped key (marked dirty by the flip gate), and
    # the receiving window must never POST. Negative: a storage event that
    # does NOT flip the toggle stays non-authoritative and the pending GET
    # still merges normally.
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
          const listeners = [];
          const dispatchedEvents = [];
          const sandbox = {
            console: { log() {}, warn() {}, error() {} },
            CustomEvent: class {
              constructor(type) { this.type = type; }
            },
            setInterval() { return 0; },
            clearInterval() {},
            setTimeout(fn, ms) {
              // Unref'd so a pending gate timer cannot hold the process open;
              // the harness then exits naturally and stdout always flushes.
              const t = setTimeout(fn, ms);
              if (t && typeof t.unref === 'function') t.unref();
              return t;
            },
            clearTimeout,
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
            dispatchEvent(event) { dispatchedEvents.push(event.type); },
          };
          vm.createContext(sandbox);
          vm.runInContext(source, sandbox);
          const storage = listeners.filter((entry) => entry.type === 'storage');
          assert(storage.length === 1, 'module must register exactly one storage listener');
          return {
            postCalls,
            getCalls,
            dispatchedEvents,
            S: sandbox.window.appState,
            fireStorage(newValue) {
              storage[0].fn({ key: 'project_neko_settings', newValue });
            },
          };
        }

        const okPost = { ok: true, json: async () => ({ success: true }) };
        const tick = () => new Promise((resolve) => setImmediate(resolve));

        async function main() {
          // Scenario 1: cross-window ASR flip while the boot GET is pending.
          const ctx = makeContext();
          assert(ctx.getCalls.length === 1, 'boot must issue the settings GET');
          assert(ctx.S.settingsHydrated === false, 'boot alone must not mark hydration');

          ctx.fireStorage(JSON.stringify({ independentAsrEnabled: true }));
          assert(ctx.S.independentAsrEnabled === true, 'the flip must be applied to S');
          assert(ctx.S.settingsHydrated === true, 'the flip must arm the start_session handshake stamp');
          assert(ctx.S.voiceSettingsPendingUntilEpoch === 11, 'the flip must target the next voice-session epoch');
          assert(ctx.S.pendingVoiceRouteIndependentAsr === true, 'the pending summary must preserve the active route');
          assert(
            ctx.dispatchedEvents.includes('neko:voice-settings-pending-changed'),
            'the flip must notify an already-open microphone popover'
          );
          assert(ctx.postCalls.length === 0, 'the receiving window must not POST (originating window owns persistence)');

          // The GET now resolves with the server value read BEFORE the other
          // window's POST landed: the flipped key is dirty, so the field-level
          // merge must preserve it.
          ctx.getCalls[0].resolve({
            ok: true,
            json: async () => ({ success: true, settings: { independentAsrEnabled: false }, telemetryBranch: null }),
          });
          await tick();
          await tick();
          assert(ctx.S.independentAsrEnabled === true, 'the stale GET merge must not overwrite the cross-window flip');
          assert(ctx.postCalls.length === 0, 'the dropped merge must not POST the stale value back');

          // Scenario 2 (negative): a storage event without an ASR flip is
          // non-authoritative — other shared keys still sync, hydration stays
          // unmarked, and the pending GET then merges exactly as before.
          const ctx2 = makeContext();
          ctx2.fireStorage(JSON.stringify({ independentAsrEnabled: false, mergeMessagesEnabled: true }));
          assert(ctx2.S.mergeMessagesEnabled === true, 'other shared keys must still sync across windows');
          assert(ctx2.S.settingsHydrated === false, 'no ASR flip means no hydration mark');
          assert(ctx2.S.voiceSettingsPendingUntilEpoch === null, 'no flip means no pending voice-session marker');
          assert(ctx2.dispatchedEvents.length === 0, 'no flip means no popover notification');
          assert(ctx2.postCalls.length === 0, 'a non-flip storage event must not POST either');

          ctx2.getCalls[0].resolve({
            ok: true,
            json: async () => ({ success: true, settings: { independentAsrEnabled: true }, telemetryBranch: null }),
          });
          await tick();
          await tick();
          assert(ctx2.S.independentAsrEnabled === true, 'the normal server merge must still apply');
          assert(ctx2.S.settingsHydrated === true, 'the normal server merge must still mark hydration');
          assert(ctx2.postCalls.length === 1, 'the same-window merge write-back POST must be unchanged');
          ctx2.postCalls[0].resolve(okPost);

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
        "cross-window ASR flip harness failed\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "HARNESS_OK" in result.stdout

def test_shared_settings_writes_carry_explicit_change_metadata():
    # Codex P2 (follow-up): saveSettings() writes independentAsrEnabled into
    # EVERY localStorage snapshot, so the receiving window could not tell a real
    # cross-window toggle from the incidental copy an unrelated save carries.
    # Pin the metadata contract: every shared write goes through
    # _writeSharedSettings, which stamps a monotonic write id, the keys the user
    # explicitly changed, and whether the writer had hydrated.
    settings_source = APP_SETTINGS_PATH.read_text(encoding="utf-8")

    # No raw write of the shared key may bypass the metadata stamp.
    assert (
        "localStorage.setItem('project_neko_settings', JSON.stringify(settings))"
        not in settings_source
    )
    assert settings_source.count("_writeSharedSettings(") == 4  # 1 def + 3 writes
    save_fn = _block_after(settings_source, "function saveSettings(options) {")
    assert "serverMerged ? [] : _collectExplicitSharedKeys(settings)" in save_fn
    assert "const serverMerged = !!(options && options.serverMerged);" in save_fn
    assert "const pendingRecovery = !!(options && options.pendingRecovery);" in save_fn
    # The pre-hydration migration write is explicitly non-authoritative.
    assert "_writeSharedSettings(settings, []);" in settings_source

    write_fn = settings_source.split(
        "function _writeSharedSettings(", 1
    )[1].split("\n    }", 1)[0]
    assert "writeId: _nextSharedWriteId()," in write_fn
    assert "changedKeys: explicitKeys || []," in write_fn
    assert "hydrated: S.settingsHydrated === true" in write_fn
    assert "pendingRecovery: pendingRecovery === true" in write_fn
    assert "ownMeta.knownKeyWrites = _knownSharedKeyWritesSnapshot();" in write_fn
    assert "ownMeta.serverRevision = _conversationSettingsRevision;" in write_fn
    assert "serverAuthoritativeKeys.slice()" in write_fn
    assert (
        "_rememberSharedKeyWrites(serverAuthoritativeKeys || [], ownMeta)"
        not in write_fn
    )
    assert "localStorage.setItem('project_neko_settings', JSON.stringify(payload));" in write_fn

    # The write id must be strictly increasing within a window and comparable
    # across windows (one wall clock per browser profile).
    id_fn = settings_source.split("function _nextSharedWriteId() {", 1)[1].split("\n    }", 1)[0]
    assert "Date.now()" in id_fn
    # Floor the mint by the highest id ever APPLIED, not just the highest this
    # window minted: otherwise a window that already applied another window's
    # write can mint an id at or below it and have its own write read as
    # superseded, discarding a genuine cross-window toggle. This covers only
    # the already-OBSERVED case -- a genuinely concurrent same-millisecond tie
    # cannot be broken at mint time and is resolved by the listener's
    # explicit-intent rule instead (pinned below).
    assert "Math.max(_lastSharedWriteId, _lastAppliedSharedWriteId)" in id_fn
    assert "_lastSharedWriteId = now > idFloor ? now : idFloor + 1;" in id_fn

    # Explicit keys = still-pending writes PLUS divergence from the dirty-diff
    # baseline (the ASR toggle handler persists locally before its userInitiated
    # sync rolls that baseline). The monotone boot-merge dirty set must not mint
    # a new per-key token for an already-acknowledged value.
    collect_fn = settings_source.split("function _collectExplicitSharedKeys(snapshot) {", 1)[
        1
    ].split("\n    }", 1)[0]
    assert "_pendingSettingsKeys.has(key)" in collect_fn
    assert "_dirtySettingsKeys.has(key)" not in collect_fn
    assert "_settingsBaseline[key] !== snapshot[key]" in collect_fn
    # Negative: only shared keys may be claimed, never the whole snapshot.
    assert "_SHARED_SETTINGS_KEYS.forEach" in collect_fn
    assert "Object.keys(snapshot)" not in collect_fn

    # Metadata-less payloads (a window still running the previous build) parse
    # to null, which routes the listener back to the legacy fallback.
    read_fn = settings_source.split("function _readSharedWriteMeta(settings) {", 1)[1].split(
        "\n    }", 1
    )[0]
    assert "if (!meta || typeof meta !== 'object') return null;" in read_fn
    assert "if (!_isValidAsrWriteId(meta.writeId)) return null;" in read_fn
    id_validator = _block_after(
        settings_source,
        "function _isValidAsrWriteId(value, serverAuthoritative) {",
    )
    assert "Number.isSafeInteger(value)" in id_validator
    assert "Date.now() + _ASR_WRITE_ID_MAX_FUTURE_SKEW_MS" in id_validator
    assert "Number.MAX_SAFE_INTEGER - 1" in id_validator
    assert "if (serverAuthoritative === true) return true;" in id_validator
    assert "Array.isArray(meta.changedKeys) ? meta.changedKeys : []" in read_fn
    assert "knownKeyWritesPresent" in read_fn
    optimization_reader = read_fn.split(
        "optimizationDecision: (meta.optimizationDecision", 1
    )[1].split("optimizationDecisionPendingSync:", 1)[0]
    assert "_isValidAsrWriteId(" in optimization_reader
    assert "meta.optimizationDecision.writeId," in optimization_reader
    assert "Number.isInteger(meta.serverRevision)" in optimization_reader
    assert "isFinite(meta.optimizationDecision.writeId)" not in optimization_reader

    listener_block = settings_source.split(
        "window.addEventListener('storage', function (event) {", 1
    )[1].split("});", 1)[0]
    # Authority requires explicit intent + freshness + outranking this window's
    # own explicit choice; absent metadata falls back to today's
    # value-difference behaviour. The fourth term is load-bearing:
    # _lastAppliedSharedWriteId only records writes RECEIVED here, so freshness
    # alone cannot order this window's own pending toggle against a concurrent
    # one from another window, and the two swap values permanently.
    assert (
        "const asrChangedByOtherWindow = meta\n"
        "                ? (asrValueDiffers && asrMarkedExplicit && asrWriteIsNewer\n"
        "                    && asrOutranksLocalChoice)\n"
        "                : asrValueDiffers;" in listener_block
    )
    assert "meta.changedKeys.indexOf('independentAsrEnabled') !== -1" in listener_block
    assert "meta.writeId > _lastAppliedSharedWriteId" in listener_block
    # Freshness bookkeeping happens AFTER the authority decision, never before.
    assert listener_block.index("const asrChangedByOtherWindow") < listener_block.index(
        "_lastAppliedSharedWriteId = meta.writeId;"
    )
    assert listener_block.index("const asrValueIsStale") < listener_block.index(
        "_lastAppliedSharedWriteId = meta.writeId;"
    )
    # A stale/superseded ASR value is dropped from the apply set rather than
    # applied — and only that key, so other shared keys keep syncing.
    assert "delete incoming.independentAsrEnabled;" in listener_block
    assert "applySharedRuntimeSettings(incoming)" in listener_block

def test_settings_get_gate_timeout_downgrades_post_to_dirty_keys_only():
    # Codex P2 (round 15): the bounded gate preserves liveness, but on timeout
    # it used to release a FULL boot snapshot — overwriting every preference
    # the user never touched. The backend resolves the telemetry branch BEFORE
    # reading the settings file (main_routers/config_router/preferences.py
    # get_conversation_settings), so a slow GET resumes by reading the file the
    # POST just overwrote and the field-level merge can no longer restore the
    # originals. Pin the fix: while the GET chain is unsettled the POST body is
    # restricted to the explicitly dirty keys, which is safe because the
    # backend MERGES partial payloads (utils/preferences.py
    # save_global_conversation_settings -> global_pref.update(filtered_settings)).
    settings_source = APP_SETTINGS_PATH.read_text(encoding="utf-8")

    # Round 16 (Codex P2 follow-up): the gate flag must track a SUCCESSFUL
    # merge, not merely "the GET attempt finished". It starts false (nothing
    # merged yet) and is never re-armed in loadSettings — a merge that already
    # happened stays valid.
    assert "let _settingsMergedFromServer = false;" in settings_source
    assert "_settingsGetSettled" not in settings_source
    load_fn = settings_source.split("function loadSettings()", 1)[1]
    assert "_settingsMergedFromServer = false;" not in load_fn
    # It flips to true ONLY inside the merge callback, past the
    # `if (!serverResult) return;` guard, i.e. only when server values were
    # really applied — and before the merge writeback so that POST is full.
    merge_cb = load_fn.split("loadSettingsFromServer().then(serverResult => {", 1)[1].split(
        "}).finally(() => {", 1
    )[0]
    assert "if (!serverResult) return;" in merge_cb
    assert merge_cb.index("if (!serverResult) return;") < merge_cb.index(
        "_settingsMergedFromServer = true;"
    )
    assert merge_cb.index("_settingsMergedFromServer = true;") < merge_cb.index(
        "saveSettings({"
    )
    assert "serverAuthoritativeKeys: Object.keys(" in merge_cb
    # Negative: the failure paths must NOT re-enable full snapshots. The
    # finally runs for merged AND failed GETs, so it may not touch the flag;
    # neither may the synchronous-throw catch, where nothing was ever read.
    finally_block = load_fn.split("}).finally(() => {", 1)[1].split("});", 1)[0]
    assert "_settingsMergedFromServer =" not in finally_block
    assert "startPeriodicSync();" in finally_block
    startup_catch = load_fn.split("console.error('服务器设置同步启动失败:', error);", 1)[1]
    assert "_settingsMergedFromServer = true;" not in startup_catch

    # The send-time body: full snapshot only when server values were merged,
    # dirty-keys-only otherwise, and the fetch must post THAT body.
    sync_fn = settings_source.split(
        "async function syncSettingsToServer(options)", 1
    )[1].split("function startPeriodicSync()", 1)[0]
    run_sync_body = sync_fn.split("const runSync = async () =>", 1)[1]
    assert (
        "const payload = _settingsMergedFromServer ? settings : _pickDirtySettings(settings);"
        in run_sync_body
    )
    assert "body: JSON.stringify(payload)" in run_sync_body
    assert "JSON.stringify(settings)" not in run_sync_body
    # Ordering: gate await -> snapshot -> payload choice -> fetch.
    assert (
        run_sync_body.index("await _settingsGetGate;")
        < run_sync_body.index("const settings = getConversationSettings();")
        < run_sync_body.index("const payload =")
        < run_sync_body.index("await _fetchConversationSettingsJsonWithTimeout(")
    )
    # An empty dirty set means nothing user-authoritative exists yet: skip the
    # POST entirely rather than write pre-merge values.
    assert "if (Object.keys(payload).length === 0) {" in run_sync_body
    assert run_sync_body.index("if (Object.keys(payload).length === 0) {") < run_sync_body.index(
        "await _fetchConversationSettingsJsonWithTimeout("
    )

    # The picker copies ONLY pending keys (negative: acknowledged or untouched
    # keys cannot be dragged back into the partial body).
    pick_fn = settings_source.split("function _pickDirtySettings(settings) {", 1)[1].split(
        "function applySharedRuntimeSettings", 1
    )[0]
    assert "_pendingSettingsKeys.forEach((key) => {" in pick_fn
    assert "Object.prototype.hasOwnProperty.call(settings, key)" in pick_fn
    assert "partial[key] = settings[key];" in pick_fn
    assert "Object.keys(settings)" not in pick_fn
    assert "Object.assign" not in pick_fn

def test_failed_boot_get_keeps_posts_dirty_only_harness():
    # Codex P2 (round 16): the round-15 flag was released in the merge chain's
    # `finally`, which also runs when the GET resolved to null (HTTP error,
    # network error, success:false, unparsable body). The settings view was
    # then marked "settled" without a single server value having been merged,
    # so the next user edit POSTed the FULL boot/localStorage snapshot and a
    # POST succeeding after a transient GET failure overwrote every untouched
    # persisted preference — independentAsrEnabled included.
    #
    # Pin the split: "the GET attempt finished" and "server values were merged"
    # are different facts, and only the latter licenses full snapshots.
    # Scenario 1: HTTP-failed GET + later unrelated user edits -> each new
    # pending edit POST carries only that key, while an acknowledged key is not
    # resent and an idle periodic pass sends nothing. Scenario 2: application-level failure
    # (success:false) + ASR toggle -> the toggle IS persisted. Scenario 3:
    # network-error GET -> still pending-only, and the periodic timer does not
    # re-fetch or resend an acknowledged key. The recovery model remains "stay
    # partial-write-only for this session" — safe because the backend merges partial
    # payloads per key. Scenario 4 (recovery): a GET that fails the bound but
    # eventually SUCCEEDS flips back to full snapshots on its merge.
    harness = textwrap.dedent(
        """
        const fs = require('node:fs');
        const vm = require('node:vm');

        const source = fs.readFileSync(__APP_SETTINGS_PATH__, 'utf8');

        function assert(cond, msg) {
          if (!cond) throw new Error('ASSERT: ' + msg);
        }

        function makeContext(initialSettings) {
          const postCalls = [];
          const getCalls = [];
          const timers = [];
          const intervals = [];
          const storage = {};
          if (initialSettings) {
            storage.project_neko_settings = JSON.stringify(initialSettings);
          }
          const sandbox = {
            console: { log() {}, warn() {}, error() {} },
            setInterval(fn, ms) { intervals.push({ fn, ms }); return 1; },
            clearInterval() {},
            setTimeout(fn, ms) { timers.push({ fn, ms }); return { unref() {} }; },
            clearTimeout() {},
            localStorage: {
              getItem(key) { return Object.prototype.hasOwnProperty.call(storage, key) ? storage[key] : null; },
              setItem() {},
              removeItem(key) { delete storage[key]; },
            },
            document: { getElementById() { return null; } },
            fetch(url, opts) {
              return new Promise((resolve, reject) => {
                if (opts && opts.method === 'POST') {
                  postCalls.push({ url, body: opts.body, headers: opts.headers, resolve, reject });
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
            firePeriodicTick() {
              assert(intervals.length === 1, 'exactly one periodic sync timer must be armed');
              intervals[0].fn();
            },
          };
        }

        const okPost = { ok: true, json: async () => ({ success: true }) };
        const tick = () => new Promise((resolve) => setImmediate(resolve));

        async function main() {
          // ---- Scenario 1: HTTP-failed boot GET, then unrelated user edits.
          const ctx = makeContext();
          assert(ctx.getCalls.length === 1, 'boot must issue the settings GET');
          ctx.getCalls[0].resolve({ ok: false, status: 500 });
          await tick();
          await tick();
          assert(
            ctx.S.settingsHydrated === false,
            'a failed GET must not hydrate the handshake view'
          );
          assert(ctx.postCalls.length === 0, 'a failed GET must not POST anything by itself');

          ctx.win.mergeMessagesEnabled = true;   // settings-popup mirror
          ctx.mod.saveSettings();                // user path -> userInitiated POST
          assert(ctx.S.settingsHydrated === true, 'the user change hydrates synchronously');
          await tick();
          await tick();
          assert(ctx.postCalls.length === 1, 'the user edit must still be persisted (liveness)');
          const body1 = JSON.parse(ctx.postCalls[0].body);
          assert(body1.mergeMessagesEnabled === true, 'the dirty key must be persisted');
          assert(
            !('independentAsrEnabled' in body1),
            'a failed GET must NOT license a full snapshot: the untouched ASR preference '
              + 'would clobber the persisted value, got: ' + JSON.stringify(body1)
          );
          assert(
            Object.keys(body1).length === 1,
            'only the dirty key may travel, got: ' + JSON.stringify(body1)
          );
          ctx.postCalls[0].resolve(okPost);
          await tick();

          // The restriction does not decay: a LATER, second edit is still
          // pending-only. The first key was acknowledged and must not be resent.
          ctx.win.focusModeEnabled = true;
          ctx.mod.saveSettings();
          await tick();
          await tick();
          assert(ctx.postCalls.length === 2, 'the second user edit is persisted too');
          const body2 = JSON.parse(ctx.postCalls[1].body);
          assert(body2.focusModeEnabled === true, 'the newly dirtied key is persisted');
          assert(!('mergeMessagesEnabled' in body2), 'the acknowledged key is no longer pending');
          assert(
            Object.keys(body2).length === 1,
            'only the new pending key travels, got: ' + JSON.stringify(body2)
          );
          ctx.postCalls[1].resolve(okPost);
          await tick();

          // ... and the periodic sync (no userInitiated) neither widens nor
          // resends an already acknowledged body, and never rejects.
          const pp = ctx.mod.syncSettingsToServer();
          await tick();
          await tick();
          assert(ctx.postCalls.length === 2, 'the periodic-style sync has no pending keys to write');
          await pp;

          // ---- Scenario 2: application-level failure + the ASR toggle.
          const ctx2 = makeContext();
          ctx2.getCalls[0].resolve({
            ok: true,
            json: async () => ({ success: false, error: 'boom' }),
          });
          await tick();
          await tick();
          ctx2.S.independentAsrEnabled = true;
          ctx2.mod.saveSettings({ skipServerSync: true });   // toggle handler persists locally
          const p2 = ctx2.mod.syncSettingsToServer({ userInitiated: true });
          assert(
            ctx2.S.settingsHydrated === true,
            'the user toggle is authoritative for the handshake even without a merge'
          );
          await tick();
          await tick();
          assert(ctx2.postCalls.length === 1, 'the ASR toggle must be persisted after a failed GET');
          const asrBody = JSON.parse(ctx2.postCalls[0].body);
          assert(asrBody.independentAsrEnabled === true, 'the toggle carries the user choice');
          assert(
            Object.keys(asrBody).length === 1,
            'the toggle POST carries nothing else, got: ' + JSON.stringify(asrBody)
          );
          ctx2.postCalls[0].resolve(okPost);
          await p2;

          // ---- Scenario 3: network-error GET; the periodic timer only POSTs.
          const ctx3 = makeContext();
          ctx3.getCalls[0].reject(new Error('offline'));
          await tick();
          await tick();
          ctx3.win.mergeMessagesEnabled = true;
          ctx3.mod.saveSettings();
          await tick();
          await tick();
          assert(ctx3.postCalls.length === 1, 'the edit is persisted after a network-error GET');
          assert(
            Object.keys(JSON.parse(ctx3.postCalls[0].body)).length === 1,
            'a rejected GET keeps POSTs dirty-only'
          );
          ctx3.postCalls[0].resolve(okPost);
          await tick();
          ctx3.firePeriodicTick();
          await tick();
          await tick();
          assert(
            ctx3.getCalls.length === 1,
            'no path re-fetches the settings GET, so dirty-only must be permanently safe '
              + 'rather than a temporary state (recovery is a fresh page load)'
          );
          assert(ctx3.postCalls.length === 1, 'the periodic tick does not resend an acknowledged key');

          // ---- Scenario 4 (recovery): the bound elapses, the POST goes out
          // dirty-only, and the GET LATER succeeds -> full snapshots resume.
          const ctx4 = makeContext();
          ctx4.win.mergeMessagesEnabled = true;
          ctx4.mod.saveSettings();
          await tick();
          ctx4.fireGateTimeout();
          await tick();
          await tick();
          assert(ctx4.postCalls.length === 1, 'the bound releases the POST');
          assert(
            Object.keys(JSON.parse(ctx4.postCalls[0].body)).length === 1,
            'an unmerged view posts dirty keys only'
          );
          ctx4.postCalls[0].resolve(okPost);
          ctx4.getCalls[0].resolve({
            ok: true,
            json: async () => ({
              success: true,
              settings: { independentAsrEnabled: true, mergeMessagesEnabled: false },
              telemetryBranch: null,
            }),
          });
          await tick();
          await tick();
          assert(ctx4.S.independentAsrEnabled === true, 'the untouched key hydrates from the server');
          assert(ctx4.S.mergeMessagesEnabled === true, 'the dirty key survives the late merge');
          assert(ctx4.postCalls.length === 2, 'the merge writeback POST follows');
          const recovered = JSON.parse(ctx4.postCalls[1].body);
          assert(
            Object.keys(recovered).length > 2,
            'a real merge restores full snapshots, got: ' + JSON.stringify(recovered)
          );
          assert(recovered.independentAsrEnabled === true, 'the writeback carries the server value');
          ctx4.postCalls[1].resolve(okPost);
          await tick();
          ctx4.win.focusModeEnabled = true;
          ctx4.mod.saveSettings();
          await tick();
          await tick();
          assert(ctx4.postCalls.length === 3, 'the post-recovery user edit POSTs');
          const afterRecovery = JSON.parse(ctx4.postCalls[2].body);
          assert(
            Object.keys(afterRecovery).length > 2,
            'post-recovery edits keep using full snapshots, got: ' + JSON.stringify(afterRecovery)
          );
          assert(
            afterRecovery.independentAsrEnabled === true,
            'the full snapshot carries the merged server value, not the boot default'
          );
          ctx4.postCalls[2].resolve(okPost);
          await tick();

          // ---- Scenario 5: a legacy/mangled delayed GET omits revision. Once
          // a POST established a comparable revision, its ETag must not be
          // downgraded by the unversioned response.
          const ctxMissingRevision = makeContext();
          ctxMissingRevision.win.mergeMessagesEnabled = true;
          ctxMissingRevision.mod.saveSettings();
          await tick();
          ctxMissingRevision.fireGateTimeout();
          await tick();
          await tick();
          ctxMissingRevision.postCalls[0].resolve({
            ok: true,
            status: 200,
            headers: {
              get(name) {
                return name.toLowerCase() === 'etag'
                  ? '"conversation-settings-1"'
                  : null;
              },
            },
            json: async () => ({
              success: true,
              settings: {
                independentAsrEnabled: false,
                mergeMessagesEnabled: true,
              },
              revision: 1,
              decisions: {},
            }),
          });
          await tick();
          await tick();
          ctxMissingRevision.getCalls[0].resolve({
            ok: true,
            headers: {
              get(name) {
                return name.toLowerCase() === 'etag'
                  ? '"conversation-settings-0"'
                  : null;
              },
            },
            json: async () => ({
              success: true,
              settings: {
                independentAsrEnabled: false,
                mergeMessagesEnabled: false,
              },
              decisions: {},
              telemetryBranch: null,
            }),
          });
          await tick();
          await tick();
          assert(
            ctxMissingRevision.S.mergeMessagesEnabled === true,
            'an unversioned late GET must preserve the acknowledged local key'
          );
          ctxMissingRevision.win.focusModeEnabled = true;
          ctxMissingRevision.mod.saveSettings();
          await tick();
          await tick();
          assert(
            ctxMissingRevision.postCalls[1].headers['If-Match']
              === '"conversation-settings-1"',
            'an unversioned late GET must not downgrade the confirmed ETag'
          );
          ctxMissingRevision.postCalls[1].resolve(okPost);
          await tick();

          // ---- Scenario 6: the delayed boot GET is newer than the POST that
          // acknowledged a local edit. The newer server snapshot must win.
          const ctxNewer = makeContext();
          ctxNewer.win.mergeMessagesEnabled = true;
          ctxNewer.mod.saveSettings();
          await tick();
          ctxNewer.fireGateTimeout();
          await tick();
          await tick();
          assert(ctxNewer.postCalls.length === 1, 'the bound releases the local edit');
          ctxNewer.postCalls[0].resolve({
            ok: true,
            status: 200,
            headers: {
              get(name) {
                return name.toLowerCase() === 'etag'
                  ? '"conversation-settings-1"'
                  : null;
              },
            },
            json: async () => ({
              success: true,
              settings: {
                independentAsrEnabled: false,
                mergeMessagesEnabled: true,
              },
              revision: 1,
              decisions: {},
            }),
          });
          await tick();
          await tick();
          ctxNewer.getCalls[0].resolve({
            ok: true,
            headers: {
              get(name) {
                return name.toLowerCase() === 'etag'
                  ? '"conversation-settings-2"'
                  : null;
              },
            },
            json: async () => ({
              success: true,
              settings: {
                independentAsrEnabled: false,
                mergeMessagesEnabled: false,
              },
              revision: 2,
              decisions: {},
              telemetryBranch: null,
            }),
          });
          await tick();
          await tick();
          assert(
            ctxNewer.S.mergeMessagesEnabled === false,
            'a boot GET newer than the acknowledged POST must win'
          );
          assert(ctxNewer.postCalls.length === 2, 'the newer merge writes back once');
          ctxNewer.postCalls[1].resolve(okPost);
          await tick();

          // ---- Scenario 7: a timeout-released partial POST advances the server
          // revision before the captured boot GET returns. The older GET must
          // not roll back even fields that were never locally dirty.
          const ctxOlder = makeContext();
          ctxOlder.win.mergeMessagesEnabled = true;
          ctxOlder.mod.saveSettings();
          await tick();
          ctxOlder.fireGateTimeout();
          await tick();
          await tick();
          ctxOlder.postCalls[0].resolve({
            ok: true,
            status: 200,
            headers: {
              get(name) {
                return name.toLowerCase() === 'etag'
                  ? '"conversation-settings-2"'
                  : null;
              },
            },
            json: async () => ({
              success: true,
              settings: {
                independentAsrEnabled: false,
                mergeMessagesEnabled: true,
                slopFilterEnabled: true,
              },
              revision: 2,
              decisions: {},
            }),
          });
          await tick();
          await tick();
          assert(ctxOlder.S.slopFilterEnabled === true, 'the POST response hydrates rev2');
          ctxOlder.getCalls[0].resolve({
            ok: true,
            headers: {
              get(name) {
                return name.toLowerCase() === 'etag'
                  ? '"conversation-settings-1"'
                  : null;
              },
            },
            json: async () => ({
              success: true,
              settings: {
                independentAsrEnabled: false,
                mergeMessagesEnabled: false,
                slopFilterEnabled: false,
              },
              revision: 1,
              decisions: {},
              telemetryBranch: null,
            }),
          });
          await tick();
          await tick();
          assert(
            ctxOlder.S.mergeMessagesEnabled === true
              && ctxOlder.S.slopFilterEnabled === true,
            'a delayed older GET must not merge any stale settings fields'
          );
          assert(
            ctxOlder.postCalls.length === 1,
            'discarding an older GET must not trigger a stale writeback'
          );

          // ---- Scenario 8: a newer explicit localStorage ASR decision arrives
          // before its origin window's POST. The boot GET is older and must not
          // overwrite either the local value or the tuple that will accompany
          // the next save.
          const ctx5 = makeContext({
            independentAsrEnabled: true,
            _sharedWriteMeta: {
              writeId: 20,
              writerId: 'window-b',
              changedKeys: ['independentAsrEnabled'],
              hydrated: true,
              asrAuthoritative: true,
              asrDecision: { writeId: 20, writerId: 'window-b', value: true },
            },
          });
          ctx5.getCalls[0].resolve({
            ok: true,
            headers: { get(name) { return name.toLowerCase() === 'etag' ? '"conversation-settings-3"' : null; } },
            json: async () => ({
              success: true,
              settings: { independentAsrEnabled: false, mergeMessagesEnabled: false },
              decisions: {
                independentAsrEnabled: {
                  writeId: 10,
                  writerId: 'window-a',
                  value: false,
                },
              },
              telemetryBranch: null,
            }),
          });
          await tick();
          await tick();
          assert(
            ctx5.S.independentAsrEnabled === true,
            'an older boot GET must not overwrite the newer local ASR choice'
          );
          assert(ctx5.postCalls.length === 0, 'preserving the local winner needs no merge writeback');
          ctx5.win.focusModeEnabled = true;
          ctx5.mod.saveSettings();
          await tick();
          await tick();
          assert(ctx5.postCalls.length === 1, 'a later unrelated edit is persisted');
          const afterLocalWinner = JSON.parse(ctx5.postCalls[0].body);
          assert(
            afterLocalWinner.independentAsrEnabled === true,
            'the later full snapshot keeps the newer local ASR value'
          );
          const decisionHeader = JSON.parse(
            ctx5.postCalls[0].headers['X-Conversation-Settings-ASR-Decision']
          );
          assert(
            decisionHeader.writeId === 20
              && decisionHeader.writerId === 'window-b'
              && decisionHeader.value === true,
            'the later POST carries the newer local ASR decision tuple'
          );
          ctx5.postCalls[0].resolve(okPost);
          await tick();

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
        "failed-boot-GET dirty-only harness failed\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "HARNESS_OK" in result.stdout

def test_startup_greeting_release_event_replaces_home_tutorial_block_state():
    source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")

    assert "STARTUP_GREETING_RELEASE_EVENT = 'neko:startup-greeting-release'" in source
    assert "STARTUP_GREETING_RELEASE_FALLBACK_MS" in source
    assert "function sendStartupGreetingReleaseRequest(reason)" in source
    assert "function consumeStartupGreetingReleasedDetail()" in source
    assert "delete window.__NEKO_STARTUP_GREETING_RELEASED__" in source
    assert "const released = consumeStartupGreetingReleasedDetail()" in source
    assert "function releaseStartupGreetingCheck(reason)" in source
    assert "function hasStartupGreetingReleaseProducer()" in source
    assert "function isStartupGreetingHomePage()" not in source
    assert "function isStartupTutorialActiveForGreeting()" in source
    assert "function scheduleStartupGreetingReleaseFallback()" in source
    assert "window.addEventListener(STARTUP_GREETING_RELEASE_EVENT" in source
    assert "if (detail.released === false)" in source
    assert "releaseStartupGreetingCheck(reason || 'startup-greeting-no-release-producer')" in source
    assert "releaseStartupGreetingCheck('startup-greeting-release-timeout')" in source
    assert "scheduleStartupGreetingReleaseFallback();" in source
    assert "clearTimeout(S._startupGreetingReleaseFallbackTimer)" in source
    assert "sendHomeTutorialState(" not in source
    assert "neko:home-tutorial-features-suppressed" not in source

    active_block = source.split("function isStartupTutorialActiveForGreeting()", 1)[1].split(
        "function scheduleStartupGreetingReleaseFallback()",
        1,
    )[0]
    assert "manager.isTutorialRunning === true" in active_block
    assert "document.body.classList.contains('yui-taking-over')" in active_block
    assert "window.isInTutorial === true" not in active_block

    producer_block = source.split("function hasStartupGreetingReleaseProducer()", 1)[1].split(
        "function isStartupTutorialActiveForGreeting()",
        1,
    )[0]
    assert "window.universalTutorialManager" in producer_block
    assert "universal-manager.js" in producer_block
    assert "isStartupGreetingHomePage" not in producer_block

def test_new_user_icebreaker_mirror_turn_end_skips_regular_subtitle_finalize():
    source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")

    assert "function isNewUserIcebreakerMirrorTurnEnd(response)" in source
    helper_block = source.split("function isNewUserIcebreakerMirrorTurnEnd(response)", 1)[1].split(
        "// turn-end / turn end agent_callback",
        1,
    )[0]
    assert "meta.source === 'new_user_icebreaker'" in helper_block
    assert "meta.kind === 'new_user_icebreaker'" in helper_block
    assert "event.source === 'new_user_icebreaker'" in helper_block

    turn_end_block = source.split("// -------- system turn end --------", 1)[1].split(
        "// AI turn_end 后只 reschedule",
        1,
    )[0]
    assert "flushRealisticBufferOnTurnEnd();" in turn_end_block
    assert "emitAssistantLifecycleEvent('neko-assistant-turn-end'" in turn_end_block
    assert "clearPendingAssistantTurnStart();" in turn_end_block
    assert "if (!isNewUserIcebreakerMirrorTurnEnd(response)) {" in turn_end_block
    assert "finalizeAssistantTurn(assistantTurnId);" in turn_end_block

def test_ws_open_resyncs_goodbye_state_and_defers_regular_greeting_until_release():
    source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")

    onopen_greeting_block = source.split("// ── 首次连接 / 切换角色：标记 greeting 意图", 1)[1].split(
        "// ── game-window-state 重连兜底",
        1,
    )[0]

    assert "window.isNekoGoodbyeModeActive()" in onopen_greeting_block
    assert "window.__nekoGoodbyeSilentState" in onopen_greeting_block
    assert "pendingGoodbyeState.pending === true" in onopen_greeting_block
    assert "action: 'goodbye_state'" in onopen_greeting_block
    assert "active: !!goodbyeSyncOnOpen.active" in onopen_greeting_block
    assert "reason: 'ws-open-goodbye'" in onopen_greeting_block
    assert "pendingGoodbyeState.active === true" in onopen_greeting_block
    assert "reason: 'ws-open-goodbye-from-sync'" in onopen_greeting_block
    assert "pending: false" in onopen_greeting_block
    assert "if (goodbyeActiveOnOpen || (goodbyeSyncOnOpen && goodbyeSyncOnOpen.active))" in onopen_greeting_block
    assert "var isGreetingSwitchOnOpen = !!S._pendingGreetingSwitch;" in onopen_greeting_block
    assert "var greetingReasonOnOpen = S._greetingCheckReason || (isGreetingSwitchOnOpen ? 'character-switch' : 'ws-open');" in onopen_greeting_block
    assert "_markGreetingCheckPending(isGreetingSwitchOnOpen, greetingReasonOnOpen);" in onopen_greeting_block
    assert "if (isGreetingSwitchOnOpen || S._startupGreetingReleaseGateUsed)" in onopen_greeting_block
    assert "_sendGreetingCheckIfReady();" in onopen_greeting_block
    assert "S._startupGreetingReleaseGateUsed = true;" in onopen_greeting_block
    assert "sendStartupGreetingReleaseRequest('ws-open')" in onopen_greeting_block

def test_blocked_lifecycle_stops_microphone_capture():
    # Codex P2. _handle_core_asr_failure pins the microphone route to "blocked"
    # and nothing re-arms it inside the session, but the frontend only cleared
    # the preview and the route flag: canUploadOrdinaryMicFrame() consults the
    # mic lease and mute/focus, never the lifecycle state, so the hardware
    # microphone (and its OS indicator) stayed open and kept uploading PCM the
    # backend decodes, denoises and VADs before dropping -- while the toast on
    # the very next line says voice input has stopped.
    websocket_source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")

    # The teardown is shared with the STARTUP-failure path, which can never
    # emit a BLOCKED lifecycle event, so it lives in a helper now.
    teardown = websocket_source.split(
        "function tearDownBlockedVoiceRoute() {", 1
    )[1].split("\n    }", 1)[0]
    assert "tearDownBlockedVoiceRoute();" in _block_after(
        websocket_source, "if (lifecycleState === 'blocked') {"
    )

    # Only the capturing window acts, and never while the game STT gate owns
    # the hardware (there the ordinary uplink is already released).
    assert "S.isRecording === true" in teardown
    assert "S.gameVoiceSttGateActive !== true" in teardown
    # stopMicCapture, not bare stopRecording: only it restores the whole
    # non-recording UI rather than leaving it claiming a live voice session.
    assert "window.stopMicCapture" in teardown
    # Teardown precedes the toast so the 5s failure message stays on screen.
    assert websocket_source.index("window.stopMicCapture") < websocket_source.index(
        "microphone.independentAsrFallback"
    )

    # The uplink gate really is lease-only today, which is what makes the
    # teardown necessary. (Gating it on the lifecycle state as well would be a
    # strictly better complementary fix -- this asserts the current shape, it
    # does not forbid that.)
    capture_source = APP_AUDIO_CAPTURE_PATH.read_text(encoding="utf-8")
    can_upload = _block_after(capture_source, "function canUploadOrdinaryMicFrame() {")
    assert "refreshMicLease() !== MIC_LEASE.CORE" in can_upload

def test_pending_request_id_is_claimed_and_released_with_the_slot():
    # The id lives and dies with the shared start slot. Left behind after a
    # release, it would make the NEXT anonymous-or-foreign ack look mismatched
    # and strand a start that nothing else settles.
    state_source = APP_STATE_PATH.read_text(encoding="utf-8")
    websocket_source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")

    assert "_pendingSessionStartRequestId: null," in state_source
    claim = state_source.split("window.claimSessionStart = function (", 1)[1].split(
        "\n    };", 1
    )[0]
    assert "S._pendingSessionStartRequestId = requestId;" in claim

    # Every slot teardown clears it. Paired with the mode, which is the field
    # that already had to be cleared everywhere -- so count against that rather
    # than list the sites.
    for source in (state_source, websocket_source):
        assert source.count("S._pendingSessionStartRequestId = null;") == source.count(
            "S._pendingSessionStartMode = null;"
        )

def test_shared_write_metadata_carries_per_key_asr_authority():
    # Codex P2. meta.hydrated is the GLOBAL hydration bit, which any unrelated
    # user edit flips -- so a window whose boot GET never merged could stamp its
    # pre-merge boot ASR default as trustworthy, and a window that HAD merged
    # the server value would adopt it, mis-stamp its next handshake and POST the
    # wrong value back. The receiver needs the per-key fact instead.
    settings_source = APP_SETTINGS_PATH.read_text(encoding="utf-8")

    write_fn = settings_source.split("function _writeSharedSettings(", 1)[1].split(
        "\n    }", 1
    )[0]
    assert "asrAuthoritative: S.independentAsrAuthoritative === true" in write_fn

    read_fn = settings_source.split("function _readSharedWriteMeta(", 1)[1].split(
        "\n    }", 1
    )[0]
    # Fail closed for snapshots written by the previous build.
    assert "asrAuthoritative: meta.asrAuthoritative === true" in read_fn

    # The stale guard consults the writer's per-key authority, not its global
    # hydration bit. The RECEIVER term stays S.settingsHydrated: tightening it
    # to the per-key latch breaks the unhydrated-writer scenario already pinned
    # by test_unrelated_save_from_unhydrated_window_is_not_an_asr_toggle_harness.
    assert (
        "(!asrWriteIsNewer || !asrOutranksLocalChoice\n"
        "                    || (!meta.asrAuthoritative && S.settingsHydrated === true))"
        in settings_source
    )

def test_write_id_doc_does_not_claim_global_uniqueness():
    # The previous round's comments claimed the applied-id floor cured
    # same-millisecond minting across windows. It does not -- that is this
    # finding. A future reader must not be told otherwise by the comment they
    # hit first.
    settings_source = APP_SETTINGS_PATH.read_text(encoding="utf-8")
    id_fn = settings_source.split("function _nextSharedWriteId() {", 1)[1].split(
        "\n    }", 1
    )[0]
    assert "already OBSERVED" in id_fn
    assert "cannot be broken at mint" in id_fn
    assert "the listener resolves it on explicit intent" in id_fn

def test_concurrent_asr_toggles_are_totally_ordered_not_swapped():
    # Codex P2. The intent tie-break added last round did not fix the real
    # failure: _lastAppliedSharedWriteId only records writes RECEIVED here, so a
    # window never orders its OWN pending toggle against a concurrent one from
    # another window. Two windows holding divergent values that both write
    # before observing each other therefore each adopt the other and stay
    # swapped -- and that needs no millisecond tie at all, a strictly older
    # foreign write still wins. Ordering must be against this window's own last
    # explicit decision, with a window-unique second key so both sides pick the
    # SAME winner.
    settings_source = APP_SETTINGS_PATH.read_text(encoding="utf-8")

    # A window-unique writer id, minted per document load and stamped on writes.
    assert "const _SHARED_WRITER_ID" in settings_source
    assert "writerId: _SHARED_WRITER_ID," in settings_source
    # NOT sessionStorage: the browser copies it into a duplicated tab, which
    # would destroy the uniqueness the whole scheme rests on. Match the ACCESS
    # form -- the comment above deliberately names it.
    assert "sessionStorage." not in settings_source

    # Previous-build snapshots carry no writerId; it must fail low so an
    # untagged concurrent write cannot outrank this window's own choice.
    read_fn = settings_source.split("function _readSharedWriteMeta(", 1)[1].split(
        "\n    }", 1
    )[0]
    assert "typeof meta.writerId === 'string' ? meta.writerId : ''" in read_fn

    # The comparison is (writeId, writerId) against the local decision.
    outranks = settings_source.split("function _settingWriteOutranksLocalChoice(", 1)[
        1
    ].split("\n    }", 1)[0]
    # Ordering is on the DECISION that produced the value, not on the id of the
    # write carrying it: a monotone dirty key makes every later unrelated save
    # re-declare the ASR key explicit with a fresh id, which would outrank a
    # genuinely newer toggle elsewhere (no race required).
    assert "decision.writeId > localDecision.writeId" in outranks
    assert "(decision.writerId || '') > localDecision.writerId" in outranks
    # A write with neither a decision tuple nor an explicit declaration is an
    # incidental copy and must never outrank a local choice.
    assert "if (!decision) return false;" in outranks
    # The decision must be DERIVED (tuple, else an explicit declaration), never
    # taken as the incoming write itself -- that is the bug being fixed.
    assert "const decision = meta[decisionKey]" in outranks
    assert "const decision = meta;" not in outranks

    # A window's OWN explicit write must be recorded, or it has nothing to
    # compare a concurrent foreign toggle against.
    write_fn = settings_source.split("function _writeSharedSettings(", 1)[1].split(
        "\n    }", 1
    )[0]
    asr_note = write_fn.split("_noteAsrDecision(", 1)[1].split(");", 1)[0]
    assert "_nextAsrDecisionWriteId(ownMeta.writeId)" in asr_note
    assert "ownMeta.writerId" in asr_note
    assert "snapshot.independentAsrEnabled" in asr_note

    # Refusing authority alone is not enough: applySharedRuntimeSettings copies
    # independentAsrEnabled unconditionally, so the losing write must also be
    # dropped from the apply set.
    listener_block = settings_source.split(
        "window.addEventListener('storage', function (event) {", 1
    )[1].split("});", 1)[0]
    assert "!asrOutranksLocalChoice" in listener_block.split("asrValueIsStale", 1)[1]

def test_asr_decision_tuple_survives_unrelated_saves():
    # Codex P2. _dirtySettingsKeys is monotone, so once a window has toggled ASR
    # every LATER unrelated save still lists independentAsrEnabled in
    # changedKeys -- and used to stamp it with that save's fresh writeId. A then
    # outranks a genuinely newer toggle from B, and the two windows swap. Unlike
    # the same-millisecond tie this follows up, it needs no race at all.
    settings_source = APP_SETTINGS_PATH.read_text(encoding="utf-8")

    # The write carries the id of the decision that produced the value...
    signature = re.search(
        r"function _writeSharedSettings\((?P<params>[^)]*)\)\s*\{",
        settings_source,
    )
    assert signature is not None, "_writeSharedSettings signature is missing"
    parameter_names = {
        parameter.strip() for parameter in signature.group("params").split(",")
    }
    assert {
        "snapshot",
        "explicitKeys",
        "pendingRecovery",
        "serverAuthoritativeKeys",
    } <= parameter_names
    write_fn = _block_after(settings_source, signature.group(0))
    assert "ownMeta.asrDecision = {" in write_fn
    assert "_lastAsrDecision.value === snapshot.independentAsrEnabled" in write_fn

    # ...the reader parses it defensively, falling back to today's behaviour...
    read_fn = _block_after(settings_source, "function _readSharedWriteMeta(settings) {")
    assert "asrDecision:" in read_fn
    assert "_isValidAsrWriteId(" in read_fn
    assert "meta.asrDecision.writeId," in read_fn
    assert "Number.isInteger(meta.serverRevision)" in read_fn

    # ...and both the boot seed and the adopted cross-window flip record the
    # ORIGINAL id, or this window re-inflates the value on its own next save.
    assert "const bootDecision = bootMeta.asrDecision || bootMeta;" in settings_source
    assert "const adopted = meta.asrDecision || meta;" in settings_source

def test_auto_restart_does_not_claim_success_on_a_blocked_route():
    # The rebuilt session can come back fail-closed; startMicCapture then
    # refuses silently, and the handler would still light the floating mic,
    # toast "restart complete", and leave the button row it disabled dead.
    websocket_source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")

    restart = websocket_source.split("await sessionStartPromise;", 1)[1].split(
        "app.restartComplete", 1
    )[0]
    assert "S.voiceInputRouteBlocked === true" in restart
    assert "window.abortVoiceStartForBlockedRoute();" in restart
    # It must bail before the toast, and restore controls the restart disabled.
    assert restart.index("S.voiceInputRouteBlocked === true") < restart.index(
        "startMicCapture"
    )
    assert "resetSessionButton(); if (_rsB) _rsB.disabled = false;" in restart

def test_in_flight_microphone_start_is_cancellable():
    # Codex P2. S.isRecording only flips at the END of startAudioWorklet, after
    # getUserMedia() and audioWorklet.addModule() have both awaited, so every
    # teardown guard keyed on `S.isRecording === true` is a no-op for the whole
    # startup window. The pending start then completed, set recording true and
    # re-claimed via refreshMicLease() the lease the backend had just revoked,
    # uploading PCM into a blocked route.
    capture_source = APP_AUDIO_CAPTURE_PATH.read_text(encoding="utf-8")

    # The attempt is claimed before the first await, so the token covers the
    # getUserMedia half of the window too, not just addModule.
    start_fn = _block_after(capture_source, "async function startMicCapture() {")
    assert "micStartGeneration += 1;" in start_fn
    # PER-ATTEMPT local, never a module field. A module-level "pending token" is
    # re-armed by the NEXT startMicCapture -- attempt #1 gets invalidated, #2
    # writes token and generation to the same new value, and #1's guard compares
    # equal again and commits, re-claiming the very lease this counter protects.
    assert "const micStartToken = micStartGeneration;" in start_fn
    assert "pendingMicStartToken" not in capture_source
    # _code_only: the function's own comments mention "await", and an ordering
    # assertion that a comment can satisfy is not an ordering assertion.
    start_code = _code_only(start_fn)
    assert start_code.index("const micStartToken = micStartGeneration;") < start_code.index(
        "await"
    )

    # ...and the commit is gated on it.
    worklet = _block_after(
        capture_source,
        "async function startAudioWorklet(\n"
        "        mediaStream,\n"
        "        startToken,\n"
        "        selectedMicrophoneIdAtStart,\n"
        "        microphoneSelectionGenerationAtStart\n"
        "    ) {",
    )
    assert "startToken !== micStartGeneration" in worklet
    assert "S.voiceInputRouteBlocked === true" in worklet
    assert "S.selectedMicrophoneId !== selectedMicrophoneIdAtStart" in worklet
    assert (
        "microphoneSelectionGeneration !== microphoneSelectionGenerationAtStart"
        in worklet
    )
    # TWO gates on that token, and both are load-bearing. The entry gate stops
    # an attempt that was superseded while still in getUserMedia from running
    # the old-pipeline teardown below it, which would close the WINNER's
    # freshly published AudioContext. The commit gate stops it from publishing.
    assert worklet.count("startToken !== micStartGeneration") == 2, (
        "expected an entry gate and a commit gate on the start token"
    )
    assert worklet.count(
        "S.selectedMicrophoneId !== selectedMicrophoneIdAtStart"
    ) == 2, "expected both gates to enforce microphone-selection ownership"
    assert worklet.count(
        "microphoneSelectionGeneration !== microphoneSelectionGenerationAtStart"
    ) == 2, "expected both gates to preserve intermediate selection changes"
    assert worklet.index("superseded before opening") < worklet.index(
        "await previousContext.close()"
    ), "the entry gate must precede the old-pipeline teardown it protects"
    assert worklet.index("superseded while opening") < worklet.index(
        "S.isRecording = true;"
    )
    # The unwind must NOT re-emit a lease snapshot -- that re-claim is the bug.
    #
    # Sliced from the COMMIT gate's own log line, not from the first
    # occurrence of the token comparison: the entry gate added a second one,
    # and anchoring on the first silently widened this slice to the whole
    # function body, where both assertions below pass for free.
    unwind = worklet.split("superseded while opening", 1)[1].split(
        "S.isRecording = true;", 1
    )[0]
    assert "refreshMicLease()" not in _code_only(unwind)

    # A superseded attempt must REPORT that it unwound. A bare `return` left
    # startMicCapture running its whole success path -- disabling the mic
    # button, toasting "speaking", lighting the floating button and silencing
    # proactive chat -- against hardware the unwind had just torn down.
    assert "return false;" in _code_only(unwind)
    assert "return true;" in _code_only(worklet)
    start_code_only = _code_only(start_fn)
    # The stream is attempt-local now (it used to be published into S.stream
    # before the token gate, where a loser whose getUserMedia settled last
    # could take the slot and then null it out from under the winner), so the
    # handoff goes through the local binding.
    assert "const selectedMicrophoneIdAtStart = S.selectedMicrophoneId;" in start_code_only
    compact_start_code = "".join(start_code_only.split())
    assert (
        "constmicStartCommitted=awaitstartAudioWorklet("
        "ownStream,micStartToken,selectedMicrophoneIdAtStart,"
        "microphoneSelectionGenerationAtStart);"
        in compact_start_code
    )
    assert "if (!micStartCommitted) {" in start_code_only
    # ...and the bail happens before every success-path side effect.
    bail = start_code_only.index("if (!micStartCommitted) {")
    for success_marker in (
        "'app.speaking'",
        "window.syncFloatingMicButtonState(true)",
        "updateMicVolumeStatusNow(true)",
        "window.stopProactiveChatSchedule()",
    ):
        assert bail < start_code_only.index(success_marker), success_marker

    # The fail-closed unwind cancels a pending start as well as a live one.
    abort_fn = _block_after(capture_source, "function abortVoiceStartForBlockedRoute() {")
    assert "invalidatePendingMicStart();" in abort_fn

def test_reconnect_reconciliation_repairs_appstate_not_only_the_dom_event():
    """chat.html has no listener that writes S.gameRoute*, so the shared
    reconnect path must repair appState itself.

    The DOM event dispatched here is only turned into appState by
    app-game-voice-control.js, and templates/index.html is the only page that
    loads that file. Without an appState write in this block, a chat window that
    reconnects or reloads while a game route is active keeps gameRouteActive
    false for the rest of the round, and the reverse desync survives too.
    """
    source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")
    reconnect_block = _block_after(
        source,
        "function syncGameWindowStateOnWsConnect() {",
    )

    assert "neko-game-window-state-change" in reconnect_block
    for field in (
        "S.gameRouteActive",
        "S.gameRouteGameType",
        "S.gameRouteSessionId",
        "S.gameRouteInstanceId",
    ):
        assert field in reconnect_block, (
            f"reconnect reconciliation does not repair {field}; chat.html has no "
            "other writer for it"
        )
    # Ordering matters: the repair must follow the dispatch, or index.html's
    # voice bridge observes already-cleared identity and broadcasts a closed
    # route with an empty session id. Every assignment, not just the flag --
    # a repair that cleared the identity fields early would produce exactly the
    # empty-session_id broadcast this ordering exists to prevent, and an
    # assertion on the flag alone would not notice.
    dispatch_at = reconnect_block.index("neko-game-window-state-change")
    for assignment in (
        "S.gameRouteActive = true",
        "S.gameRouteGameType = data.game_type",
        "S.gameRouteSessionId = data.session_id",
        "S.gameRouteInstanceId = data.sdk_route_instance_id",
        "S.gameRouteActive = false",
        "S.gameRouteGameType = ''",
        "S.gameRouteSessionId = ''",
        "S.gameRouteInstanceId = ''",
    ):
        assert assignment in reconnect_block, assignment
        assert dispatch_at < reconnect_block.index(assignment), (
            f"appState repair ({assignment}) must follow the dispatch so the "
            "voice bridge keeps its ordering"
        )

def test_game_voice_command_commits_its_teardown_before_it_can_yield():
    """The microphone teardown must be issued while the admitting check still holds.

    ``stopMicCapture()`` is process-global. If a command could issue it after
    awaiting, a stop belonging to a route that has since been superseded would
    land on whatever owns the microphone by then -- the replacement route, or
    the ordinary chat capture the host resumes on route exit -- and kill it
    mid-utterance with no transcript and nothing logged.

    Two properties keep that unreachable, and both are easy to lose in an edit:
      1. ``routeMatches()`` admits the command and the single ``stopMicCapture()``
         call sits in the same synchronous segment -- no ``await`` between them,
         so no route change can interleave.
      2. Nothing after the awaited command tears anything down. The
         route-superseded branch reports and returns; it never issues a
         teardown, and never re-starts. (The runtime harness in
         tests/frontend/test_game_voice_control_runtime.js asserts the
         behaviour; this pins the ordering the behaviour depends on.)
    """
    source = APP_GAME_VOICE_CONTROL_PATH.read_text(encoding="utf-8")

    stop_body = _block_after(source, "async function stopOfficialVoiceSession() {")
    assert stop_body.count("stopMicCapture(") == 1, (
        "the stop helper issues more than one microphone teardown"
    )
    assert stop_body.index("stopMicCapture(") < stop_body.index("waitFor("), (
        "the microphone teardown is issued after this helper has already yielded, "
        "so it can land on a capture the command never opened"
    )

    handler = _block_after(source, "async function handleRequest(request) {")
    admit_at = handler.index("if (!routeMatches(request))")
    dispatch_at = handler.index("await startOfficialVoiceSession()")
    admitted_segment = handler[admit_at:dispatch_at]
    assert "await" not in admitted_segment, (
        "an await was introduced between the route check that admits a voice "
        "command and the command dispatch, so the route can change underneath it"
    )

    superseded_at = handler.index("if (!routeSnapshotIsCurrent(acceptedRoute))")
    superseded_branch = handler[superseded_at:handler.index("broadcastState({", superseded_at)]
    # Code only: this branch carries a long comment explaining which mechanisms
    # it deliberately does NOT reach for, and those names must not trip the check.
    superseded_code = chr(10).join(
        line for line in superseded_branch.splitlines()
        if not line.strip().startswith("//")
    )
    for teardown in ("stopMicCapture(", "startMicCapture(", ".click("):
        assert teardown not in superseded_code, (
            f"the superseded branch reaches for {teardown}; a command whose route "
            "is gone must not touch the process-global microphone"
        )
