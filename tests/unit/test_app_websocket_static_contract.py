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

def test_independent_asr_terminal_status_clears_partial_preview():
    source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")

    status_block = source.split(
        "if (statusCode && statusCode.indexOf('ASR_INDEPENDENT_') === 0)",
        1,
    )[1].split("if (statusCode === 'TTS_CONNECTION_FAILED')", 1)[0]
    terminal_branch = status_block.split(
        "if (statusCode === 'ASR_INDEPENDENT_INJECTION_FAILED')",
        1,
    )[1]

    # The preview clear and the route-flag reset now live in the shared
    # teardown helper (it is also used by the startup-failure path, which can
    # never emit a BLOCKED lifecycle event). The terminal tail must call it
    # before showing its per-code toast.
    assert terminal_branch.index("tearDownBlockedVoiceRoute();") < terminal_branch.index(
        "showStatusToast"
    )
    teardown_fn = source.split("function tearDownBlockedVoiceRoute() {", 1)[1].split(
        "\n    }", 1
    )[0]
    assert "removeExternalAsrPreview();" in teardown_fn
    assert "S.independentAsrActive = false;" in teardown_fn

def test_independent_asr_terminal_status_reports_stopped_voice_input():
    source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")

    assert "Independent ASR unavailable; using Omni native recognition" not in source
    assert "Voice input has stopped for this session" in source

def test_provider_unavailable_status_names_provider_and_denies_silent_switch():
    source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")

    assert "ASR_INDEPENDENT_PROVIDER_UNAVAILABLE" in source
    assert "microphone.independentAsrProviderUnavailable" in source
    assert "{ providerKey: asrProvider || 'unknown' }" in source
    assert "It did not switch to another speech recognition service" in source

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

def test_lease_resync_status_resends_snapshot_only_from_capturing_window():
    source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")
    capture_source = APP_AUDIO_CAPTURE_PATH.read_text(encoding="utf-8")

    resync_branch = source.split(
        "if (statusCode === 'VOICE_INPUT_LEASE_RESYNC_REQUIRED')",
        1,
    )[1].split("if (statusCode && statusCode.indexOf('ASR_INDEPENDENT_') === 0)", 1)[0]

    assert "S.isRecording === true" in resync_branch
    assert "window.appAudioCapture.sendVoiceInputControlState(true);" in resync_branch
    assert resync_branch.index("S.isRecording === true") < resync_branch.index(
        "window.appAudioCapture.sendVoiceInputControlState(true);"
    )
    assert "return;" in resync_branch
    assert "setInterval" not in resync_branch
    assert "setTimeout" not in resync_branch
    assert "mod.sendVoiceInputControlState = sendVoiceInputControlState;" in capture_source

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

def test_external_asr_preview_message_is_declared_app_state_field():
    app_state = APP_STATE_PATH.read_text(encoding="utf-8")

    assert "externalAsrPreviewMessage: null," in app_state

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

def test_empty_preview_message_clears_streaming_preview_bubble():
    # Codex P2: a turn that ends with an EMPTY final (OpenAI/Step stalled-item
    # timeouts) deliberately injects no user_transcript, yet user_transcript
    # was the only per-turn message removing the streaming preview bubble —
    # it lingered forever and got reused by the next turn. The backend now
    # sends user_transcript_preview with empty text as an explicit clear
    # (asr_runtime.py _send_core_asr_preview_clear); pin the handler split.
    source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")

    event_block = source.split("// -------- user_transcript_preview", 1)[1].split(
        "// -------- user_transcript --------", 1
    )[0]

    # Empty text removes the preview...
    empty_branch = event_block.split("if (externalPreviewText === '') {", 1)[1].split(
        "} else {", 1
    )[0]
    assert "removeExternalAsrPreview();" in empty_branch
    assert "upsertExternalAsrPreview" not in empty_branch

    # ... and ONLY empty text: non-empty partials still upsert (negative:
    # the upsert sits in the else branch, so a clear can never spawn a new
    # empty bubble and a partial can never be dropped).
    else_branch = event_block.split("} else {", 1)[1]
    assert (
        "S.externalAsrPreviewMessage = upsertExternalAsrPreview(externalPreviewText);"
        in else_branch
    )
    assert "removeExternalAsrPreview" not in else_branch
    assert event_block.count("upsertExternalAsrPreview(") == 1

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

def test_user_dirty_keys_survive_boot_get_merge_field_level():
    # Codex P2 (field-level authority): the conversation-settings GET may read
    # the server BEFORE a user change POSTs its new value, yet resolve AFTER
    # it. The earlier whole-merge-drop design discarded the ENTIRE server
    # merge as soon as ANY userInitiated change happened while the GET was in
    # flight — so changing one unrelated preference made the full local
    # snapshot (including a boot-default independentAsrEnabled) authoritative
    # and the POST clobbered the persisted ASR choice. Pin the replacement:
    # a dirty-key set records exactly which settings the user changed, and the
    # merge applies server values to NON-dirty keys while preserving dirty
    # ones.
    settings_source = APP_SETTINGS_PATH.read_text(encoding="utf-8")

    # (1) Dirty keys are recorded only inside the userInitiated gate of
    # syncSettingsToServer, synchronously alongside the hydration mark and
    # before any await.
    sync_fn = settings_source.split(
        "async function syncSettingsToServer(options)", 1
    )[1].split("function startPeriodicSync()", 1)[0]
    assert sync_fn.count("_markUserDirtySettings();") == 1
    user_initiated_gate = _block_after(sync_fn, "if (userInitiated) {")
    assert "_markUserDirtySettings();" in user_initiated_gate
    assert sync_fn.index("_markUserDirtySettings();") < sync_fn.index(
        "await _fetchConversationSettingsJsonWithTimeout("
    )

    # (2) loadSettings snapshots the pre-GET settings as the diff baseline
    # before issuing the GET, so keys changed while it is pending diverge
    # from the snapshot and get marked dirty.
    load_fn = settings_source.split("function loadSettings()", 1)[1]
    snapshot_index = load_fn.index("_settingsBaseline = getConversationSettings();")
    get_index = load_fn.index("loadSettingsFromServer().then(serverResult => {")
    assert snapshot_index < get_index

    # (3) The merge is field-level: pending keys are always preserved, while
    # acknowledged dirty keys yield only to a newer server revision. Both
    # guards run before the S mutation, the subtitle-bridge mirrors carry the
    # same gating, and
    # the baseline is rolled to the merged state BEFORE the writeback
    # saveSettings() so server-applied values are never misattributed as
    # user-dirty by the writeback's own userInitiated diff.
    merge_block = settings_source.split(
        "loadSettingsFromServer().then(serverResult => {", 1
    )[1].split(".finally(", 1)[0]
    null_guard_index = merge_block.index("if (!serverResult) return;")
    hydrate_index = merge_block.index("S.settingsHydrated = true;")
    assert null_guard_index < hydrate_index
    pending_skip_index = merge_block.index(
        "if (_pendingSettingsKeys.has(key)) continue;"
    )
    dirty_guard_index = merge_block.index("if (_dirtySettingsKeys.has(key)")
    mutation_index = merge_block.index("S[key] = serverSettings[key];")
    assert pending_skip_index < dirty_guard_index < mutation_index
    assert "&& !serverSnapshotNewerThanCurrent) continue;" in merge_block
    assert "!_pendingSettingsKeys.has('subtitleEnabled')" in merge_block
    assert "!_pendingSettingsKeys.has('userLanguage')" in merge_block
    assert "serverSnapshotNewerThanCurrent" in merge_block
    roll_index = merge_block.index("_settingsBaseline = getConversationSettings();")
    assert pending_skip_index < roll_index < merge_block.index("saveSettings({")
    assert "serverAuthoritativeKeys: Object.keys(" in merge_block
    # The whole-merge drop is gone: no early return between the null-guard
    # and the hydration mark, and the old drop log no longer exists.
    after_null_guard = null_guard_index + len("if (!serverResult) return;")
    assert "return;" not in merge_block[after_null_guard:hydrate_index]
    assert "丢弃过期的服务器合并" not in settings_source
    assert "_localSettingsGeneration" not in settings_source

    # (4) Negative validation — non-user flows never dirty keys: the periodic
    # tick passes no options (its POST is not a user change), the boot-time
    # skipServerSync save bypasses syncSettingsToServer entirely, and the
    # boot-merge authority set is monotone so a toggle-and-back survives a
    # stale in-flight GET. The separate pending set is cleared only after a
    # successful POST and only while the acknowledged value is still current.
    tick_body = settings_source.split("_syncTimerId = setInterval(() => {", 1)[1].split(
        "}, SYNC_INTERVAL_MS);", 1
    )[0]
    assert "_markUserDirtySettings" not in tick_body
    assert "_dirtySettingsKeys" not in tick_body
    first_launch_block = settings_source.split(
        "console.log('未找到保存的设置，使用默认值');", 1
    )[1].split("} catch (error) {", 1)[0]
    assert "_dirtySettingsKeys" not in first_launch_block
    assert "saveSettings({ skipServerSync: true });" in first_launch_block
    assert "_dirtySettingsKeys.delete" not in settings_source
    assert "_dirtySettingsKeys.clear" not in settings_source
    clear_fn = settings_source.split(
        "function _clearAcknowledgedPendingSettings(payload) {", 1
    )[1].split("function applySharedRuntimeSettings", 1)[0]
    assert "current[key] === payload[key]" in clear_fn
    assert "_pendingSettingsKeys.delete(key);" in clear_fn
    assert "_clearAcknowledgedPendingSettings(payload);" in sync_fn

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

def test_boot_merge_orders_asr_on_the_decision_tuple_not_only_the_dirty_mark():
    # Structural half of the #2540 pin. The behavioural harness above can only
    # observe the outcome; this asserts the boot path actually keeps the two
    # inputs the issue said were conflated -- "I never changed this key" and
    # "I never had an authoritative value for it" -- separate.
    settings_source = APP_SETTINGS_PATH.read_text(encoding="utf-8")

    load_fn = _block_after(settings_source, "function loadSettings() {")
    # The boot snapshot restores the decision tuple, so a flip a peer wrote
    # before this window loaded is still ordered against the server's.
    assert "bootMeta.asrDecision" in load_fn
    assert "_adoptAsrDecisionTuple(" in load_fn
    assert "const bootDecision = bootMeta.asrDecision || bootMeta;" in load_fn

    # The field-level merge skips the ASR key on decision ordering, which is
    # evaluated INDEPENDENTLY of _dirtySettingsKeys.
    assert (
        "if (key === 'independentAsrEnabled' && preserveLocalAsrDecision) continue;"
        in load_fn
    )
    preserve = load_fn.split("const preserveLocalAsrDecision =", 1)[1].split(
        ";", 1
    )[0]
    assert "_lastAsrDecision" in preserve
    assert "_asrDecisionOutranks(_lastAsrDecision, serverAsrDecision)" in preserve
    # A server snapshot with no decision tuple carries no ordering evidence:
    # the local decision must win rather than the absence counting as newer.
    assert "!serverAsrDecision" in preserve
    assert "_dirtySettingsKeys" not in preserve

def test_unrelated_change_during_pending_get_preserves_server_asr_harness():
    # Behavioral pin for the field-level authority fix (Codex P2): with the
    # old whole-merge-drop, changing ANY unrelated preference while the boot
    # settings GET was pending made the full saveSettings() POST authoritative
    # — the entire server merge was discarded and the POST (built from local
    # state including the boot-default independentAsrEnabled=false) overwrote
    # the persisted ASR choice. Drive the real module: the user POST must be
    # gated until the GET settles, the merge must hydrate the untouched ASR
    # key from the server while preserving the user's dirty key, and every
    # POST body must then carry the server's ASR value. On the pre-fix code
    # these assertions fail (the POST fires immediately with ASR=false and the
    # merge is dropped wholesale). Second scenario: the ASR-toggle-while-
    # pending flow is unchanged — the toggled key stays authoritative over the
    # stale merge and its POST carries the user's choice.
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
          const sandbox = {
            console: { log() {}, warn() {}, error() {} },
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
          };
        }

        const okPost = { ok: true, json: async () => ({ success: true }) };
        const tick = () => new Promise((resolve) => setImmediate(resolve));

        async function main() {
          // Scenario 1: unrelated preference changed while the boot GET is
          // pending; the server holds independentAsrEnabled=true, this boot
          // only has the default false.
          const ctx = makeContext();
          assert(ctx.getCalls.length === 1, 'boot must issue the settings GET');

          ctx.win.mergeMessagesEnabled = true; // the settings-popup mirror
          ctx.mod.saveSettings();              // full user path -> userInitiated POST
          assert(ctx.S.settingsHydrated === true, 'a user change still hydrates synchronously');
          await tick();
          assert(ctx.postCalls.length === 0, 'the user POST must wait (bounded) for the pending GET, not fire with boot defaults');

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

          assert(ctx.S.independentAsrEnabled === true, 'the untouched ASR key must hydrate from the server, not stay a boot default');
          assert(ctx.S.mergeMessagesEnabled === true, 'the user-changed (dirty) key must survive the merge');
          assert(ctx.postCalls.length === 1, 'the gated user POST goes out once the GET settled');
          const body1 = JSON.parse(ctx.postCalls[0].body);
          assert(body1.independentAsrEnabled === true, 'the send-time snapshot must carry the SERVER ASR value, not the boot default');
          assert(body1.mergeMessagesEnabled === true, 'the send-time snapshot must carry the user change');

          // The merge writeback POST (queued behind the user POST) carries the
          // same merged state, converging the server.
          ctx.postCalls[0].resolve(okPost);
          await tick();
          await tick();
          assert(ctx.postCalls.length === 2, 'the merge writeback POST follows the user POST');
          const body2 = JSON.parse(ctx.postCalls[1].body);
          assert(body2.independentAsrEnabled === true, 'the writeback keeps the server ASR value');
          assert(body2.mergeMessagesEnabled === true, 'the writeback keeps the user change');
          ctx.postCalls[1].resolve(okPost);
          await tick();

          // Scenario 2: the ASR-toggle-while-GET-pending flow is unchanged —
          // the toggled key is dirty, so the stale merge cannot revert it and
          // its POST carries the user's choice.
          const ctx2 = makeContext();
          ctx2.S.independentAsrEnabled = true;
          ctx2.mod.saveSettings({ skipServerSync: true });
          const p = ctx2.mod.syncSettingsToServer({ userInitiated: true });
          assert(ctx2.S.settingsHydrated === true, 'the toggle must hydrate synchronously at call time');
          await tick();
          assert(ctx2.postCalls.length === 0, 'the toggle POST is gated behind the pending GET too');

          ctx2.getCalls[0].resolve({
            ok: true,
            json: async () => ({
              success: true,
              settings: { independentAsrEnabled: false },
              telemetryBranch: null,
            }),
          });
          await tick();
          await tick();
          assert(ctx2.S.independentAsrEnabled === true, 'the stale merge must not revert the user toggle');
          assert(ctx2.postCalls.length === 1, 'a merge that only skipped the dirty key must not add a writeback POST');
          assert(JSON.parse(ctx2.postCalls[0].body).independentAsrEnabled === true, 'the toggle POST carries the user choice');
          ctx2.postCalls[0].resolve(okPost);
          await p;

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
        "unrelated-change-during-pending-GET harness failed\n"
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

def test_unsynced_optimization_decision_survives_reload_until_posted_harness():
    """A persisted explicit choice stays authoritative until its POST succeeds."""
    harness = textwrap.dedent(
        """
        const fs = require('node:fs');
        const vm = require('node:vm');

        const source = fs.readFileSync(__APP_SETTINGS_PATH__, 'utf8');
        const optimizationKey = 'voiceInputResourceOptimizationEnabled';

        function assert(cond, msg) {
          if (!cond) throw new Error('ASSERT: ' + msg);
        }

        function makeContext(initialSnapshot) {
          let stored = initialSnapshot ? JSON.stringify(initialSnapshot) : null;
          const postCalls = [];
          const getCalls = [];
          const sandbox = {
            console: { log() {}, warn() {}, error() {} },
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
                return key === 'project_neko_settings' ? stored : null;
              },
              setItem(key, value) {
                if (key === 'project_neko_settings') stored = value;
              },
              removeItem() {},
            },
            document: { getElementById() { return null; } },
            fetch(url, opts) {
              return new Promise((resolve, reject) => {
                if (opts && opts.method === 'POST') {
                  postCalls.push({ body: opts.body, resolve, reject });
                } else {
                  getCalls.push({ resolve, reject });
                }
              });
            },
          };
          sandbox.window = {
            appState: {
              independentAsrEnabled: true,
              settingsHydrated: false,
              independentAsrAuthoritative: false,
              voiceInputResourceOptimizationEnabled: true,
              voiceInputResourceOptimizationAuthoritative: false,
            },
            appConst: {},
            appUtils: { mapRenderQualityToFollowPerf() { return 'medium'; } },
            addEventListener() {},
            removeEventListener() {},
            dispatchEvent() {},
          };
          vm.createContext(sandbox);
          vm.runInContext(source, sandbox);
          return {
            getCalls,
            postCalls,
            S: sandbox.window.appState,
            mod: sandbox.window.appSettings,
            snapshot() { return JSON.parse(stored); },
          };
        }

        const tick = () => new Promise((resolve) => setImmediate(resolve));
        const okPost = { ok: true, json: async () => ({ success: true }) };

        async function main() {
          // This is a snapshot written by the previous PR head: it records the
          // explicit decision but predates the durable pending-sync marker.
          const legacyPendingSnapshot = {
            [optimizationKey]: false,
            _sharedWriteMeta: {
              writeId: 41,
              writerId: 'window-a',
              changedKeys: [optimizationKey],
              hydrated: true,
              asrAuthoritative: false,
              optimizationDecision: {
                writeId: 41,
                writerId: 'window-a',
                value: false,
              },
            },
          };

          // If the boot GET also fails, pre-merge sync must still retry the
          // durable decision. `_pickDirtySettings()` reads only the pending
          // set, so restoring just dirty membership would produce no POST.
          const offline = makeContext(legacyPendingSnapshot);
          offline.getCalls[0].resolve({ ok: false });
          await tick();
          await tick();
          const offlineSync = offline.mod.syncSettingsToServer();
          await tick();
          assert(
            offline.postCalls.length === 1,
            'failed boot GET must not forget the pending optimization POST'
          );
          assert(
            JSON.parse(offline.postCalls[0].body)[optimizationKey] === false,
            'dirty-only retry must carry the durable optimization choice'
          );
          offline.postCalls[0].resolve(okPost);
          await offlineSync;

          const ctx = makeContext(legacyPendingSnapshot);
          assert(ctx.S[optimizationKey] === false, 'boot must load the local choice');
          assert(ctx.S.settingsHydrated === true, 'pending choice must hydrate the handshake');
          assert(
            ctx.S.voiceInputResourceOptimizationAuthoritative === true,
            'pending choice must be authoritative for the next start handshake'
          );

          ctx.getCalls[0].resolve({
            ok: true,
            json: async () => ({
              success: true,
              settings: { [optimizationKey]: true },
              telemetryBranch: null,
            }),
          });
          await tick();
          await tick();
          assert(
            ctx.S[optimizationKey] === false,
            'stale server GET must not overwrite the unsynced local choice'
          );

          const sync = ctx.mod.syncSettingsToServer();
          await tick();
          assert(ctx.postCalls.length === 1, 'pending choice must be POSTed after reload');
          assert(
            JSON.parse(ctx.postCalls[0].body)[optimizationKey] === false,
            'POST must carry the pending local choice'
          );
          ctx.postCalls[0].resolve(okPost);
          await sync;
          const syncedSnapshot = ctx.snapshot();
          assert(
            syncedSnapshot._sharedWriteMeta.optimizationDecisionPendingSync === false,
            'successful POST must durably clear the pending marker'
          );

          // Once synchronization is durable, a later reload may accept newer
          // server truth instead of pinning the old local choice forever.
          const reloaded = makeContext(syncedSnapshot);
          reloaded.getCalls[0].resolve({
            ok: true,
            json: async () => ({
              success: true,
              settings: { [optimizationKey]: true },
              telemetryBranch: null,
            }),
          });
          await tick();
          await tick();
          assert(
            reloaded.S[optimizationKey] === true,
            'synced decision must no longer block server truth on a later reload'
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
        "unsynced-optimization-reload harness failed\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "HARNESS_OK" in result.stdout

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

def test_failure_paths_keep_status_provided_asr_provider():
    # Negative counterpart to the teardown reset: failure paths receive the
    # provider from the status event and must keep it for the toasts/hint,
    # so only the normal teardown clears S.independentAsrProvider.
    source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")

    lifecycle_block = source.split("if (statusCode === 'ASR_LIFECYCLE_STATE')", 1)[1].split(
        "if (statusCode === 'VOICE_INPUT_LEASE_RESYNC_REQUIRED')",
        1,
    )[0]
    blocked_branch = lifecycle_block.split("if (lifecycleState === 'blocked')", 1)[1]
    assert "S.independentAsrProvider = ''" not in blocked_branch

    prefix_block = source.split(
        "if (statusCode && statusCode.indexOf('ASR_INDEPENDENT_') === 0)",
        1,
    )[1].split("if (statusCode === 'TTS_CONNECTION_FAILED')", 1)[0]
    assert "S.independentAsrProvider = asrProvider;" in prefix_block
    assert "S.independentAsrProvider = ''" not in prefix_block

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

def test_session_ended_by_server_stops_assistant_text_output():
    source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")
    app_state = APP_STATE_PATH.read_text(encoding="utf-8")

    assert "suppressAssistantStreamUntilNextSession: false," in app_state
    helper_block = source.split("function stopAssistantTextOutputOnSessionEnd(source)", 1)[1].split(
        "window.addEventListener('neko-assistant-turn-start'",
        1,
    )[0]
    assert "S.suppressAssistantStreamUntilNextSession = true;" in helper_block
    assert "window._realisticGeminiVersion = (window._realisticGeminiVersion || 0) + 1;" in helper_block
    assert "window._realisticGeminiQueue = [];" in helper_block
    assert "window._realisticGeminiBuffer = '';" in helper_block
    assert "window._geminiTurnFullText = '';" in helper_block
    assert "window._isProcessingRealisticQueue = false;" in helper_block
    assert "window._realisticProcessingOwner = null;" in helper_block
    assert "window.setReactMessageStatus(bubble, 'assistant', 'sent');" in helper_block
    assert "window._clearPendingHostMessagesByIds(currentBubbleIds);" in helper_block
    assert "window.currentGeminiMessage = null;" in helper_block
    assert "window.currentTurnGeminiBubbles = [];" in helper_block

    rollback_helper = source.split("function clearPendingRollbackForRequest(requestId)", 1)[1].split(
        "function isNewUserIcebreakerMirrorTurnEnd(response)",
        1,
    )[0]
    assert "window.reactChatWindowHost.clearPendingRollbackDraft(requestId);" in rollback_helper
    assert "window._lastSubmittedRequestId === requestId" in rollback_helper
    assert "window._lastSubmittedText = '';" in rollback_helper
    assert "window._lastSubmittedRequestId = '';" in rollback_helper

    session_ended_block = source.split("// -------- session_ended_by_server --------", 1)[1].split(
        "// -------- reload_page --------",
        1,
    )[0]
    assert "stopAssistantTextOutputOnSessionEnd('session_ended_by_server');" in session_ended_block
    assert session_ended_block.index("stopAssistantTextOutputOnSessionEnd('session_ended_by_server');") < session_ended_block.index(
        "clearAssistantLifecycleOnDisconnect('session_ended_by_server');"
    )

    gemini_block = source.split("// -------- gemini_response --------", 1)[1].split(
        "// -------- response_discarded --------",
        1,
    )[0]
    assert "if (S.suppressAssistantStreamUntilNextSession)" in gemini_block
    assert gemini_block.index("if (S.suppressAssistantStreamUntilNextSession)") < gemini_block.index(
        "window.appendMessage(response.text, 'gemini', isNewMessage)"
    )
    assert "return;" in gemini_block.split("if (S.suppressAssistantStreamUntilNextSession)", 1)[1].split(
        "var isNewMessage",
        1,
    )[0]

    discard_block = source.split("// -------- response_discarded --------", 1)[1].split(
        "// -------- summary_response --------",
        1,
    )[0]
    assert "if (S.suppressAssistantStreamUntilNextSession)" in discard_block
    assert discard_block.index("if (S.suppressAssistantStreamUntilNextSession)") < discard_block.index(
        "// Fallback: clear trailing gemini bubbles not tracked"
    )
    assert "return;" in discard_block.split("if (S.suppressAssistantStreamUntilNextSession)", 1)[1].split(
        "emitAssistantSpeechCancel('response_discarded');",
        1,
    )[0]

    session_started_block = source.split("// -------- session_started --------", 1)[1].split(
        "// -------- session_failed --------",
        1,
    )[0]
    assert "S.suppressAssistantStreamUntilNextSession = false;" in session_started_block

    agent_callback_turn_end_block = source.split("// -------- system turn end (agent_callback", 1)[1].split(
        "// -------- system turn end --------",
        1,
    )[0]
    assert "if (S.suppressAssistantStreamUntilNextSession)" in agent_callback_turn_end_block
    assert agent_callback_turn_end_block.index("if (S.suppressAssistantStreamUntilNextSession)") < agent_callback_turn_end_block.index(
        "flushRealisticBufferOnTurnEnd();"
    )
    assert agent_callback_turn_end_block.index("clearPendingRollbackForRequest(response.request_id);") < agent_callback_turn_end_block.index(
        "clearPendingAssistantTurnStart();"
    )

    turn_end_block = source.split("// -------- system turn end --------", 1)[1].split(
        "// AI turn_end 后只 reschedule",
        1,
    )[0]
    assert "if (S.suppressAssistantStreamUntilNextSession)" in turn_end_block
    assert turn_end_block.index("if (S.suppressAssistantStreamUntilNextSession)") < turn_end_block.index(
        "flushRealisticBufferOnTurnEnd();"
    )
    assert turn_end_block.index("clearPendingRollbackForRequest(response.request_id);") < turn_end_block.index(
        "clearPendingAssistantTurnStart();"
    )

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

def test_equal_write_ids_are_broken_by_explicit_asr_intent():
    # Codex P2 follow-up. The applied-id floor in _nextSharedWriteId only rises
    # once this window has APPLIED another window's write, so two windows saving
    # in the same millisecond before either processes the other's storage event
    # still mint the same id. With a strict `>` freshness test the second write
    # reads as superseded and its ASR value is dropped -- and the value dropped
    # is a genuine, explicitly-marked toggle, not an incidental copy. Concurrent
    # writes have no clock order, so the tie is broken on intent instead, which
    # makes both delivery orders converge on the user's choice.
    settings_source = APP_SETTINGS_PATH.read_text(encoding="utf-8")

    listener_block = settings_source.split(
        "window.addEventListener('storage', function (event) {", 1
    )[1].split("});", 1)[0]

    assert (
        "|| (meta.writeId === _lastAppliedSharedWriteId && asrMarkedExplicit)"
        in listener_block
    )
    # A strictly OLDER write must still be refused.
    assert "meta.writeId > _lastAppliedSharedWriteId" in listener_block
    # The applied floor must advance only on a strict `>`, so a tie does not
    # consume the id and both tied writes stay eligible.
    assert "if (meta && meta.writeId > _lastAppliedSharedWriteId) {" in listener_block

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

def test_status_fanout_comment_states_the_real_delivery_contract():
    # An earlier round shipped a comment claiming status "fans out to every
    # window". It does not: send_status targets the manager's current socket,
    # and sync_message_queue feeds the monitor process on a port no app window
    # connects to. The fix routes mic control-plane codes to the lease holder
    # instead, and the comment must say so or the next reader repeats the
    # mistake.
    websocket_source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")
    assert "fans out to every window" not in websocket_source
    assert "_send_to_voice_owner" in websocket_source

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

def test_server_side_teardowns_do_not_send_pause_session():
    # A pause_session from a SUPERSEDED recorder socket is not a voice-path
    # message, so the router reads it as a character switch, closes that socket,
    # and its 3s auto-reconnect re-steals the session identity from the window
    # that legitimately owns it. Both server-initiated teardowns must stop the
    # capture without notifying.
    websocket_source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")

    ended_handler = websocket_source.split(
        "} else if (response.type === 'session_ended_by_server') {", 1
    )[1].split("} else if (response.type ===", 1)[0]
    assert "window.stopRecording({ notifyServer: false })" in ended_handler

    auto_close = _block_after(
        websocket_source, "async function resetVoiceUiAfterAutoClose(options) {"
    )
    # Drop recording first so stopMicCapture's own bare stopRecording() hits its
    # !S.isRecording early return and never reaches the pause_session send.
    assert "window.stopRecording({ notifyServer: false });" in auto_close
    assert auto_close.index("window.stopRecording({ notifyServer: false });") < auto_close.index(
        "await window.stopMicCapture();"
    )

def test_deferred_session_start_resolve_is_pinned_to_the_ack_it_belongs_to():
    # Codex P2, twice. A matching session_started clears the start timeout
    # immediately but defers the resolve by 500ms to let the UI settle. The
    # resolver lives in a SHARED slot, and on mobile the composer stays visible
    # during an audio session (the `_shouldHide` guard excludes mobile), so the
    # user can send text inside that window and app-buttons.js then installs a
    # new resolver + mode for the text start.
    #
    # Both halves are load-bearing, and they pull in opposite directions:
    #   * the SLOT must only be cleared while it still holds this ack's start,
    #     or the old audio timer resolves the newer text promise and lets a
    #     queued message go out before the backend acknowledged it;
    #   * the PROMISE must be settled regardless, because its timeout was
    #     already cleared at ack time -- gating the settle on identity too left
    #     the mic-button handler suspended at `await sessionStartPromise`
    #     forever, isMicStarting true and the button stuck.
    source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")

    capture_line = next(
        l for l in source.splitlines() if "var _ackedResolver =" in l
    )
    assert "S.sessionStartedResolver" in capture_line, (
        "the ack must capture the pending start it belongs to"
    )
    # And only when the ack is answering THIS window's request: a same-mode ack
    # fanned out for somebody else's start must not settle our promise.
    assert "_ackAnswersThisWindow" in capture_line
    capture = capture_line.strip()

    # The capture has to happen at ack time, i.e. before the deferred callback
    # is scheduled -- capturing inside it would read the same shared slot again
    # and pin nothing.
    deferred_end = source.index("}, 500);")
    assert source.index(capture) < deferred_end

    block = source[source.index(capture):deferred_end]
    assert "S.sessionStartedResolver === _ackedResolver" in block, (
        "the shared slot must only be released for the start this ack matched"
    )

    settle = "_ackedResolver(response.input_mode);"
    assert settle in block, "the acknowledged promise must be settled"

    # Structural, not textual: the slot clearing sits INSIDE the identity
    # branch and the settle sits OUTSIDE it, so compare their nesting depth.
    lines = block.splitlines()
    clear_line = next(l for l in lines if "S._pendingSessionStartMode = null;" in l)
    settle_line = next(l for l in lines if settle in l)
    indent = lambda l: len(l) - len(l.lstrip())
    assert indent(clear_line) > indent(settle_line), (
        "clearing the shared slot must be gated on identity while settling the "
        "acknowledged promise must not be"
    )
    assert block.index("S._pendingSessionStartMode = null;") < block.index(settle), (
        "release the slot before settling, so the awaiter never observes a slot "
        "that still points at an already-settled start"
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
