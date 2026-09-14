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

def test_voice_lifecycle_status_is_validated_and_exposed_to_ui():
    source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")

    assert "statusCode === 'ASR_LIFECYCLE_STATE'" in source
    assert "voiceInputLifecycleState" in source
    assert "voice-input-lifecycle-changed" in source
    assert "data-voice-input-state" in source

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

def test_core_capability_refresh_failures_fail_open_and_coalesce_requests_harness():
    source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")
    start = source.index("function publishCoreApiCapability(provider, capability)")
    end = source.index("// Prime the capability once", start)
    refresh_source = source[start:end]
    harness = (
        """
        const S = {
          coreApiProvider: 'free',
          coreApiSupportsIndependentAsr: false,
        };
        let _coreApiCapabilityRefreshPromise = null;
        let _coreApiCapabilityRequestGeneration = 0;
        const events = [];
        class CustomEvent {
          constructor(type, options) {
            this.type = type;
            this.detail = options && options.detail;
          }
        }
        const window = {
          dispatchEvent(event) { events.push(event); },
          fetch: null,
        };
        """
        + refresh_source
        + """
        function response(data) {
          return { ok: true, json: async () => data };
        }
        function assert(condition, message) {
          if (!condition) throw new Error(message);
        }
        async function main() {
          window.fetch = async () => response({ success: false, coreApi: 'qwen' });
          await refreshCoreApiCapability({ force: true });
          assert(S.coreApiSupportsIndependentAsr === null, 'success:false must fail open');
          assert(S.coreApiProvider === '', 'failed response provider must become unknown');

          S.coreApiProvider = 'free';
          S.coreApiSupportsIndependentAsr = false;
          window.fetch = async () => response({ success: true, coreApi: 'qwen' });
          await refreshCoreApiCapability({ force: true });
          assert(S.coreApiSupportsIndependentAsr === null, 'legacy response must fail open');
          assert(S.coreApiProvider === 'qwen', 'usable provider context should be retained');

          const validCapability = {
            success: true,
            coreApi: 'free',
            effectiveCoreApi: 'qwen',
            supportsIndependentAsr: true,
          };
          let fetchCalls = 0;
          let resolveShared;
          window.fetch = () => {
            fetchCalls += 1;
            return new Promise((resolve) => { resolveShared = resolve; });
          };
          const firstRequest = refreshCoreApiCapability({ force: true });
          const joinedForceRequest = refreshCoreApiCapability({ force: true });
          const joinedDefaultRequest = refreshCoreApiCapability();
          assert(firstRequest === joinedForceRequest, 'force callers must share the in-flight request');
          assert(firstRequest === joinedDefaultRequest, 'all callers must share the in-flight request');
          assert(fetchCalls === 1, 'coalesced callers must issue one fetch');
          resolveShared(response(validCapability));
          await firstRequest;
          assert(S.coreApiSupportsIndependentAsr === true, 'shared success must publish capability');
          assert(S.coreApiProvider === 'qwen', 'effective provider must win');

          let resolveNext;
          window.fetch = () => {
            fetchCalls += 1;
            return new Promise((resolve) => { resolveNext = resolve; });
          };
          const nextRequest = refreshCoreApiCapability({ force: true });
          assert(nextRequest !== firstRequest, 'force must bypass completed cache data');
          assert(fetchCalls === 2, 'force after settlement must issue a fresh fetch');
          resolveNext(response(validCapability));
          await nextRequest;
          assert(
            events.length === 3
              && events.every((event) => event.type === 'neko:core-api-capability-changed'),
            'capability changes should notify the shared UI exactly once per change'
          );
          console.log('ok');
        }
        main().catch((error) => {
          console.error(error);
          process.exitCode = 1;
        });
        """
    )
    result = _run_settings_node_harness(harness)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"

def test_settings_hydration_marked_on_server_merge_and_user_change():
    # S.settingsHydrated must flip true on authoritative settings evidence:
    # events, and never merely at boot:
    #   (1) the conversation-settings GET succeeded (server values merged);
    #   (2) the user explicitly changed a setting — the independent-ASR toggle
    #       handler (app-audio-capture.js) runs saveSettings({skipServerSync})
    #       + syncSettingsToServer({ userInitiated: true }), so the synchronous
    #       marker inside syncSettingsToServer covers it even when the POST
    #       later fails (a user action is authoritative even pre-hydration);
    #   (3) a cross-window independent-ASR flip arrived via the 'storage'
    #       listener — the originating window's user action, pinned by
    #       test_cross_window_asr_flip_marks_hydration_and_asr_dirty.
    #   (4) a durable, explicit optimization decision survived a reload while
    #       its server synchronization is still pending.
    settings_source = APP_SETTINGS_PATH.read_text(encoding="utf-8")
    capture_source = APP_AUDIO_CAPTURE_PATH.read_text(encoding="utf-8")

    # (1) Server merge marks hydration only after the null-guard, i.e. only
    # when the GET actually returned a usable result.
    merge_block = settings_source.split(
        "loadSettingsFromServer().then(serverResult => {",
        1,
    )[1].split(".finally(", 1)[0]
    guard_index = merge_block.index("if (!serverResult) return;")
    hydrate_index = merge_block.index("S.settingsHydrated = true;")
    assert guard_index < hydrate_index, (
        "hydration must only be marked after the serverResult null-guard"
    )

    # (2) syncSettingsToServer marks hydration synchronously, before any
    # await, so a failed POST still leaves the user's choice authoritative
    # and the start_session handshake keeps carrying it — but ONLY for
    # userInitiated callers. The periodic timer passes no options and must
    # never mark hydration (pinned by
    # test_periodic_sync_skips_post_and_never_marks_hydration_while_unhydrated).
    sync_fn = settings_source.split(
        "async function syncSettingsToServer(options)", 1
    )[1].split(
        "function startPeriodicSync()",
        1,
    )[0]
    assert sync_fn.count("S.settingsHydrated = true;") == 1
    user_initiated_gate = _block_after(sync_fn, "if (userInitiated) {")
    assert "S.settingsHydrated = true;" in user_initiated_gate, (
        "the hydration mark must sit inside the userInitiated gate"
    )
    assert sync_fn.index("S.settingsHydrated = true;") < sync_fn.index(
        "await _fetchConversationSettingsJsonWithTimeout("
    )

    # The independent-ASR toggle handler reaches that marker via
    # syncSettingsToServer({ userInitiated: true }); the saveSettings call it
    # makes skips the internal server sync, so the direct call is the seam.
    toggle_handler = capture_source.split(
        "var asrToggle = createVoiceSettingToggle(", 1
    )[1].split("asrRow.appendChild(asrCopy);", 1)[0]
    persist_block = capture_source.split(
        "function persistVoiceSettingChange() {", 1
    )[1].split("function markVoiceSettingsPending", 1)[0]
    assert "S.independentAsrEnabled = enabled;" in toggle_handler
    assert "persistVoiceSettingChange();" in toggle_handler
    assert "window.appSettings.syncSettingsToServer({ userInitiated: true })" in persist_block

    # Boot must NOT mark hydration: the first-launch initialization save goes
    # through saveSettings({ skipServerSync: true }) which bypasses
    # syncSettingsToServer, keeping the default-false value non-authoritative
    # until the GET resolves or the user acts.
    first_launch_block = settings_source.split(
        "console.log('未找到保存的设置，使用默认值');",
        1,
    )[1].split("} catch (error) {", 1)[0]
    assert "saveSettings({ skipServerSync: true });" in first_launch_block
    assert "S.settingsHydrated" not in first_launch_block
    # The only synchronous load-time hydration is guarded by a durable pending
    # optimization decision; ordinary boot defaults still cannot gain authority.
    sync_load_body = settings_source.split("function loadSettings()", 1)[1].split(
        "loadSettingsFromServer().then(serverResult => {",
        1,
    )[0]
    assert sync_load_body.count("S.settingsHydrated = true;") == 1
    assert "bootMeta.optimizationDecisionPendingSync" in sync_load_body
    assert (
        sync_load_body.index("bootMeta.optimizationDecisionPendingSync")
        < sync_load_body.index("S.settingsHydrated = true;")
    )

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

def test_cross_window_settings_posts_use_cas_and_persist_asr_decision_order():
    settings_source = APP_SETTINGS_PATH.read_text(encoding="utf-8")
    sync_fn = _block_after(
        settings_source, "async function syncSettingsToServer(options) {"
    )

    assert "let _conversationSettingsEtag = null;" in settings_source
    assert "headers['If-Match'] = _conversationSettingsEtag;" in sync_fn
    assert "response.status === 412" in sync_fn
    assert "const preservedKeys = new Set(_pendingSettingsKeys);" in sync_fn
    assert "_conversationSettingsEtagKeyMutationVersions" in sync_fn
    assert "_crossWindowSettingsNewerThanEtag(" in sync_fn
    assert "etagKeyMutationVersionsAtSend" in sync_fn
    assert "_markEtagConfirmedSharedSettings(" in sync_fn
    assert "_settingsChangedSince(settings, mutationVersionAtSend).forEach" in sync_fn
    assert "_mergeConversationSettingsSnapshot(data, preservedKeys);" in sync_fn
    assert "_CONVERSATION_SETTINGS_MAX_ATTEMPTS" in sync_fn
    assert "headers['X-Conversation-Settings-ASR-Decision']" in sync_fn
    assert "JSON.stringify(requestDecision)" in sync_fn
    mark_signature = re.search(
        r"function _markEtagConfirmedSharedSettings\([^)]*\)\s*\{",
        settings_source,
    )
    assert mark_signature is not None
    mark_confirmed = _block_after(settings_source, mark_signature.group(0))
    assert "payloadWasFull" not in mark_confirmed
    assert "settingsAtSend[key] !== serverSettings[key]" in mark_confirmed

    # Both the state snapshot and the ASR token are rebuilt inside the retry
    # loop. An older window that loses the server decision comparison must not
    # resend its stale pre-conflict body.
    retry_loop = sync_fn.split(
        "for (let attempt = 0; attempt < _CONVERSATION_SETTINGS_MAX_ATTEMPTS;",
        1,
    )[1]
    assert retry_loop.index("const settings = getConversationSettings();") < retry_loop.index(
        "await _fetchConversationSettingsJsonWithTimeout("
    )
    assert retry_loop.index("const requestDecision = (") < retry_loop.index(
        "await _fetchConversationSettingsJsonWithTimeout("
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

def test_settings_cas_conflict_rebuilds_body_from_winning_asr_decision_harness():
    harness = textwrap.dedent(
        """
        const fs = require('node:fs');
        const vm = require('node:vm');
        const source = fs.readFileSync(__APP_SETTINGS_PATH__, 'utf8');

        function assert(cond, msg) {
          if (!cond) throw new Error('ASSERT: ' + msg);
        }
        function response(ok, status, etag, data) {
          return {
            ok,
            status,
            headers: {
              get(name) { return name.toLowerCase() === 'etag' ? etag : null; },
            },
            json: async () => data,
          };
        }
        const tick = () => new Promise((resolve) => setImmediate(resolve));

        function makeContext(
          bootFails,
          bootData,
          initialProjectSettings,
          failSharedStorageWrites
        ) {
          let storageListener = null;
          const runtime = {
            stoppedSpeech: 0,
            stoppedScreening: 0,
            stoppedTracks: 0,
            scheduled: 0,
          };
          const store = new Map([
            ['project_neko_settings', JSON.stringify(initialProjectSettings || {
              independentAsrEnabled: false,
              proactiveVisionEnabled: true,
              slopFilterEnabled: false,
              mergeMessagesEnabled: false,
              mouseTrackingEnabled: false,
            })],
            ['neko_noise_reduction', '0'],
          ]);
          const postCalls = [];
          const sandbox = {
            console: { log() {}, warn() {}, error() {} },
            setInterval() { return 0; },
            clearInterval() {},
            setTimeout(fn, ms) {
              const timer = setTimeout(fn, ms);
              if (timer && typeof timer.unref === 'function') timer.unref();
              return timer;
            },
            clearTimeout,
            localStorage: {
              getItem(key) { return store.has(key) ? store.get(key) : null; },
              setItem(key, value) {
                if (failSharedStorageWrites && key === 'project_neko_settings') {
                  throw new Error('shared localStorage unavailable');
                }
                store.set(key, String(value));
              },
              removeItem(key) { store.delete(key); },
            },
            document: { getElementById() { return null; } },
            fetch(url, opts) {
              if (opts && opts.method === 'POST') {
                return new Promise((resolve) => {
                  postCalls.push({ url, opts, resolve });
                });
              }
              if (bootFails) return Promise.resolve({ ok: false, status: 500 });
              return Promise.resolve(response(
                true,
                200,
                '"conversation-settings-0"',
                bootData || {
                  success: true,
                  settings: { independentAsrEnabled: false },
                  telemetryBranch: null,
                  decisions: {},
                }
              ));
            },
          };
          sandbox.window = {
            appState: {
              independentAsrEnabled: false,
              proactiveVisionEnabled: true,
              slopFilterEnabled: false,
              mergeMessagesEnabled: false,
              focusModeEnabled: false,
              settingsHydrated: false,
              screenCaptureStream: {
                getTracks() {
                  return [{ stop() { runtime.stoppedTracks += 1; } }];
                },
              },
            },
            appConst: {},
            appUtils: { mapRenderQualityToFollowPerf() { return 'medium'; } },
            proactiveVisionEnabled: true,
            slopFilterEnabled: false,
            focusModeEnabled: false,
            stopProactiveVisionDuringSpeech() { runtime.stoppedSpeech += 1; },
            stopScreening() { runtime.stoppedScreening += 1; },
            scheduleProactiveChat() { runtime.scheduled += 1; },
            addEventListener(type, listener) {
              if (type === 'storage') storageListener = listener;
            },
            removeEventListener() {},
          };
          vm.createContext(sandbox);
          vm.runInContext(source, sandbox);
          return {
            S: sandbox.window.appState,
            win: sandbox.window,
            mod: sandbox.window.appSettings,
            postCalls,
            store,
            runtime,
            fireStorage(newValue) {
              storageListener({ key: 'project_neko_settings', newValue });
            },
          };
        }

        async function runBootMetadataScenario() {
          const tuple = {
            // Valid at an authority whose clock is just over one second ahead,
            // but beyond this browser's independently measured +1 year bound.
            writeId: Date.now() + (365 * 24 * 60 * 60 * 1000) + 1000,
            writerId: 'server-ahead',
            value: false,
          };
          const tupleOnly = makeContext(false, {
            success: true,
            settings: { independentAsrEnabled: false },
            revision: 1,
            telemetryBranch: null,
            decisions: { independentAsrEnabled: tuple },
          });
          await tick();
          await tick();
          const persistedTuple = JSON.parse(
            tupleOnly.store.get('project_neko_settings')
          )._sharedWriteMeta.asrDecision;
          const tupleEnvelope = JSON.parse(
            tupleOnly.store.get('project_neko_settings')
          )._sharedWriteMeta.writeId;
          assert(
            JSON.stringify(persistedTuple) === JSON.stringify(tuple),
            'a newer same-value server tuple must persist for offline write-id flooring'
          );
          assert(
            tupleEnvelope < tuple.writeId,
            'a server ASR floor must not inflate the localStorage envelope id'
          );

          const resetPriorDecision = {
            writeId: Date.now() + 1000,
            writerId: 'server-before-reset',
            value: true,
          };
          assert(
            resetPriorDecision.value !== false,
            'the pre-reset ASR decision must differ from the disabled reset default'
          );
          const reset = makeContext(false, {
            success: true,
            settings: {},
            revision: 10,
            reset: true,
            telemetryBranch: null,
            decisions: { independentAsrEnabled: resetPriorDecision },
          });
          await tick();
          await tick();
          const resetVisionDefault = reset.mod._isUserRegionChina();
          assert(
            reset.S.slopFilterEnabled === true
              && reset.S.proactiveVisionEnabled === resetVisionDefault
              && reset.S.independentAsrEnabled === false
              && reset.S.voiceInputResourceOptimizationEnabled === true,
            'an empty authoritative restore must reset stale local values to defaults: '
              + JSON.stringify({
                slop: reset.S.slopFilterEnabled,
                vision: reset.S.proactiveVisionEnabled,
                visionDefault: resetVisionDefault,
                asr: reset.S.independentAsrEnabled,
                optimization: reset.S.voiceInputResourceOptimizationEnabled,
              })
          );
          assert(reset.postCalls.length === 1, 'the reset defaults must be written back once');
          const resetWritebackDecision = JSON.parse(
            reset.postCalls[0].opts.headers[
              'X-Conversation-Settings-ASR-Decision'
            ]
          );
          assert(
            resetWritebackDecision.value === false
              && resetWritebackDecision.writeId
                > resetPriorDecision.writeId,
            'a reset writeback must rebase the stale tuple onto the reset default'
          );
          assert(
            reset.postCalls[0].opts.headers['X-Conversation-Settings-Full-Snapshot'] === '1',
            'a reset writeback must declare that it can clear the tombstone'
          );
          const resetBody = JSON.parse(reset.postCalls[0].opts.body);
          assert(
            resetBody.slopFilterEnabled === true
              && resetBody.proactiveVisionEnabled === resetVisionDefault
              && resetBody.independentAsrEnabled === false
              && resetBody.voiceInputResourceOptimizationEnabled === true,
            'the reset writeback must not repopulate the server with stale localStorage'
          );
          reset.postCalls[0].resolve(response(
            true,
            200,
            '"conversation-settings-11"',
            {
              success: true,
              settings: resetBody,
              revision: 11,
              reset: false,
              decisions: {
                independentAsrEnabled: resetWritebackDecision,
              },
            }
          ));
          await tick();
          const resetPersisted = JSON.parse(
            reset.store.get('project_neko_settings')
          );
          assert(
            resetPersisted._sharedWriteMeta.asrDecision.value === false
              && resetPersisted._sharedWriteMeta.serverRevision === 11,
            'a full-write success must adopt and rebroadcast the generated server ASR tuple'
          );

          const resetRace = makeContext(false, {
            success: true,
            settings: {},
            revision: 10,
            reset: true,
            telemetryBranch: null,
            decisions: { independentAsrEnabled: resetPriorDecision },
          });
          await tick();
          await tick();
          assert(
            resetRace.postCalls.length === 1,
            'the reset-race writeback must start'
          );
          const resetRaceBaselineDecision = JSON.parse(
            resetRace.postCalls[0].opts.headers[
              'X-Conversation-Settings-ASR-Decision'
            ]
          );
          resetRace.S.independentAsrEnabled = true;
          resetRace.mod.saveSettings({
            skipServerSync: true,
            explicitSharedKeys: ['independentAsrEnabled'],
          });
          const resetToggleSync = resetRace.mod.syncSettingsToServer({
            userInitiated: true,
          });
          await tick();
          assert(
            resetRace.postCalls.length === 1,
            'the user toggle must queue behind the reset writeback'
          );
          const resetToggleDecision = JSON.parse(
            resetRace.store.get('project_neko_settings')
          )._sharedWriteMeta.asrDecision;
          assert(
            resetToggleDecision.value === true
              && resetToggleDecision.writeId
                > resetRaceBaselineDecision.writeId,
            'a toggle during reset must mint above the rebased reset decision'
          );
          resetRace.postCalls[0].resolve(response(
            true,
            200,
            '"conversation-settings-11"',
            {
              success: true,
              settings: {
                independentAsrEnabled: false,
                slopFilterEnabled: true,
              },
              revision: 11,
              reset: false,
              decisions: {
                independentAsrEnabled: resetRaceBaselineDecision,
              },
            }
          ));
          await tick();
          await tick();
          assert(
            resetRace.S.independentAsrEnabled === true
              && resetRace.postCalls.length === 2,
            'the older reset response must not overwrite the queued toggle'
          );
          const resetToggleBody = JSON.parse(resetRace.postCalls[1].opts.body);
          const resetToggleHeader = JSON.parse(
            resetRace.postCalls[1].opts.headers[
              'X-Conversation-Settings-ASR-Decision'
            ]
          );
          assert(
            resetToggleBody.independentAsrEnabled === true
              && JSON.stringify(resetToggleHeader)
                === JSON.stringify(resetToggleDecision),
            'the queued sync must persist the newer toggle tuple'
          );
          resetRace.postCalls[1].resolve(response(
            true,
            200,
            '"conversation-settings-12"',
            {
              success: true,
              settings: resetToggleBody,
              revision: 12,
              reset: false,
              decisions: {
                independentAsrEnabled: resetToggleDecision,
              },
            }
          ));
          await resetToggleSync;

          const noiseMerge = makeContext(false, {
            success: true,
            settings: {
              independentAsrEnabled: false,
              noiseReductionEnabled: true,
            },
            revision: 1,
            telemetryBranch: null,
            decisions: {},
          });
          await tick();
          await tick();
          assert(
            noiseMerge.store.get('neko_noise_reduction') === '1',
            'a boot merge must synchronize the legacy noise cache'
          );

          const serverBroadcaster = makeContext(false, {
            success: true,
            settings: {
              independentAsrEnabled: false,
              proactiveVisionEnabled: false,
            },
            revision: 2,
            telemetryBranch: null,
            decisions: {},
          });
          await tick();
          await tick();
          const serverSnapshot = JSON.parse(
            serverBroadcaster.store.get('project_neko_settings')
          );
          assert(
            serverSnapshot._sharedWriteMeta.serverRevision === 2
              && serverSnapshot._sharedWriteMeta.serverAuthoritativeKeys
                .includes('proactiveVisionEnabled'),
            'a server winner must carry its real revision and authoritative fields'
          );
          serverBroadcaster.fireStorage(JSON.stringify({
            proactiveVisionEnabled: true,
            _sharedWriteMeta: {
              writeId: 1,
              writerId: 'window-old',
              changedKeys: ['proactiveVisionEnabled'],
              hydrated: true,
              asrAuthoritative: true,
              knownKeyWrites: {
                proactiveVisionEnabled: {
                  writeId: 1,
                  writerId: 'window-old',
                },
              },
            },
          }));
          assert(
            serverBroadcaster.S.proactiveVisionEnabled === true,
            'an unconfirmed explicit edit must outrank a server snapshot '
              + 'that did not observe it'
          );
          assert(
            !serverSnapshot._sharedWriteMeta.knownKeyWrites
              .proactiveVisionEnabled
              && serverSnapshot._sharedWriteMeta.serverKeyRevisions
                .proactiveVisionEnabled === 2,
            'the server floor must be serialized without forging a source token'
          );
          const reloadedServerWinner = makeContext(
            true,
            null,
            serverSnapshot
          );
          await tick();
          await tick();
          reloadedServerWinner.fireStorage(JSON.stringify({
            proactiveVisionEnabled: true,
            _sharedWriteMeta: {
              writeId: 1,
              writerId: 'window-old-after-reload',
              changedKeys: ['proactiveVisionEnabled'],
              hydrated: true,
              asrAuthoritative: true,
              knownKeyWrites: {
                proactiveVisionEnabled: {
                  writeId: 1,
                  writerId: 'window-old-after-reload',
                },
              },
            },
          }));
          assert(
            reloadedServerWinner.S.proactiveVisionEnabled === true,
            'reload must not let the persisted server floor suppress '
              + 'an unconfirmed explicit edit'
          );

          const receiver = makeContext(true);
          await tick();
          await tick();
          receiver.fireStorage(JSON.stringify({
            proactiveVisionEnabled: true,
            _sharedWriteMeta: {
              writeId: 1,
              writerId: 'window-old',
              changedKeys: ['proactiveVisionEnabled'],
              hydrated: true,
              asrAuthoritative: true,
              knownKeyWrites: {
                proactiveVisionEnabled: {
                  writeId: 1,
                  writerId: 'window-old',
                  confirmedRevision: 1,
                },
              },
            },
          }));
          receiver.fireStorage(JSON.stringify(serverSnapshot));
          assert(
            receiver.S.proactiveVisionEnabled === false,
            'a newer authoritative server winner must outrank an older local token'
          );
          assert(
            receiver.runtime.stoppedSpeech === 1
              && receiver.runtime.stoppedScreening === 1
              && receiver.runtime.stoppedTracks === 1,
            'the authoritative privacy disable must stop active vision runtime'
          );
          receiver.fireStorage(JSON.stringify({
            proactiveVisionEnabled: true,
            _sharedWriteMeta: {
              writeId: 1,
              writerId: 'window-old',
              changedKeys: ['proactiveVisionEnabled'],
              hydrated: true,
              asrAuthoritative: true,
              knownKeyWrites: {
                proactiveVisionEnabled: {
                  writeId: 1,
                  writerId: 'window-old',
                  confirmedRevision: 1,
                },
              },
            },
          }));
          assert(
            receiver.S.proactiveVisionEnabled === false,
            'an accepted server winner must retain a floor against delayed old events'
          );
          const newerVisionWriteId =
            serverSnapshot._sharedWriteMeta.writeId + 1;
          receiver.fireStorage(JSON.stringify({
            proactiveVisionEnabled: true,
            _sharedWriteMeta: {
              writeId: newerVisionWriteId,
              writerId: 'window-new',
              changedKeys: ['proactiveVisionEnabled'],
              hydrated: true,
              asrAuthoritative: true,
              knownKeyWrites: {
                proactiveVisionEnabled: {
                  writeId: newerVisionWriteId,
                  writerId: 'window-new',
                },
              },
            },
          }));
          assert(
            receiver.S.proactiveVisionEnabled === true,
            'a genuinely newer explicit event must still supersede the server floor'
          );

          const subtitleReceiver = makeContext(true);
          await tick();
          await tick();
          subtitleReceiver.fireStorage(JSON.stringify({
            subtitleEnabled: false,
            userLanguage: null,
            _sharedWriteMeta: {
              writeId: 700,
              writerId: 'window-subtitle-editor',
              changedKeys: ['subtitleEnabled', 'userLanguage'],
              hydrated: true,
              asrAuthoritative: false,
              knownKeyWrites: {
                subtitleEnabled: {
                  writeId: 700,
                  writerId: 'window-subtitle-editor',
                },
                userLanguage: {
                  writeId: 700,
                  writerId: 'window-subtitle-editor',
                },
              },
            },
          }));
          subtitleReceiver.fireStorage(JSON.stringify({
            subtitleEnabled: true,
            userLanguage: 'ja',
            _sharedWriteMeta: {
              writeId: 800,
              writerId: 'window-stale-server-reader',
              changedKeys: [],
              hydrated: true,
              asrAuthoritative: false,
              serverRevision: 1,
              serverAuthoritativeKeys: [
                'subtitleEnabled',
                'userLanguage',
              ],
              knownKeyWrites: {},
            },
          }));
          assert(
            subtitleReceiver.S.subtitleEnabled === false
              && subtitleReceiver.S.userLanguage === null,
            'same-value subtitle intent must enter per-key ordering and survive '
              + 'a delayed stale server merge'
          );

          const noSharedStorage = makeContext(
            false,
            {
              success: true,
              settings: {},
              revision: 0,
              telemetryBranch: null,
              decisions: {},
            },
            null,
            true
          );
          await tick();
          await tick();
          noSharedStorage.S.noiseReductionEnabled = false;
          noSharedStorage.mod.saveSettings();
          await tick();
          await tick();
          assert(
            noSharedStorage.postCalls.length === 1
              && JSON.parse(
                noSharedStorage.postCalls[0].opts.body
              ).noiseReductionEnabled === false,
            'a failed shared localStorage write must not suppress the noise CAS POST'
          );
          noSharedStorage.postCalls[0].resolve(response(
            true,
            200,
            '"conversation-settings-1"',
            {
              success: true,
              settings: { noiseReductionEnabled: false },
              revision: 1,
              decisions: {},
            }
          ));

          const staleReceiver = makeContext(false, {
            success: true,
            settings: {
              independentAsrEnabled: false,
              focusModeEnabled: false,
            },
            revision: 2,
            telemetryBranch: null,
            decisions: {},
          });
          await tick();
          await tick();
          staleReceiver.fireStorage(JSON.stringify({
            focusModeEnabled: true,
            _sharedWriteMeta: {
              writeId: 50,
              writerId: 'window-editor',
              changedKeys: ['focusModeEnabled'],
              hydrated: true,
              asrAuthoritative: true,
              knownKeyWrites: {
                focusModeEnabled: {
                  writeId: 50,
                  writerId: 'window-editor',
                },
              },
            },
          }));
          const staleServerSnapshot = {
            focusModeEnabled: false,
            _sharedWriteMeta: {
              writeId: 500,
              writerId: 'window-stale-server-reader',
              changedKeys: [],
              hydrated: true,
              asrAuthoritative: true,
              serverRevision: 2,
              serverAuthoritativeKeys: ['focusModeEnabled'],
              knownKeyWrites: {},
            },
          };
          staleReceiver.fireStorage(JSON.stringify(staleServerSnapshot));
          assert(
            staleReceiver.S.focusModeEnabled === true,
            'a same-revision server snapshot must not launder its envelope '
              + 'over an unconfirmed explicit edit'
          );

          const equalRevision = makeContext(false, {
            success: true,
            settings: {
              independentAsrEnabled: false,
              focusModeEnabled: false,
            },
            revision: 2,
            telemetryBranch: null,
            decisions: {},
          });
          await tick();
          await tick();
          equalRevision.fireStorage(JSON.stringify({
            focusModeEnabled: true,
            _sharedWriteMeta: {
              writeId: 600,
              writerId: 'window-editor',
              changedKeys: ['focusModeEnabled'],
              hydrated: true,
              asrAuthoritative: true,
              knownKeyWrites: {
                focusModeEnabled: {
                  writeId: 600,
                  writerId: 'window-editor',
                },
              },
            },
          }));
          equalRevision.fireStorage(JSON.stringify({
            focusModeEnabled: true,
            _sharedWriteMeta: {
              writeId: 601,
              writerId: 'window-server-reader',
              changedKeys: [],
              hydrated: true,
              asrAuthoritative: true,
              serverRevision: 2,
              serverAuthoritativeKeys: ['focusModeEnabled'],
              knownKeyWrites: {
                focusModeEnabled: {
                  writeId: 600,
                  writerId: 'window-editor',
                  confirmedRevision: 2,
                },
              },
            },
          }));
          const equalSync = equalRevision.mod.syncSettingsToServer();
          await tick();
          assert(equalRevision.postCalls.length === 1, 'the confirmed view must POST');
          equalRevision.postCalls[0].resolve(response(
            false,
            412,
            '"conversation-settings-3"',
            {
              success: false,
              settings: {
                independentAsrEnabled: false,
                focusModeEnabled: false,
              },
              revision: 3,
              decisions: {},
            }
          ));
          await tick();
          await tick();
          assert(equalRevision.postCalls.length === 2, 'the newer conflict must retry');
          const equalRetryBody = JSON.parse(equalRevision.postCalls[1].opts.body);
          assert(
            equalRetryBody.focusModeEnabled === false,
            'an equal-revision confirmation must stop the older local value '
              + 'being preserved over revision 3'
          );
          equalRevision.postCalls[1].resolve(response(
            true,
            200,
            '"conversation-settings-4"',
            {
              success: true,
              settings: equalRetryBody,
              revision: 4,
              decisions: {},
            }
          ));
          await equalSync;
        }

        async function runScenario(serverDecisionIsNewer) {
          const ctx = makeContext();
          await tick();
          await tick();
          assert(ctx.S.settingsHydrated === true, 'boot GET must hydrate settings');

          ctx.S.independentAsrEnabled = true;
          ctx.mod.saveSettings({ skipServerSync: true });
          const syncPromise = ctx.mod.syncSettingsToServer({ userInitiated: true });
          await tick();
          assert(ctx.postCalls.length === 1, 'first CAS POST must be issued');
          const first = ctx.postCalls[0];
          const firstBody = JSON.parse(first.opts.body);
          const localDecision = JSON.parse(
            first.opts.headers['X-Conversation-Settings-ASR-Decision']
          );
          assert(
            first.opts.headers['If-Match'] === '"conversation-settings-0"',
            'boot ETag must guard the first POST'
          );
          assert(localDecision.value === true, 'first request carries local ASR intent');

          const serverDecision = {
            writeId: serverDecisionIsNewer
              ? localDecision.writeId + 1
              : Math.max(0, localDecision.writeId - 1),
            writerId: serverDecisionIsNewer ? 'window-z' : 'window-a',
            value: false,
          };
          first.resolve(response(
            false,
            412,
            '"conversation-settings-1"',
            {
              success: false,
              settings: { independentAsrEnabled: false, slopFilterEnabled: true },
              revision: 1,
              decisions: { independentAsrEnabled: serverDecision },
            }
          ));
          await tick();
          await tick();
          assert(ctx.S.slopFilterEnabled === true, 'conflict merge must adopt the server field');
          assert(
            ctx.win.slopFilterEnabled === true,
            'conflict merge must synchronize the window mirror'
          );
          assert(ctx.postCalls.length === 2, 'a CAS conflict must retry once');
          const retry = ctx.postCalls[1];
          const retryBody = JSON.parse(retry.opts.body);
          assert(
            retry.opts.headers['If-Match'] === '"conversation-settings-1"',
            'retry must use the conflict response ETag'
          );
          assert(
            retryBody.independentAsrEnabled === !serverDecisionIsNewer,
            'retry body must use the winning decision value'
          );
          assert(
            retryBody.slopFilterEnabled === true,
            'retry must not roll the conflict-merged value back from a stale window mirror'
          );
          const retryDecision =
            JSON.parse(retry.opts.headers['X-Conversation-Settings-ASR-Decision']);
          assert(
            retryDecision.writeId === (
              serverDecisionIsNewer ? serverDecision.writeId : localDecision.writeId
            ),
            'retry must carry the winning decision token'
          );
          retry.resolve(response(
            true,
            200,
            '"conversation-settings-2"',
            {
              success: true,
              settings: { independentAsrEnabled: retryBody.independentAsrEnabled },
              revision: 2,
              decisions: { independentAsrEnabled: retryDecision },
            }
          ));
          await syncPromise;
          if (serverDecisionIsNewer) {
            ctx.S.independentAsrEnabled = true;
            ctx.mod.saveSettings({ skipServerSync: true });
            const nextSharedSnapshot = JSON.parse(
              ctx.store.get('project_neko_settings')
            );
            const nextDecision = nextSharedSnapshot._sharedWriteMeta.asrDecision;
            assert(
              nextDecision.writeId > serverDecision.writeId,
              'the next explicit local toggle must supersede an adopted server decision'
            );
            assert(nextDecision.value === true, 'the superseding tuple carries the new choice');
          }
        }

        async function runAcknowledgedDirtyScenario() {
          const ctx = makeContext();
          await tick();
          await tick();

          ctx.win.slopFilterEnabled = true;
          ctx.mod.saveSettings();
          assert(ctx.S.slopFilterEnabled === true, 'the first edit updates shared state immediately');
          await tick();
          assert(ctx.postCalls.length === 1, 'the first user edit must POST');
          const acknowledged = JSON.parse(ctx.postCalls[0].opts.body);
          ctx.postCalls[0].resolve(response(
            true,
            200,
            '"conversation-settings-1"',
            {
              success: true,
              settings: acknowledged,
              revision: 1,
              decisions: {},
            }
          ));
          await tick();
          await tick();

          const acknowledgedLocal = JSON.parse(
            ctx.store.get('project_neko_settings')
          );
          const acknowledgedSlopToken =
            acknowledgedLocal._sharedWriteMeta.knownKeyWrites.slopFilterEnabled;
          ctx.fireStorage(JSON.stringify({
            slopFilterEnabled: false,
            proactiveMusicEnabled: false,
            _sharedWriteMeta: {
              writeId: acknowledgedLocal._sharedWriteMeta.writeId + 20,
              writerId: 'window-unrelated-editor',
              changedKeys: ['proactiveMusicEnabled'],
              hydrated: true,
              asrAuthoritative: true,
              knownKeyWrites: {
                slopFilterEnabled: {
                  writeId: Math.max(0, acknowledgedSlopToken.writeId - 1),
                  writerId: 'window-stale',
                },
                proactiveMusicEnabled: {
                  writeId: acknowledgedLocal._sharedWriteMeta.writeId + 20,
                  writerId: 'window-unrelated-editor',
                },
              },
            },
          }));
          assert(
            ctx.S.slopFilterEnabled === true
              && ctx.S.proactiveMusicEnabled === false,
            'an unrelated explicit snapshot must filter a stale incidental field'
          );
          ctx.fireStorage(JSON.stringify({
            slopFilterEnabled: false,
            _sharedWriteMeta: {
              writeId: Math.max(
                0,
                acknowledgedLocal._sharedWriteMeta.writeId - 1
              ),
              writerId: 'window-delayed',
              changedKeys: ['slopFilterEnabled'],
              hydrated: true,
              asrAuthoritative: true,
            },
          }));
          assert(
            ctx.S.slopFilterEnabled === true,
            'an older explicit event must not roll back an acknowledged edit'
          );
          ctx.fireStorage(JSON.stringify({
            slopFilterEnabled: false,
            _sharedWriteMeta: {
              writeId: acknowledgedLocal._sharedWriteMeta.writeId,
              writerId: 'zzzzzzzzzzzzzzzz',
              changedKeys: [],
              hydrated: true,
              asrAuthoritative: true,
            },
          }));
          assert(
            ctx.S.slopFilterEnabled === true,
            'a delayed same-id server merge must not roll back an acknowledged local edit'
          );

          // A newer explicit value may only have been received from another
          // window, so it advances the applied floor without minting a local
          // write id. A delayed merge must respect that floor too.
          const externalWriteId =
            acknowledgedLocal._sharedWriteMeta.writeId + 10;
          ctx.fireStorage(JSON.stringify({
            mergeMessagesEnabled: true,
            _sharedWriteMeta: {
              writeId: externalWriteId,
              writerId: 'window-c',
              changedKeys: ['mergeMessagesEnabled'],
              hydrated: true,
              asrAuthoritative: true,
            },
          }));
          ctx.fireStorage(JSON.stringify({
            mergeMessagesEnabled: false,
            _sharedWriteMeta: {
              writeId: externalWriteId,
              writerId: 'window-d',
              changedKeys: [],
              hydrated: true,
              asrAuthoritative: true,
            },
          }));
          assert(
            ctx.S.mergeMessagesEnabled === true,
            'a delayed merge must not roll back a newer externally applied value'
          );

          // This explicit edit arrives after the boot ETag but before the next
          // CAS request starts. It is already present in that request snapshot,
          // so mutationVersionAtSend alone cannot detect it later; the ETag's
          // cross-window watermark must preserve it across the 412.
          ctx.fireStorage(JSON.stringify({
            avatarReactionBubbleEnabled: true,
            _sharedWriteMeta: {
              writeId: externalWriteId + 1,
              writerId: 'window-c',
              changedKeys: ['avatarReactionBubbleEnabled'],
              hydrated: true,
              asrAuthoritative: true,
            },
          }));
          assert(
            ctx.S.avatarReactionBubbleEnabled === true,
            'the pre-request cross-window edit must be accepted locally'
          );

          // A different local edit races a newer server revision. The earlier
          // slopFilterEnabled=true was already acknowledged and must no longer
          // be protected as pending during the 412 merge.
          ctx.win.focusModeEnabled = true;
          ctx.mod.saveSettings();
          await tick();
          assert(ctx.postCalls.length === 2, 'the unrelated edit must POST');

          // A sibling's unrelated explicit edit still carries a full snapshot.
          // Its incidental copy of this window's pending key must not replace
          // the pending value merely because changedKeys is nonempty.
          ctx.fireStorage(JSON.stringify({
            focusModeEnabled: false,
            mergeMessagesEnabled: true,
            _sharedWriteMeta: {
              writeId: 98,
              writerId: 'window-b',
              changedKeys: ['mergeMessagesEnabled'],
              hydrated: true,
              asrAuthoritative: true,
            },
          }));
          assert(
            ctx.S.focusModeEnabled === true && ctx.S.mergeMessagesEnabled === true,
            'a nonempty explicit broadcast preserves unrelated pending keys'
          );

          // A server-merge broadcast may have been built before this pending
          // edit. It must neither overwrite the local value nor leave its
          // stale full snapshot in shared localStorage.
          ctx.fireStorage(JSON.stringify({
            focusModeEnabled: false,
            _sharedWriteMeta: {
              writeId: 99,
              writerId: 'server-window',
              changedKeys: [],
              hydrated: true,
              asrAuthoritative: true,
            },
          }));
          assert(
            ctx.S.focusModeEnabled === true,
            'a server-merge broadcast must preserve a pending local edit'
          );
          const reasserted = JSON.parse(ctx.store.get('project_neko_settings'));
          assert(
            reasserted.focusModeEnabled === true
              && reasserted._sharedWriteMeta.changedKeys.length === 0,
            'the pending value must be restored without advertising new user intent'
          );

          // Cross-window ABA edits do not enter this window's pending set and
          // leave the final value equal to the request snapshot. The mutation
          // itself must still protect the latest choice from the 412 snapshot.
              ctx.fireStorage(JSON.stringify({
                mergeMessagesEnabled: true,
                _sharedWriteMeta: {
                  writeId: externalWriteId + 2,
              writerId: 'window-b',
              changedKeys: ['mergeMessagesEnabled'],
              hydrated: true,
            },
          }));
              ctx.fireStorage(JSON.stringify({
                mergeMessagesEnabled: false,
                _sharedWriteMeta: {
                  writeId: externalWriteId + 3,
              writerId: 'window-b',
              changedKeys: ['mergeMessagesEnabled'],
              hydrated: true,
            },
          }));
          ctx.postCalls[1].resolve(response(
            false,
            412,
            '"conversation-settings-2"',
            {
              success: false,
              settings: {
                independentAsrEnabled: false,
                proactiveVisionEnabled: false,
                slopFilterEnabled: false,
                focusModeEnabled: false,
                mergeMessagesEnabled: true,
                avatarReactionBubbleEnabled: false,
              },
              revision: 2,
              decisions: {},
            }
          ));
          await tick();
          await tick();
          assert(ctx.postCalls.length === 3, 'the conflict must retry');
          const retryBody = JSON.parse(ctx.postCalls[2].opts.body);
          assert(
            retryBody.slopFilterEnabled === false,
            'the retry must adopt the newer server value for an acknowledged old edit'
          );
          assert(
            retryBody.focusModeEnabled === true,
            'the still-pending local edit must survive the conflict merge'
          );
          assert(
            retryBody.mergeMessagesEnabled === false,
            'a non-pending ABA edit made after send must survive the conflict merge'
          );
          assert(
            retryBody.avatarReactionBubbleEnabled === true,
            'an explicit cross-window edit after the ETag but before send must survive'
          );
          assert(
            retryBody.proactiveVisionEnabled === false,
            'the retry must retain the server privacy winner'
          );
          assert(
            ctx.runtime.stoppedSpeech === 1
              && ctx.runtime.stoppedScreening === 1
              && ctx.runtime.stoppedTracks === 1,
            'the privacy winner must stop every active vision runtime path'
          );
          const reconciledLocal = JSON.parse(ctx.store.get('project_neko_settings'));
          assert(
            reconciledLocal.slopFilterEnabled === false
              && reconciledLocal.focusModeEnabled === true
              && reconciledLocal.proactiveVisionEnabled === false
              && reconciledLocal.mergeMessagesEnabled === false
              && reconciledLocal.avatarReactionBubbleEnabled === true,
            'the conflict winners and pending local edit must persist to shared localStorage'
          );
          assert(
            reconciledLocal.mouseTrackingEnabled === false,
            'server reconciliation must preserve local-only settings'
          );
          assert(
            reconciledLocal._sharedWriteMeta.changedKeys.length === 0,
            'server winners must not be advertised as new user intent'
          );
          ctx.postCalls[2].resolve(response(
            true,
            200,
            '"conversation-settings-3"',
            {
              success: true,
              settings: retryBody,
              revision: 3,
              decisions: {},
            }
          ));
          await tick();
        }

        async function runSuccessfulPartialSnapshotScenario() {
          const ctx = makeContext(true);
          await tick();
          await tick();
          assert(ctx.S.settingsHydrated === false, 'failed boot GET stays unhydrated');

          // A cross-window edit arrives before this partial request starts.
          // It is not this window's pending key and therefore is absent from
          // the dirty-only payload, but the success snapshot must not erase it.
          ctx.fireStorage(JSON.stringify({
            avatarReactionBubbleEnabled: true,
            _sharedWriteMeta: {
              writeId: 101,
              writerId: 'window-b',
              changedKeys: ['avatarReactionBubbleEnabled'],
              hydrated: true,
              asrAuthoritative: true,
            },
          }));

          ctx.win.focusModeEnabled = true;
          ctx.mod.saveSettings();
          await tick();
          assert(ctx.postCalls.length === 1, 'the first pending edit must POST');
          const firstBody = JSON.parse(ctx.postCalls[0].opts.body);
          assert(
            Object.keys(firstBody).length === 1 && firstBody.focusModeEnabled === true,
            'an unmerged view must send only its pending key, got: '
              + JSON.stringify(firstBody)
          );

          ctx.fireStorage(JSON.stringify({
            focusModeEnabled: false,
            _sharedWriteMeta: {
              writeId: 1,
              writerId: 'window-old',
              changedKeys: ['focusModeEnabled'],
              hydrated: true,
              asrAuthoritative: true,
              knownKeyWrites: {
                focusModeEnabled: {
                  writeId: 1,
                  writerId: 'window-old',
                },
              },
            },
          }));
          assert(
            ctx.S.focusModeEnabled === true,
            'an older explicit token must not replace a newer pending edit'
          );

          // A later edit happens while the partial write is in flight. The
          // successful response snapshot is authoritative for untouched keys,
          // but must not overwrite this still-pending local value.
          ctx.win.slopFilterEnabled = true;
          ctx.mod.saveSettings();
          assert(ctx.S.slopFilterEnabled === true, 'the in-flight edit updates shared state');
          await tick();
          assert(ctx.postCalls.length === 1, 'the later edit queues behind the first POST');

          // The other window explicitly chooses the value this stale local
          // view already holds. The metadata still represents a newer user
          // mutation and must protect the key from the delayed response.
          ctx.fireStorage(JSON.stringify({
            mergeMessagesEnabled: false,
            _sharedWriteMeta: {
              writeId: 102,
              writerId: 'window-b',
              changedKeys: ['mergeMessagesEnabled'],
              hydrated: true,
              asrAuthoritative: true,
            },
          }));

          ctx.postCalls[0].resolve(response(
            true,
            200,
            '"conversation-settings-1"',
            {
              success: true,
              settings: {
                independentAsrEnabled: true,
                proactiveVisionEnabled: false,
                slopFilterEnabled: false,
                focusModeEnabled: true,
                mergeMessagesEnabled: true,
                noiseReductionEnabled: true,
                avatarReactionBubbleEnabled: false,
              },
              revision: 1,
              decisions: {},
            }
          ));
          await tick();
          await tick();

          assert(ctx.S.settingsHydrated === true, 'the complete success snapshot hydrates the view');
          assert(
            ctx.S.independentAsrEnabled === true,
            'an untouched field hydrates from the successful partial-write response'
          );
          assert(
            ctx.S.slopFilterEnabled === true,
            'an edit made while the request was in flight remains pending'
          );
          assert(
            ctx.S.mergeMessagesEnabled === false,
            'a same-value explicit cross-window edit survives the delayed response'
          );
          assert(
            ctx.S.avatarReactionBubbleEnabled === true,
            'a pre-request cross-window edit absent from the partial body survives'
          );
          assert(
            ctx.store.get('neko_noise_reduction') === '1',
            'accepted shared noise reduction mirrors into the legacy cache'
          );
          assert(
            ctx.runtime.stoppedSpeech === 1
              && ctx.runtime.stoppedScreening === 1
              && ctx.runtime.stoppedTracks === 1,
            'hydrating the privacy winner stops active vision runtime'
          );
          assert(ctx.postCalls.length === 2, 'the queued edit runs after hydration');
          const secondBody = JSON.parse(ctx.postCalls[1].opts.body);
          assert(
            secondBody.independentAsrEnabled === true
              && secondBody.proactiveVisionEnabled === false
              && secondBody.slopFilterEnabled === true
              && secondBody.mergeMessagesEnabled === false
              && secondBody.noiseReductionEnabled === true
              && secondBody.avatarReactionBubbleEnabled === true,
            'the queued retry uses the reconciled full snapshot plus the pending edit'
          );
          const reconciledLocal = JSON.parse(ctx.store.get('project_neko_settings'));
          assert(
            reconciledLocal.independentAsrEnabled === true
              && reconciledLocal.proactiveVisionEnabled === false
              && reconciledLocal.slopFilterEnabled === true,
            'the reconciled success snapshot persists for offline restart'
          );
          assert(
            reconciledLocal.mouseTrackingEnabled === false,
            'success reconciliation preserves local-only settings'
          );
          assert(
            reconciledLocal._sharedWriteMeta.knownKeyWrites.focusModeEnabled
              .confirmedRevision === 1,
            'the acknowledged explicit token must carry its confirmed revision'
          );
          ctx.postCalls[1].resolve(response(
            true,
            200,
            '"conversation-settings-2"',
            {
              success: true,
              settings: secondBody,
              revision: 2,
              decisions: {},
            }
          ));
          await tick();
        }

        async function runPartialResetSnapshotScenario() {
          const ctx = makeContext(true);
          await tick();
          await tick();
          assert(ctx.S.slopFilterEnabled === false, 'the harness starts with stale local slop');

          ctx.win.focusModeEnabled = true;
          ctx.mod.saveSettings();
          await tick();
          await tick();
          assert(ctx.postCalls.length === 1, 'the pre-hydration edit must POST');
          assert(
            !('X-Conversation-Settings-Full-Snapshot' in ctx.postCalls[0].opts.headers),
            'a dirty-only pre-hydration write must not clear the reset tombstone'
          );

          ctx.postCalls[0].resolve(response(
            true,
            200,
            '"conversation-settings-1"',
            {
              success: true,
              settings: { focusModeEnabled: true },
              revision: 1,
              reset: true,
              decisions: {},
            }
          ));
          await tick();
          await tick();

          assert(
            ctx.S.focusModeEnabled === true
              && ctx.S.slopFilterEnabled === true
              && ctx.S.independentAsrEnabled === false,
            'a partial reset response must materialize defaults without losing the edit'
          );
          const restoredLocal = JSON.parse(ctx.store.get('project_neko_settings'));
          assert(
            restoredLocal.focusModeEnabled === true
              && restoredLocal.slopFilterEnabled === true
              && restoredLocal.independentAsrEnabled === false,
            'the partial reset response must replace stale localStorage values'
          );
        }

        async function runPartialConfirmedWatermarkScenario() {
          const ctx = makeContext(true);
          await tick();
          await tick();

          ctx.fireStorage(JSON.stringify({
            mergeMessagesEnabled: true,
            _sharedWriteMeta: {
              writeId: 110,
              writerId: 'window-b',
              changedKeys: ['mergeMessagesEnabled'],
              hydrated: true,
              asrAuthoritative: true,
            },
          }));
          ctx.win.focusModeEnabled = true;
          ctx.mod.saveSettings();
          await tick();
          await tick();

          assert(ctx.postCalls.length === 1, 'the dirty-only edit must POST');
          const firstBody = JSON.parse(ctx.postCalls[0].opts.body);
          assert(firstBody.focusModeEnabled === true, 'the pending key is sent');
          assert(
            !Object.prototype.hasOwnProperty.call(
              firstBody,
              'mergeMessagesEnabled'
            ),
            'the sibling cross-window key stays outside the partial request'
          );

          ctx.postCalls[0].resolve(response(
            true,
            200,
            '"conversation-settings-1"',
            {
              success: true,
              settings: {
                independentAsrEnabled: false,
                focusModeEnabled: true,
                mergeMessagesEnabled: true,
                slopFilterEnabled: false,
              },
              revision: 1,
              decisions: {},
            }
          ));
          await tick();
          await tick();

          ctx.win.slopFilterEnabled = true;
          ctx.mod.saveSettings();
          await tick();
          await tick();

          assert(ctx.postCalls.length === 2, 'the next edit sends a full snapshot');
          ctx.postCalls[1].resolve(response(
            false,
            412,
            '"conversation-settings-2"',
            {
              success: false,
              settings: {
                independentAsrEnabled: false,
                focusModeEnabled: true,
                mergeMessagesEnabled: false,
                slopFilterEnabled: false,
              },
              revision: 2,
              decisions: {},
            }
          ));
          await tick();
          await tick();

          assert(ctx.postCalls.length === 3, 'the CAS mismatch retries once');
          const retryBody = JSON.parse(ctx.postCalls[2].opts.body);
          assert(
            retryBody.mergeMessagesEnabled === false,
            'a sibling edit confirmed by the prior response no longer masks server state'
          );
          assert(
            retryBody.slopFilterEnabled === true,
            'the still-pending local edit survives the CAS merge'
          );

          ctx.postCalls[2].resolve(response(
            true,
            200,
            '"conversation-settings-3"',
            {
              success: true,
              settings: retryBody,
              revision: 3,
              decisions: {},
            }
          ));
          await tick();
        }

        async function main() {
          await runBootMetadataScenario();
          await runScenario(false);
          await runScenario(true);
          await runAcknowledgedDirtyScenario();
          await runSuccessfulPartialSnapshotScenario();
          await runPartialResetSnapshotScenario();
          await runPartialConfirmedWatermarkScenario();
          console.log('CAS_HARNESS_OK');
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
        "settings CAS harness failed\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "CAS_HARNESS_OK" in result.stdout

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

def test_blocked_greeting_check_retries_without_home_tutorial_state():
    source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")

    blocked_branch = source.split("if (_isGreetingCheckBlocked()) {", 1)[1].split(
        "try {",
        1,
    )[0]
    assert "sendHomeTutorialState(" not in blocked_branch
    assert "_scheduleGreetingCheckRetry();" in blocked_branch

def test_greeting_check_defers_until_new_user_icebreaker_ends():
    source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")

    send_block = source.split("function _sendGreetingCheckIfReady()", 1)[1].split(
        "function _onModelReady()",
        1,
    )[0]
    assert send_block.index("if (_deferGreetingCheckForNewUserIcebreaker())") < send_block.index(
        "if (_isGreetingCheckBlocked())"
    )

    defer_block = source.split("function _deferGreetingCheckForNewUserIcebreaker()", 1)[1].split(
        "function _sendGreetingCheckIfReady()",
        1,
    )[0]
    blocking_block = source.split("function isNewUserIcebreakerBlockingGreeting(reason)", 1)[1].split(
        "function normalizeAssistantTurnId(turnId)",
        1,
    )[0]
    assert "return isNewUserIcebreakerActiveForGreeting();" in blocking_block
    assert "isTutorialReleaseGreetingReason" not in blocking_block
    active_block = source.split("function isNewUserIcebreakerActiveForGreeting()", 1)[1].split(
        "function isNewUserIcebreakerPeriodActive()",
        1,
    )[0]
    assert "window.NekoNewUserIcebreakerState" in active_block
    assert "state.isPeriodActive()" in active_block
    assert "window.newUserIcebreaker.getActiveSession()" in active_block
    assert "return isNewUserIcebreakerStorePeriodActive();" in active_block
    assert "hasRuntimeState" not in active_block
    period_block = source.split("function isNewUserIcebreakerPeriodActive()", 1)[1].split(
        "function isNewUserIcebreakerBlockingGreeting(reason)",
        1,
    )[0]
    assert "isNewUserIcebreakerActiveForGreeting()" in period_block
    assert "isNewUserIcebreakerStorePeriodActive()" not in period_block
    assert "readNewUserIcebreakerStore()" not in period_block
    store_block = source.split("function isNewUserIcebreakerStorePeriodActive()", 1)[1].split(
        "function isNewUserIcebreakerActiveForGreeting()",
        1,
    )[0]
    assert "readNewUserIcebreakerStore()" in store_block
    assert "isNewUserIcebreakerEntryBlocking(entry)" in store_block
    entry_block = source.split("function isNewUserIcebreakerEntryBlocking(entry)", 1)[1].split(
        "function isNewUserIcebreakerStorePeriodActive()",
        1,
    )[0]
    assert "entry.completed !== true" in entry_block
    assert "isRecentNewUserIcebreakerEntry(entry)" in entry_block
    assert "return false;" in store_block
    assert "sendHomeTutorialState(" not in defer_block
    assert "_scheduleGreetingCheckRetry();" in defer_block
    assert "S._greetingCheckPending = false;" not in defer_block
    assert "S._greetingCheckReason = '';" not in defer_block
    assert "_resetGreetingCheckRetry(true);" not in defer_block
    assert "var greetingReason = S._greetingCheckReason || (greetingIsSwitch ? 'character-switch' : 'ws-open');" in send_block
    assert "sendHomeTutorialState(" not in send_block
    assert "reason: greetingReason" in send_block
    assert "if (S._startupGreetingReleasePending) {" in send_block
    assert send_block.index("if (S._startupGreetingReleasePending)") < send_block.index(
        "if (_deferGreetingCheckForNewUserIcebreaker())"
    )
    assert "window.addEventListener('neko:new-user-icebreaker-ended'" in source
    assert "function _consumeGreetingCheckForNewUserIcebreaker()" not in source

    assert "function _isTutorialBlockingGreeting()" not in source
    assert "function isHomeTutorialLockedForGreeting()" not in source

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

def test_text_session_start_stops_an_active_microphone():
    # PR #2345 removed streaming.py's audio-branch session rebuild, so a
    # microphone left running into a text session has every frame accepted at
    # ingress and dropped at routing — no status, no recovery, mic toggle
    # required. The user's most recent explicit action wins: installing a text
    # session stops recording. One-directional on purpose; rebuilding the audio
    # session from the ingress path would re-arm the start_session teardown
    # ping-pong instead.
    websocket_source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")

    started = websocket_source.split(
        "S.isTextSessionActive = response.input_mode === 'text';",
        1,
    )[1].split("var _tiaStarted", 1)[0]

    # CodeRabbit: assert the ENCLOSURE, not three independent substrings. Bare
    # existence checks over the whole block would still pass if the stop call
    # were moved out of the text branch, or if the S.isRecording check belonged
    # to some unrelated path -- exactly the contract this test exists to hold.
    # So slice the smallest guard body and assert the call lives inside it.
    guard_open = "if (response.input_mode === 'text'\n"
    assert guard_open in started, "the teardown must be gated on a text session"
    guard_body = started.split(guard_open, 1)[1].split("\n                    }", 1)[0]

    # Both conditions belong to that one guard, not to separate statements.
    assert "S.isRecording === true" in guard_body
    assert "typeof window.stopRecording === 'function'" in guard_body

    # notifyServer:false is load-bearing, not cosmetic: the default path sends
    # pause_session, which websocket_router.py maps to an ungated end_session()
    # against the text session this very ack just installed, 500 ms before
    # app-buttons.js sends the queued user text.
    assert "window.stopRecording({ notifyServer: false });" in guard_body
    # And the call appears nowhere else in the handler, guarded or not.
    assert started.count("window.stopRecording(") == 1
    assert "window.stopRecording();" not in started
    # stopMicCapture would reject the in-flight text-start promise outright.
    # Match the CALL form: the comment above deliberately names the function.
    assert "window.stopMicCapture(" not in started

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

def test_every_start_session_send_carries_a_request_id():
    # #2539 / Codex P2. The ack names the start it answers, and the receiver
    # ignores acks that name a different one. A send site that forgets the id
    # gets an anonymous ack back, which every window treats as "mine" -- the
    # exact failure the id exists to prevent. Discovered rather than listed, so
    # a NEW send site cannot slip past this.
    sources = {
        "app-buttons.js": APP_BUTTONS_PATH.read_text(encoding="utf-8"),
        "app-websocket.js": APP_WEBSOCKET_PATH.read_text(encoding="utf-8"),
    }
    found = 0
    for name, source in sources.items():
        cursor = 0
        while True:
            at = source.find("action: 'start_session'", cursor)
            if at == -1:
                break
            cursor = at + 1
            found += 1
            payload_end = source.index("}", at)
            payload = source[at:payload_end]
            assert "request_id: window.sessionStartRequestId(" in payload, (
                f"{name}: a start_session send at offset {at} carries no request id"
            )
    assert found >= 4, "the send sites were not discovered; the search anchor moved"

    # Read off the flow's OWN owner token, never the shared slot: a start
    # displaced during its reconnect await would otherwise stamp the newer
    # start's id onto its own stale request.
    state_source = APP_STATE_PATH.read_text(encoding="utf-8")
    assert "window.sessionStartRequestId = function (owner) {" in state_source
    assert "startRequestIdByOwner.get(owner)" in state_source

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

def test_microphone_switch_requires_a_live_committed_replacement():
    capture_source = APP_AUDIO_CAPTURE_PATH.read_text(encoding="utf-8")
    select_fn = _code_only(
        _block_after(capture_source, "async function selectMicrophone(deviceId) {")
    )
    await_marker = "const microphoneStarted = await startMicCapture();"
    success_marker = "if (microphoneStarted === true) {"
    retry_marker = "const latestSelectionNeedsRetry = ("
    assert "while (true) {" in select_fn
    assert await_marker in select_fn
    assert success_marker in select_fn
    assert retry_marker in select_fn
    assert select_fn.index(await_marker) < select_fn.index(success_marker)
    assert select_fn.index(success_marker) < select_fn.index(retry_marker)
    retry_condition = select_fn.split(retry_marker, 1)[1].split(");", 1)[0]
    assert (
        "microphoneSelectionGeneration !== selectionGenerationForRestart"
        in retry_condition
    )
    assert "micStartGeneration === expectedRestartGeneration" in retry_condition
    assert "S.voiceInputRouteBlocked !== true" in retry_condition
    assert select_fn.index(retry_marker) < select_fn.index(
        "await window.startScreenSharing();"
    )

    start_fn = _code_only(
        _block_after(capture_source, "async function startMicCapture() {")
    )
    assert "let microphoneSelectionGeneration = 0;" in capture_source
    assert "microphoneSelectionGeneration += 1;" in capture_source
    assert len(re.findall(r"S\.selectedMicrophoneId\s*=(?!=)", capture_source)) == 1, (
        "all microphone-selection writes must go through the generation-tracked helper"
    )
    finish_cancelled = _code_only(
        _block_after(
            capture_source,
            "function finishCancelledMicStart(micElement, micStartToken) {",
        )
    )
    assert (
        start_fn.count(
            "return finishCancelledMicStart(_mic, micStartToken);"
        )
        == 4
    )
    assert "if (hasLiveCommittedMicrophonePipeline()) {" in finish_cancelled
    assert "pendingMicStartUiOwnerToken !== micStartToken" in finish_cancelled
    assert "S.isRecording = false;" in finish_cancelled
    assert "window.isRecording = false;" in finish_cancelled

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
