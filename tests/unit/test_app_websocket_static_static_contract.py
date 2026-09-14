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

def test_game_route_speech_cancel_is_scoped_to_the_sdk_correlation_id():
    source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")
    block = _block_after(
        source,
        "} else if (response.type === 'game_route_speech_cancel') {",
    )

    assert "response.sdk_speech_correlation_id" in block
    assert "cancelledCorrelationId === S.currentPlayingSpeechCorrelationId" in block
    assert "applyUserActivityCancel(" in block

def test_late_stt_gate_cannot_reactivate_the_most_recently_ended_route():
    source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")
    state_source = APP_STATE_PATH.read_text(encoding="utf-8")
    prune_opener = "function pruneRecentlyEndedGameRouteIdentities() {"
    remember_opener = "function rememberEndedGameRouteIdentity(gameType, sessionId, routeInstanceId) {"
    check_opener = (
        "function isRecentlyEndedGameRouteIdentity(gameType, sessionId, routeInstanceId) {"
    )
    prune_body = _block_after(source, prune_opener)
    remember_body = _block_after(source, remember_opener)
    check_body = _block_after(source, check_opener)
    node_path = shutil.which("node")
    if not node_path:
        pytest.skip("node is not installed; skipping ended-route identity harness")

    result = run_node_script(
        node_path,
        textwrap.dedent(
            f"""
            const GAME_ROUTE_ENDED_IDENTITY_LIMIT = 8;
            const GAME_ROUTE_ENDED_IDENTITY_TTL_MS = 2 * 60 * 1000;
            let now = 1000000;
            Date.now = () => now;
            const S = {{ gameRouteRecentlyEndedIdentities: [] }};
            {prune_opener}
            {prune_body}
            }}
            {remember_opener}
            {remember_body}
            }}
            {check_opener}
            {check_body}
            }}
            function assert(value, message) {{ if (!value) throw new Error(message); }}

            rememberEndedGameRouteIdentity('example-game', 'legacy-session', '');
            assert(!isRecentlyEndedGameRouteIdentity(
              'example-game', 'legacy-session', 'identified-successor'
            ), 'identified successor of a generation-less route was rejected');
            assert(isRecentlyEndedGameRouteIdentity(
              'example-game', 'legacy-session', ''
            ), 'generation-less late gate for a generation-less route was not rejected');

            now += 1;
            rememberEndedGameRouteIdentity('example-game', 'reused-session', 'generation-A');
            now += 1;
            rememberEndedGameRouteIdentity('example-game', 'reused-session', 'generation-B');
            assert(isRecentlyEndedGameRouteIdentity(
              'example-game', 'reused-session', 'generation-A'
            ), 'older ended generation was forgotten after its successor closed');
            assert(isRecentlyEndedGameRouteIdentity(
              'example-game', 'reused-session', 'generation-B'
            ), 'latest ended generation was not rejected');
            assert(!isRecentlyEndedGameRouteIdentity(
              'example-game', 'reused-session', 'generation-C'
            ), 'new generation reusing a session was rejected');
            assert(isRecentlyEndedGameRouteIdentity(
              'example-game', 'reused-session', ''
            ), 'generation-less late gate for an identified ended route was not rejected');
            assert(!isRecentlyEndedGameRouteIdentity(
              'example-game', 'new-session', 'generation-A'
            ), 'different session was rejected');

            S.gameRouteRecentlyEndedIdentities = [];
            for (let i = 0; i < 10; i += 1) {{
              now += 1;
              rememberEndedGameRouteIdentity('example-game', `session-${{i}}`, `generation-${{i}}`);
            }}
            assert(S.gameRouteRecentlyEndedIdentities.length === 8, 'ended identity history exceeded capacity');
            assert(!isRecentlyEndedGameRouteIdentity(
              'example-game', 'session-0', 'generation-0'
            ), 'capacity eviction did not release the oldest identity');
            assert(isRecentlyEndedGameRouteIdentity(
              'example-game', 'session-9', 'generation-9'
            ), 'capacity pruning removed the newest identity');

            now += GAME_ROUTE_ENDED_IDENTITY_TTL_MS + 1;
            pruneRecentlyEndedGameRouteIdentities();
            assert(S.gameRouteRecentlyEndedIdentities.length === 0, 'expired identities were not released');
            """
        ),
        cwd=str(Path(__file__).resolve().parents[2]),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr

    ended_block = _block_after(source, "if (statusCode === 'GAME_ROUTE_ENDED') {")
    stt_gate_block = _block_after(
        source,
        "if (statusCode === 'GAME_VOICE_STT_GATE_ACTIVE') {",
    )
    window_block = _block_after(
        source,
        "} else if (response.type === 'game_window_state_change') {",
    )
    assert "gameRouteRecentlyEndedIdentities: []" in state_source
    assert "GAME_ROUTE_ENDED_IDENTITY_LIMIT = 8" in source
    assert "GAME_ROUTE_ENDED_IDENTITY_TTL_MS = 2 * 60 * 1000" in source
    assert ended_block.index("rememberEndedGameRouteIdentity(") < ended_block.index(
        "S.gameRouteActive = false;"
    )
    assert stt_gate_block.index("isRecentlyEndedGameRouteIdentity(") < stt_gate_block.index(
        "advanceGameRouteStateRevision();"
    )
    assert window_block.index("pruneRecentlyEndedGameRouteIdentities();") < window_block.index(
        "S.gameRouteActive = true;"
    )
    assert window_block.index("rememberEndedGameRouteIdentity(") < window_block.index(
        "S.gameRouteActive = false;"
    )

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

def test_voice_lifecycle_status_is_validated_and_exposed_to_ui():
    source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")

    assert "statusCode === 'ASR_LIFECYCLE_STATE'" in source
    assert "voiceInputLifecycleState" in source
    assert "voice-input-lifecycle-changed" in source
    assert "data-voice-input-state" in source

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

def test_independent_asr_failure_copy_matches_hard_route_in_all_locales():
    expected = {
        "en.json": (
            "Independent ASR unavailable. Voice input has stopped for this session. Check the independent ASR configuration, then start a new voice session.",
            "Enabled for the next voice session; it will not automatically switch to Omni if unavailable.",
            "{{providerKey}} is temporarily unavailable. Voice input has stopped for this session. It did not switch to another speech recognition service. Please start a new voice session later.",
        ),
        "es.json": (
            "El ASR independiente no está disponible. La entrada de voz se ha detenido para esta sesión. Revisa la configuración del ASR independiente y después inicia una nueva sesión de voz.",
            "Se activará en la próxima sesión de voz; no cambiará automáticamente a Omni si no está disponible.",
            "{{providerKey}} no está disponible temporalmente. La entrada de voz se ha detenido para esta sesión. No se cambió a otro servicio de reconocimiento de voz. Inicia una nueva sesión de voz más tarde.",
        ),
        "ja.json": (
            "独立 ASR を利用できないため、この音声セッションの入力を停止しました。独立 ASR の設定を確認してから、新しい音声セッションを開始してください。",
            "次の音声セッションから有効になります。利用できない場合も Omni へ自動的に切り替わりません。",
            "{{providerKey}} は一時的に利用できません。この音声セッションの入力を停止しました。別の音声認識サービスには切り替えていません。後でもう一度音声セッションを開始してください。",
        ),
        "ko.json": (
            "독립 ASR을 사용할 수 없어 이번 음성 세션의 입력을 중지했습니다. 독립 ASR 설정을 확인한 다음 새 음성 세션을 시작하세요.",
            "다음 음성 세션부터 활성화되며, 사용할 수 없어도 Omni로 자동 전환되지 않습니다.",
            "{{providerKey}}을(를) 일시적으로 사용할 수 없어 이번 음성 세션의 입력을 중지했습니다. 다른 음성 인식 서비스로 전환하지 않았습니다. 나중에 새 음성 세션을 시작하세요.",
        ),
        "pt.json": (
            "O ASR independente não está disponível. A entrada de voz foi interrompida nesta sessão. Verifique a configuração do ASR independente e depois inicie uma nova sessão de voz.",
            "Será ativado na próxima sessão de voz; não mudará automaticamente para o Omni se estiver indisponível.",
            "{{providerKey}} está temporariamente indisponível. A entrada de voz foi interrompida nesta sessão. O sistema não mudou para outro serviço de reconhecimento de voz. Inicie uma nova sessão de voz mais tarde.",
        ),
        "ru.json": (
            "Независимый ASR недоступен. Голосовой ввод в этом сеансе остановлен. Проверьте настройки независимого ASR, затем начните новый голосовой сеанс.",
            "Будет включён в следующем голосовом сеансе; при недоступности автоматического переключения на Omni не произойдёт.",
            "{{providerKey}} временно недоступен. Голосовой ввод в этом сеансе остановлен. Переключения на другую службу распознавания речи не произошло. Начните новый голосовой сеанс позже.",
        ),
        "zh-CN.json": (
            "独立 ASR 不可用，本次语音输入已停止。请检查独立 ASR 配置，然后重新开始语音会话。",
            "将在下次语音会话启用；不可用时不会自动切换到 Omni。",
            "{{providerKey}} 暂时不可用，本次语音输入已停止。未切换到其他语音识别服务，请稍后重新开始语音会话。",
        ),
        "zh-TW.json": (
            "獨立 ASR 無法使用，本次語音輸入已停止。請檢查獨立 ASR 設定，然後重新開始語音會話。",
            "將於下次語音會話啟用；無法使用時不會自動切換到 Omni。",
            "{{providerKey}} 暫時無法使用，本次語音輸入已停止。未切換到其他語音辨識服務，請稍後重新開始語音會話。",
        ),
    }

    for locale_name, copy in expected.items():
        locale = json.loads((LOCALES_PATH / locale_name).read_text(encoding="utf-8"))
        microphone = locale["microphone"]
        assert microphone["independentAsrFallback"] == copy[0]
        assert microphone["independentAsrNextSession"] == copy[1]
        assert microphone["independentAsrProviderUnavailable"] == copy[2]

def test_external_asr_preview_message_is_declared_app_state_field():
    app_state = APP_STATE_PATH.read_text(encoding="utf-8")

    assert "externalAsrPreviewMessage: null," in app_state

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

def test_every_start_session_send_sits_behind_the_ensure_websocket_gate():
    # The settings-sync gate lives in ensureWebSocketOpen(), so it only closes
    # the toggle-vs-session-start race if every start_session send awaits
    # ensureWebSocketOpen() right before it.
    websocket_source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")
    buttons_source = APP_BUTTONS_PATH.read_text(encoding="utf-8")

    checked = 0
    for source, ensure_call in (
        (websocket_source, "await ensureWebSocketOpen();"),
        (buttons_source, "await window.ensureWebSocketOpen();"),
    ):
        for match in re.finditer(r"action: 'start_session'", source):
            preceding = source[max(0, match.start() - 600):match.start()]
            assert ensure_call in preceding, (
                "start_session send not preceded by ensureWebSocketOpen(): ..."
                + source[max(0, match.start() - 120):match.end()]
            )
            checked += 1
    assert checked >= 4

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

def test_session_started_ack_latches_a_blocked_microphone_route():
    # The one clear of the latch that is NOT tied to a route verdict is user
    # intent (app-buttons.js, next to _pendingSessionStartMode = 'audio'). What
    # keeps that from opening the microphone onto a dead route is this branch:
    # the ack carries the settled route (send_session_started in notify.py), so
    # a still-blocked route re-latches before the start promise settles.
    #
    # It is also the only channel that reaches a window which never got an
    # ASR_INDEPENDENT_* status at all -- a fenced start emits none, which is the
    # case the backend's dedupe re-decision (#2539) exists to shrink but cannot
    # remove.
    websocket_source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")
    buttons_source = APP_BUTTONS_PATH.read_text(encoding="utf-8")

    # The clear-on-intent this backstops, in the flow that arms the audio start.
    assert "S.voiceInputRouteBlocked = false;" in buttons_source

    started_handler = websocket_source.split(
        "S.isTextSessionActive = response.input_mode === 'text';", 1
    )[1].split("var _tiaStarted", 1)[0]
    latch = started_handler.split("S.voiceInputRouteBlocked = true;", 1)[0].rsplit(
        "if (", 1
    )[1]
    # Only for a request this window actually made: the latch is set-only, so a
    # blocked verdict belonging to another window's start would stick and this
    # window's own healthy ack could not clear it.
    assert "_ackAnswersThisWindow" in latch
    assert "response.input_mode !== 'text'" in latch
    # Set-only, and only on a blocked verdict: the latch is sticky by design
    # (tearDownBlockedVoiceRoute relies on it surviving), and an ack that says
    # native/independent must not clear what a status verdict set.
    assert "response.microphone_route === 'blocked'" in latch
    # Guarded on the field being present, so an older backend that omits it
    # keeps its current behaviour rather than refusing every microphone.
    assert "response.microphone_route !== 'blocked'" not in started_handler

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

def test_blocked_route_refuses_to_open_the_microphone():
    # THE guard that closes the cold-start hole. On a cold voice start the mic
    # is opened only AFTER session_started -- i.e. after the failure status --
    # so a server-side lease revoke has nothing to revoke yet, and
    # startMicCapture's own refreshMicLease would re-claim the lease anyway
    # (_handle_voice_input_control enforces only generation monotonicity, and
    # the revoke reset the generation to -1). Placed at the top of
    # startMicCapture so it also covers the device-change restore callers.
    capture_source = APP_AUDIO_CAPTURE_PATH.read_text(encoding="utf-8")

    start_fn = capture_source.split("async function startMicCapture() {", 1)[1]
    head = start_fn.split("const _mic = micButton();", 1)[0]
    assert "S.voiceInputRouteBlocked === true" in head
    assert "return false;" in head

    # A refused start must unwind the starting-voice UI rather than throw --
    # throwing would replace the accurate ASR toast with a generic failure.
    assert "function abortVoiceStartForBlockedRoute()" in capture_source
    unwind = capture_source.split("function abortVoiceStartForBlockedRoute() {", 1)[
        1
    ].split("\n    }", 1)[0]
    for expected in (
        "S.isRecording = false;",
        "S.voiceStartPending = false;",
        "window.isMicStarting = false;",
    ):
        assert expected in unwind
    assert "throw" not in _code_only(unwind)

    buttons_source = APP_BUTTONS_PATH.read_text(encoding="utf-8")
    assert "window.abortVoiceStartForBlockedRoute();" in buttons_source

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

def test_bootstrap_route_snapshot_is_rejected_when_it_lands_late(template_name):
    """The init-time /route/active read must not re-open a route that just closed.

    The request can start while route A is active and resolve after A's `closed`
    websocket event has already been handled; dispatching the snapshot then
    re-opens a dead route on this page, which locks the chat window into its
    collapsed game layout and suppresses proactive chat for the rest of the
    round. Every close path advances the route state revision, so the bootstrap
    compares it across the request the way the reconnect reconciliation in
    app-websocket.js already does.

    Both templates carry their own copy of this IIFE, so both are checked --
    a guard in one of them is a guard in neither for the other window.
    """
    source = (TEMPLATES_DIR / template_name).read_text(encoding="utf-8")
    marker = "fetch('/api/game/route/active?lanlan_name="
    assert source.count(marker) == 1, template_name
    fetch_at = source.index(marker)
    prologue = source[max(0, fetch_at - 1200):fetch_at]
    assert "gameRouteStateRevision" in prologue, (
        f"{template_name} bootstrap does not capture the route state revision "
        "before its /route/active request"
    )
    handler = source[fetch_at:source.index("dispatchEvent", fetch_at)]
    assert "gameRouteStateRevision" in handler, (
        f"{template_name} bootstrap dispatches its snapshot without re-checking "
        "the route state revision, so a snapshot that lands after the route "
        "closed re-opens a dead route"
    )
    assert "return" in handler, (
        f"{template_name} bootstrap compares the revision but never bails out"
    )
