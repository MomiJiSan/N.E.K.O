# 本地流式 ASR 服务接入、准备与热重载（M1b/M1c/M2）

`local_streaming` 是用户显式选择的独立 ASR provider，连接 N.E.K.O-ASR 的 v1 WebSocket 服务。它复用现有独立 ASR 的音频队列、Smart Turn、来源票据、VoiceInputRegistry、字幕预览和 Core 聊天派发。宿主不导入 sherpa/ONNX ASR 后端；模型进程由独立服务持有。

## 配置和启动

推荐显式启用托管模式。先在独立服务仓库安装 Python 3.12 依赖，准备固定权重及一条公开的非空语音 WAV，再给宿主设置受信本地路径：

```powershell
$env:NEKO_ASR_SERVICE_PYTHON = 'C:\N.E.K.O-ASR\.venv\Scripts\python.exe'
$env:NEKO_ASR_SERVICE_ROOT = 'C:\N.E.K.O-ASR'
$env:NEKO_ASR_MODEL_DIR = 'C:\models\paraformer'
$env:NEKO_ASR_SMOKE_WAV = 'C:\samples\public-speech.wav'
$env:NEKO_ASR_SERVICE_THREADS = '1'
$env:NEKO_ASR_IDLE_SECONDS = '300'
```

四个路径必须一起设置且为绝对路径。WAV 为 16kHz mono PCM16、0.1–10 秒的公开/授权非空语音。HTTP 不接收可执行文件、模型路径或下载 URL。宿主生成随机 token、启动隐藏的 lazy supervisor（动态 loopback 端口），把 Windows 启动链及模型进程纳入自己持有的 Job；外部进程不属于它。

点击开启语音后，网页与 Electron 共用的 prepare 步骤先创建 operation，再轮询到 ready。此时还没发送正式 start_session、也没开启麦克风；可用「取消准备」按钮取消自己的等待。文件验证、模型加载、预热和非空语音 smoke 均成功后才建立会话。缺文件/损坏模型/超时明确失败，不下载、不自动换云端。

若使用手动管理的外部服务，保留 M1b 配置方式：

1. 先在独立 N.E.K.O-ASR 仓库按 README 配置 Python 3.12、固定版本的 Paraformer INT8 权重及随机 token，再执行 `uv run neko-asr --model-dir <权重目录> --threads 1`。默认监听 `127.0.0.1:50120`。
2. 启动宿主前设置同一个 `NEKO_ASR_TOKEN`。宿主使用 Python 3.11；`NEKO_ASR_URL` 默认 `ws://127.0.0.1:50120/v1/asr/stream`。允许显式 loopback 端口和 IPv6 `::1`，拒绝远程地址、URL 内凭据、查询参数和 fragment。不经代理连接。
3. 沿用项目既有流程准备宿主 Silero VAD 和 Smart Turn 模型。ASR 服务加载成功不代表宿主断句模型也已就绪。
4. 在语音识别设置中启用独立 ASR，选择「本地流式服务」。与 faster-whisper 共用同一个 `independentAsrProviderPreference`，二者互斥；下次会话使用新选择。

token 至少 32 字符、最多 256 字符，为无空白的可打印 ASCII。使用环境配置向两个进程提供相同 token，不把它写入浏览器设置、URL 或日志。网页/Electron 客户端均连接宿主既有通道，不直接访问 ASR 服务。

兼容 eager 外部服务时，仅当 prepare 返回 HTTP404（旧版本无控制面）或 HTTP501（明确不支持），宿主才严格验证其健康状态、固定模型及数据合同后将其视作已驻留；不伪造 prepare 租约、不续租、不负责它的 idle 或退出。managed、普通 HTTP409、新 lazy 服务失败或 health503 不走此兼容路径。

`/api/config/core_api` 的 `localStreamingAsrConfigured` 仅检查环境配置格式，不探测网络、不加载模型、不证明 ready。真实连接会核对 hello、协议版本、实例标识、PCM 格式、partial 及 finish 能力。配置错误、鉴权失败、服务关闭均明确失败；显式选择不会自动切到云端。原 `localAsrAvailable` 仍只表示 faster-whisper 依赖存在。

不支持独立 ASR 的 Core 仍保持原有限制。关闭独立 ASR 主开关不会因为保存了本地 provider 而启动独立识别；无法使用的已保存选项可以关闭。

## 语句与交付合同

- 使用现有音频标准化输出的 16kHz mono PCM16LE，不再重采样。每句冻结 `(generation, buffer_epoch, utterance_id)` 并对应一个新的服务 UUID。
- 每句由现有 Smart Turn/manual 事件 commit，不用全会话 `finish_and_drain` 代替逐句结束；模型 endpoint 不取得聊天结束权。
- partial 是可修订的完整快照，经现有 `user_transcript_preview` 事件更新字幕；final 经既有转写回调、VoiceInputRegistry 和 Core 派发器交付。
- 单帧最多 6400 字节。宿主出队 PCM 继续通过现有 audio hold 计入 2 秒容量，直到其所有子帧收到 `consumed`。服务的 accepted、consumed、物理写入均不直接表示聊天历史确认。
- 来源票据以宿主原请求为结算单位。大请求拆帧后部分发送失败，整请求保守保持 UNKNOWN，不虚报部分成功，不自动重放消费证据不确定的音频。
- clear 先阻止旧 scope 回调，再等 cancelled，之后可开新句。final 与 cancel 竞争时，服务可以回复终态 `cancelled(terminal_state="final")`，这表示旧 native stream 已结束，不撤销已经产生的 final；宿主已 clear 的结果仍被抑制。
- 每句至多一个 final，终态后不接受旧 partial。服务每连接最多 64 句，宿主仅在安全的语句边界换连接；不会重放之前音频。
- `finish_and_drain` 排空整场并等待既有回调队列。取消、断线、超时释放自己持有的音频和任务，不影响其他 owner。

## 验证

单元及状态机测试：

```powershell
uv run python -m pytest tests/unit/test_asr_local_streaming_worker.py tests/unit/test_asr_local_streaming_selection.py tests/unit/test_local_streaming_preferences.py
node --test tests/frontend/local_streaming_preference.test.cjs
uv run python -m pytest tests/frontend/test_local_streaming_asr_settings.py
```

真实 CPU 链路测试（先启动服务、安装宿主 endpointing 模型，并设置同一个 URL/token）：

```powershell
$env:NEKO_ASR_E2E_WAV = 'C:\samples\speech.wav'
$env:NEKO_ASR_E2E_REPORT = 'C:\results\local-asr-chain.json'
uv run python -m pytest tests/integration/test_local_streaming_asr.py -q
```

音频需为公开/已获授权的 mono PCM16 16kHz WAV。测试使用真实 VAD、Smart Turn、Paraformer、WebSocket 和宿主派发，外部聊天生成与浏览器连接使用记录器；它不调用远程 LLM。具体替代边界在测试及结果 JSON 中声明。不设置音频时测试明确 skip，不等同真实链路已验证。

## 租约与资源回收

prepare 同模型 single-flight，每个请求有独立 UUID/租约；取消一个等待者不取消其他人。未知/丢失 POST 响应通过 request_id DELETE 补偿，迟到完成不能重新发布取消的请求。准备默认总期限 10 秒，前端 prepare 总预算 20 秒；既有正式会话 15 秒计时从发送 start_session 后才开始。

语音会话保留已准备模型的资源租约，后台最多每 20 秒续租；即使既有资源优化在 25 秒后关闭 warm WebSocket，模型仍由该语音模式拥有。前端 prepare 等待者与运行中的语音 owner 分别销账，不会因为 UI 释放等待者而卸载正在用的模型。临时 hard mute、焦点暂停和游戏接管不等同整个语音模式结束。

普通关麦沿用 capture_end 控制，结束正常 capture 后释放同一资源 owner，停止续租；无需结束正在回复的 LLM 会话。空句/尚在硬件启动中的本窗口 stop 也销账，辅助窗口的广播 voiceChatActive 不被当作本窗口采集 ownership。失败/取消/新会话接管都只清理自己捕获的 owner。

最后租约、WebSocket pin、活动语句和在途调用均归零，才开始默认 300 秒 idle timer；新 owner 使旧 timer 失效。idle 卸载 native 模型进程、保留 supervisor，下一次 prepare 创建新 worker。首版不承诺在活跃语音模式中卸载模型，再靠起声缓冲完成冷恢复。

应用关闭先封锁新准备请求，使用实例标识进行优雅退出，再用自己的 Job 作为异常清理兜底；不按端口或进程名杀用户进程。手动外部服务不被宿主终止。单次 prepare lease 默认 60 秒，过期不会因查询而复活。

服务端有有界原型额度：64 个同时 lease、128 条近期 operation、4096 条终态 request tombstone；达到额度明确拒绝，新 supervisor 实例重置额度。安装/重试流程不能把消费结果不确定的语音自动重放。

真实托管链路测试可在上述配置后另设 `NEKO_ASR_E2E_MANAGED=1`，复用原集成命令。独立冷启动/续租/idle/owned PID 验证设置 `NEKO_ASR_REAL_LIFECYCLE=1` 并运行 `tests/integration/test_local_service_lifecycle.py`；测试缩短 idle/lease，仅用于验证状态机，不改变生产默认值。

## M2 热重载控制

在上述四个托管路径之外，可由操作者配置可信 catalog 与独立状态目录：

```powershell
$env:NEKO_ASR_CATALOG = 'C:\asr-config\catalog.json'
$env:NEKO_ASR_STATE_DIR = 'C:\asr-state'
$env:NEKO_ASR_MEMORY_BUDGET_MB = '2048'
```

catalog/state-dir 必须成对设置、为绝对路径，并使用托管配置。catalog 的生成、manifest 和信任边界见独立服务仓库 `docs/hot-reload.md`。catalog 固定候选 ID、源目录、manifest 路径及其原始字节 SHA-256；文件校验通过后复制至不可变目录，不原位替换正在映射的模型。客户端只能提交 candidate_id，不能传模型路径、命令或下载 URL。未配置 catalog 的服务继续支持原 prepare，但不能 reload。

宿主提供以下接口，均沿用现有本地请求/Origin/CSRF 检查；reload 还检查当前 Core 是否支持独立 ASR。调用者使用 canonical UUID request_id，先完成 prepare，再发 reload。这里是控制 API 原型，尚无模型选择界面。

| 方法与宿主路径 | 请求与语义 |
| --- | --- |
| GET `/api/config/local-asr/candidates` | 查询已运行服务，不启动进程、不加载模型；仅返回安全候选 metadata |
| POST `/api/config/local-asr/reload` | `{request_id, enabled:true, provider:"local_streaming", candidate_id}`；返回 operation，重复 ID 查询同一操作当前状态 |
| GET `/api/config/local-asr/reloads/{operation_id}` | 轮询 state、active_changed、retirement_pending、worker、generation、预算与计时 |
| DELETE `/api/config/local-asr/reload-requests/{request_id}` | 丢失 POST 响应时取消/找回操作；取消已跨发布边界时保留 active_changed=true，不伪称回滚 |
| DELETE `/api/config/local-asr/reloads/{operation_id}` | 取消尚未发布的候选 |
| POST `/api/config/local-asr/reloads/{operation_id}/finalize` | `{}`；提前结束旧模型保留窗口，等待其语句及交付销账 |
| POST `/api/config/local-asr/reloads/{operation_id}/rollback` | `{}`；仅将未来语句回到仍健康、未退休的旧 worker |

宿主操作记录保留最初服务 endpoint；不会把旧 operation 发给后来重启的服务。取消/rollback/finalize 的已接受网络任务由 manager 持有，HTTP 等待者取消不使它们变成无主任务，manager 关闭时先收敛，再关闭自有服务。外部服务始终不被宿主 kill。达到记录额度明确拒绝；未知取消不能联系服务时不会假报其未发布。

服务 hello 的 `per_utterance_binding=true` 开启逐句身份校验。宿主冻结每句 `(worker_instance, route_generation, bundle_id)`，同句所有事件必须一致；下一句允许换 worker，generation 不能倒退，同 generation 不能换模型。回滚也递增 generation。旧句迟到 partial/error、重复 final 继续被既有终态 fence 抑制，字幕与聊天交付仍经原 Smart Turn、VoiceInputRegistry 和 Core 链路。

服务最多同时保留两个模型 worker；校验、预算、加载、预热、非空 smoke 成功后才发布。发布前失败保留旧 route。旧语句及 final 的发送交付 pin 未销账时不退休旧 worker；长句超出排空期限显示 drain_deferred，阻止第三个模型加载。默认旧模型保留 30 秒以支持回滚；预算不足返回 requires_pause，不自动停麦维护切换。内存准入是声明预算、实际 RSS 与系统可用内存的预检，不是操作系统硬内存限制。

本轮真实链路还暴露并修复既有跨句准备竞态：pending turn 进入 ACTIVE 后，Smart Turn 仍可能处于异步准备中。所有路由现在均沿用既有 activation fence，在有界 active prefix 中保留此间尚未交付的音频，准备完成后一次性按序交给 dispatcher。旧式匿名音频保留匿名 span；不能与有来源范围的音频混合。容量不足明确失败，不驱逐已接受前缀；关闭/身份失效会清空旧 owner，迟到准备结果不能发送旧音频。该改动有 gate 控制的回归测试，覆盖两帧保留、预算、按序唯一交付和撤销。

发布后候选故障只改变未来句路由；已经在故障 worker 消费的句明确失败，不重放、不重复插入聊天回合。回滚 marker 恢复期间，新句在尚未接收音频前有界等待；旧句继续自己的 owner。旧模型已经退休时返回 ROLLBACK_UNAVAILABLE，后续需 prepare。磁盘 marker 从不代表 ready；shutdown 可能完成持久提交但阻止 RAM route 发布，下一次启动仍需加载/预热/smoke。

M2 真实测试使用相同固定权重的两个可信 manifest 版本，证明生命周期和交付隔离，不代表模型质量提升。复现宿主 M2 测试需安装既有 VAD/Smart Turn 资产，配置四个托管路径，并显式启用 `NEKO_ASR_REAL_RELOAD=1`，运行 `tests/integration/test_local_streaming_asr_reload.py`。测试自行建立临时可信 catalog，分别运行基线与热切换链路，真实音频取配置中的公开 smoke WAV。

安装包分发、设备麦克风、声学回声、远端生成/TTS 和长期对话仍需整机验收。
