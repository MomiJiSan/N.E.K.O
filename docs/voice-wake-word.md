# 本地唤醒词激活（实验接入）

此入口附加在已经录入并启用的语音激活上。“悠宜 / yui”关键词命中豁免
声纹判定，普通语音继续按原声纹规则处理。它不是本人身份认证。仍然要求
输入权限、音频处理、CAM++ 与关键词模型就绪；模型故障不会被当作命中。

## 配置与安装

在项目目录运行（模型目录放在仓库以外）：

```powershell
uv sync --extra wake-word
uv run scripts/provision_wake_word_model.py --model-dir C:\NEKO-models\wake-word
$env:NEKO_WAKE_WORD_MODEL_DIR = 'C:\NEKO-models\wake-word'
```

当前还必须使用带时间戳修复的 `sherpa-onnx==1.13.8+neko.kws1`；上面的
extra 只安装上游基础依赖，不包含修复。准备阶段会拒绝未验证的运行库版本。
Windows 安装 VS 2022 C++ Build Tools、CMake、Git 后，可构建本地 wheel：

```powershell
./scripts/build_wake_word_runtime.ps1 -Python '<项目 Python 路径>' -OutputDirectory C:\NEKO-build\wake-word
$wakeRuntimeWheel = (Get-ChildItem C:\NEKO-build\wake-word\sherpa-onnx\dist\*.whl).FullName
uv run --no-sync --with $wakeRuntimeWheel python launcher.py
```

脚本固定上游提交 `11afbd009a7f8c08f4bcf2fc1b265d0df4670fbf`，应用仓库内
`scripts/patches/sherpa-onnx-kws-timestamps.patch` 后构建。输出目录须为新目录。
Windows 应使用示例中的短输出路径，避免 MSBuild FileTracker 的路径长度限制。
后续启动、离线评估和模型回归也应携带同一 `--with $wakeRuntimeWheel`；
直接从 PyPI 安装原版或仅执行 `uv sync` 不足以复现此修复。

从设置了该环境变量的终端启动后端。需要已有声纹档案并打开语音激活。
新 worktree 还需准备 CAM++ 权重；已有档案不等于本地模型文件已存在。
可用 `uv run scripts/prepare_speaker_model.py --source-cache <已有的 speaker_shadow/models 目录>`
复制并校验已有模型，缺失时语音激活会进入 `model_unavailable`。
未设置该变量时保留原有仅声纹入口；设置了无效目录会报告不可用，不能
用“未检测”冒充“命中”。模型准备不在麦克风帧处理锁或事件循环中运行。
冻结发布包还需由入口调用 `multiprocessing.freeze_support()`，并包含可选
运行时及其原生库；本文的运行验证针对源码启动，未验证所有发布包平台。

模型为上游 `sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20`，使用 chunk-8、
int8 encoder/joiner 与 fp32 decoder。脚本只提取这三个 ONNX 文件及
`tokens.txt`，校验完整下载包 SHA-256：

```text
68447f4fbc67e70eee3a93961f36e81e98f47aef73ce7e7ca00885c6cd3616a6
```

可用 `--archive <本地下载包>` 复用同一哈希的文件。脚本不下载或上传用户
音频，也不把模型权重加入仓库。`sherpa-onnx==1.13.8` 软件采用 Apache-2.0；
该模型压缩包内未附模型许可证，不能将运行库许可证当成权重再分发授权。
这里使用上游直接下载，不对权重再分发权限作结论。

“yui”和“悠宜”是同一个名字的两种写法，目标读音相同，不要求用户区分
英文版和中文版。词表由 `config/voice_wake_word.py` 提供。中文使用带声调的拼音音素
`y ōu y í`；名字 yui 的候选使用 `Y UW1 IY0`，不是逐个字母拼读。
同时接受 `y ōu y ú`、`l iú y ú` 两种近音，三种中文发音统一返回“悠宜”。
近似名字的发音触发是已接受的产品行为；无关对话和背景声仍需单独评估。
这些编码都存在于模型词表，但存在于词表不等于真实语音命中合格。
检测器配置 `max_active_paths` 默认设为 8，并显式传入 sherpa，扩大同时保留的
解码候选数。关键词阈值保持 0.25、关键词加分保持 1.0，不修改声纹阈值 0.40。
候选数 8 是待真人验收的改进，不是识别率保证，也没有加入第二阶段复核模型。

## 验证本地录音

准备 PCM16、16 kHz、单声道 WAV，以及以下 JSON 清单。`wav` 相对于
清单目录；未填写时使用 `<id>.wav`。只保存测试编号、预期结果及必要标注，
无需记录用户原始转写。

```json
[
  {"id": "name-alone", "expected": "wake", "keyword_end_seconds": 0.8},
  {"id": "name-and-command", "expected": "wake", "keyword_end_seconds": 0.7},
  {"id": "ordinary-conversation", "expected": "none"}
]
```

```powershell
uv run --no-sync --with $wakeRuntimeWheel python scripts/evaluate_wake_word.py --model-dir C:\NEKO-models\wake-word --manifest C:\NEKO-tests\manifest.json --output C:\NEKO-tests\report.json
```

合成语音必须加 `--synthetic`。脚本实际调用隔离推理进程，包含 WAV 中
真实静音，不自动补静音；结果包含命中、原始采样位置、送入多少音频时
命中、进程 CPU、采样 RSS 和每帧调用耗时。未标注词尾就不推算“说完后
延迟”。离线快速送帧不是实时麦克风延迟，也不测试 ASR/native 网络交付。
负样本应覆盖足够长的真实普通对话；几秒内零误触发不能证明小时误触发率低。

## 当前验证边界（2026-09-11）

- 7 个本地 Windows TTS 合成样本：5 个正例中命中 3 个，8.755 秒负例
  中 0 次命中。Huihui 的单喊名字及带停顿指令、Kangkang 单喊名字命中；
  Kangkang 连读指令和 Haruka 日语“ゆい”未命中。没有据此降低阈值或宣称
  “两个名字均已验收”。真人麦克风、口音、距离、噪声仍待验证。
- 这次离线运行准备约 0.78 秒，采样 worker RSS 约 95.8 MB；这些数字
  依赖机器和热缓存，不能当作性能保证。连续 600 秒无命中静音，离线处理
  约 9.90 秒，RSS 从第 10 秒约 96.6 MB 到第 600 秒约 96.9 MB。
- 后端每次只允许一个在途请求，默认每帧最多 1 秒音频；默认准备总预算
  30 秒，推理预算 2 秒，超时/取消关闭进程。清理使用一次 terminate 和
  最多两次 0.5 秒 join（必要时 kill），不等待推理自行返回。
- 新待机轮次、generation 切换、PCM 不连续都创建新解码流。命中被控制器
  拒绝后，下帧重新创建流；不拼接断裂音频。上游 1.13.8 在长静音后的
  内部重置会丢失累计帧偏移，且结果 `start_time` 仍为零；长待机时间映射
  需要底层 Reset 保留 `frame_offset` 的修复。修复版须通过下述长静音回归，
  不能仅凭 Python 接口或短样本测试宣称支持长时间待机。
  最后词元增加一个 40 ms 模型步长作为证据范围，仍非精确声学边界。
- 源码和长静音实测支持消费过的特征被释放；没有引入会截断跨界唤醒词的
  固定周期重置。模型内部内存不等于操作系统硬内存配额。

上游参考：[模型与词表说明](https://k2-fsa.github.io/sherpa/onnx/kws/pretrained_models/index.html)、
[1.13.8 特征消费实现](https://github.com/k2-fsa/sherpa-onnx/blob/v1.13.8/sherpa-onnx/csrc/features.cc)、
[关键词接口](https://github.com/k2-fsa/sherpa-onnx/blob/v1.13.8/sherpa-onnx/python/sherpa_onnx/keyword_spotter.py)。

长静音回归使用本地正例 WAV，不录制麦克风：

```powershell
$env:NEKO_WAKE_WORD_TEST_WAV = 'C:\NEKO-tests\name-and-command.wav'
uv run --no-sync --with $wakeRuntimeWheel pytest tests/unit/test_wake_word_reset_timing.py tests/unit/test_wake_word_model_smoke.py -q
```

用例在 WAV 前添加 12 / 45 秒真实静音，要求真实模型命中位置位于名字输入后、
控制器从完整缓存启动回放，且输出范围连续、PCM 字节与原输入一致。

## 现场诊断

启动前设置 `$env:NEKO_WAKE_WORD_DIAGNOSTICS = '1'` 可临时启用诊断。
worker 会报告就绪、每约 5 秒输入音频的帧数/流重建次数/音量统计和命中采样范围；
运行时会报告命中后的接受或拒绝原因，包括同状态内被原有状态日志省略的拒绝。
不保存 PCM，不输出关键词或转写文本。诊断输出失败不改变激活判定。
关闭该环境变量并重启后恢复默认关闭；此开关不调整任何识别阈值。

### Electron 候选数 8 验收

重启受影响服务并重新开启 Electron 麦克风后，检查当前 worker 的
`event=ready` 同行有 `max_active_paths=8`，以及当前麦克风/路由最新的
`Voice activation state=waiting`。同时应有持续的 `event=progress` 输入记录。
健康接口成功或设置中的 `effective_enabled=true` 只证明服务/设置状态，不能
代替当前采集会话处于 WAITING 的证据。

可在 PowerShell 对本次启动的 stdout/stderr 做一次只读筛选；日志路径由
实际启动记录填写，不要误读重启前的旧日志：

```powershell
$wakeLogPaths = @('<本次 stdout 日志绝对路径>', '<本次 stderr 日志绝对路径>')
Select-String -LiteralPath $wakeLogPaths -Pattern '\[WakeWord\]|Wake word decision|Voice activation state=' | Select-Object -Last 80
```

每轮都先确认待机，再分别测试单喊“悠宜”、名字后停顿、名字后直接连续
说指令。“yui”沿用同一名字读法。做两轮独立新录音/真人呼叫，并记录测试编号、
当前会话、预期词、是否被关键词接受及后续指令是否完整。两轮均为待验收，
现有两份录音及偏移/DSP 变体不是独立真人样本，不能换算为实际识别率。

- `event=hit` 只证明模型返回结果；`Wake word decision reason=wake_word_detected`
  才证明控制器接受该关键词入口。`owner_confirmed` 是声纹入口，不计入关键词
  命中或误触发。不要将 worker 与 Core 的同次事件重复计数，也不要逐行相加
  `progress` 中累计的 `hits`。日志筛选是观察入口，不是跨会话自动统计器。
- 负例覆盖无关普通对话、背景声音；已接受的名字近音属于正例。记录有效待机/验证监听时长和被接受的
  关键词误触发次数。误触发后 ACTIVE 的时间不计入监听分母；不足一小时可记录
  次数与时长，不将短时零命中描述为低小时误触发率。两轮真人与长负例均 pending。
- `max_inference_ms` 是 worker 每次解码调用的窗口最大耗时，`queued_bytes`
  是决策时的排队快照；二者不能证明全程无积压或给出端到端延迟。检查
  `wake_word_queue_overflow`、`wake_word_runtime_failed` 和 worker `event=failed`。
  完整积压曲线、进程 CPU/RSS 与超时分型需另行采样，不能从没有报错推算。
- `state=active reason=replay_handed_off` 说明本地回放队列按现有回执规则推进，
  不等于 Provider 已确认。真实首句验收需按同一会话/交付批次核对接收端音频
  范围、顺序及一次交付；转写和回复只能作体验观察，不能独自证明 PCM 完整。
- “说完名字到激活”的延迟需要词尾标注与激活提交位于可关联时间轴；目前日志
  的 worker UTC 时间和采样位置不足以直接相减。首个转写/回复的延迟另列，
  不用离线送帧时间或模型块长代替。没有采集这类证据时明确记 pending。
