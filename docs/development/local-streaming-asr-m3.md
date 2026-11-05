# M3 本地流式 ASR 长稳验收

本阶段先验证源码运行中的持续对话、反复启停和模型热重载。它不等同于 Windows/Electron 安装包、真实麦克风、声学回声、真实 LLM/TTS、离线安装或整个 M3 发布验收完成。

## 固定环境与数据

沿用 M2 的 Windows 本机、宿主 Python 3.11、独立服务 Python 3.12、sherpa-onnx 2.0.1、Paraformer INT8 和单推理线程。语音来自公开 `TEST_MEETING_T0000000113.wav`，mono PCM16 / 16kHz / 3.0665 秒，SHA-256 `5489c8096e3f2708563aa23205b406eb3f3c7d845a74faabc189e9746852a90e`；按真实速率反复送入并保留自然断句所需的静音。

热切换的 A/B 具有不同可信 manifest，但共享同一固定权重。测试验证状态、资源和交付合同，不评估识别准确率提升。宿主真实执行 Silero、Smart Turn、生产 ASR provider、转写准入、字幕事件、同步队列和 Core handoff；远端生成、浏览器以及其他外部 sink 使用记录器。

每路测试使用自己持有的 supervisor、动态 loopback 端口、随机 token、catalog 和 state-dir，不共享另一条测试的服务。token、模型和音频不写入源码或 issue。`artifacts/m3/preflight.json` 固定生产源码集合、模型文件、音频 hash 和运行前的诊断门槛；测试新增文件不改变生产源码 hash。生产源码若修复，必须重新开始完整长稳，不能将跨版本片段拼接为一次通过。

## 运行前固定的检查

| 项目 | 本轮诊断门槛 |
| --- | --- |
| 持续时间 | 每路至少 1800 秒单调时钟真实运行，分列准备和清理时间 |
| 反复启停 | 至少 40 次完整正常循环；达到次数后仍运行至时长门槛 |
| 每条服务模型 worker | 不超过 2 个；退休未收敛时不能加载第三个 |
| 正式交付 | 每个已完成句只有一个 final、正式通知和对应 handoff；取消/失败句按其终态单列，不重放 |
| 连续会话 | 超过 64 句，验证安全边界上的 WebSocket 轮换；不要求全程同一个连接 |
| 音频 ownership | 有界队列/hold，已接受前缀不驱逐；取消与启停不污染后继语句 |
| 清理 | 无意外错误或自有进程孤儿；完整关闭后精确进程实例均退出 |
| 稳态 RSS 增长 | 对可比的单 worker 稳态样本，首末 5 分钟窗口增长不超过 `max(64 MiB, 基线的20%)`，并报告样本数与趋势 |

这些门槛是本轮预先固定的诊断标准，不是已承诺的最低硬件、长期内存上限或产品 SLA。双 worker 预热峰值与稳态内存分开；树 RSS 是进程工作集之和，可能重复计入共享页，不能当作独占物理内存。另记录 private bytes、CPU、句柄/线程和进程身份，并区分测试记录器的占用。

## 三路验证

1. 宿主连续对话：持续公开语音、真实 Smart Turn 自然 commit、字幕与正式交付；周期交替 reload，验证旧句完成、新句绑定新 worker，跨连接不重复。
2. 宿主启停：prepare、开启、语句、正常 capture_end、租约/续租/WS 销账、短测试 idle 后 native 退出；下一轮新 worker。交替资源优化配置，间歇完整 manager/supervisor 重启，穿插准备取消和逐句取消。短 idle 仅测试用，不改变生产 300 秒默认值。
3. 独立服务：实时流协议、逐句身份、消费序号、连续 reload、显式 rollback、取消和受控候选故障；验证 marker、排空和最多两个模型 worker。预期故障单独计数，不将其吞入成功识别。

记录器采用计数、有限环形缓冲或每句终态后清理，避免用无限保留的音频/逐帧事件制造假的应用泄漏。checkpoint 定期落盘，异常时保留失败阶段、已完成计数和清理结果。P50/P95/P99 必须附带起止定义、样本量与并行负载，不能把短样本重复结果当作多说话人质量评测。

## 结果

| 场景 | 实际时长 | 完整结果 |
| --- | --- | --- |
| 宿主连续对话+周期reload | 1800.297秒 active，pytest1806.19秒 | 274片段→258真实final/正式字幕/Core/sync，1408预览，20次reload，5条WS；12次明确旧句发布后仍送音频 |
| 启停/取消/idle/完整服务重启 | 1809.610秒 | 151完整轮、204实际final交付、38显式流cancel恢复、15次完整manager/supervisor重启；每轮lease/WS/queue销账 |
| 独立服务协议+reload/rollback | 1802.047秒，pytest1802.38秒 | 626opened=536唯一final+89取消无final+1注入故障；30次reload操作含24跨发布、16finalize、8显式rollback、6预发布取消；9次64句轮换 |

三路最终均通过各自固定门槛。主宿主PCM **57,383,536 bytes送出=确认消费**，只有1个静音warm successor在正常关闭时取消，其取消在途字节为0；主宿主最大native2，host音频queue/hold峰值24,448B（小于64,000B），detector采样队列峰值320ms。主宿主24个观测到的自有进程实例全部按PID/create_time及Windows handle确认退出。启停最终观测199个自有进程实例均退出；独立服务所有native handle退休、catalog任务收敛均true。取消/注入失败与正常final分开计数；不将取消PCM宣称全部消费。

| 范围 | 首→末稳态RSS MiB（增量） | private增量MiB | 样本/口径 |
| --- | --- | --- | --- |
| 主宿主进程 | 195.945→200.566 (+4.621) | +4.352 | 224/234 eligible样本 |
| 独立实际服务进程树 | 400.742→401.961 (+1.219) | +0.797 | 224/234 eligible样本 |
| 启停宿主测试进程 | 180.648→213.984 (+33.336) | 32.352 | 21/25轮后样本，含pytest日志/记录开销 |
| 协议run2 native（旁证） | 325.238→324.660 (-0.578) | -0.742 | 255/255，调用端含HTTP历史保留 |
| 协议run2调用端+测试supervisor | 63.117→82.668 (19.551) | 18.957 | 255/255，调用端含HTTP历史保留 |

独立实际服务树采样RSS峰值 **744.066MiB**，private峰值 **2145.668MiB（约2.095GiB）**；主宿主自身RSS峰值200.922MiB。各自峰值不相加冒称同一瞬间峰值。主连续测试CPU在1795.296秒checkpoint跨度平均 **28.29%单核等价**（一核100%，不是整机百分比）。这些数值包含验收程序，且测试并行；不外推低配设备。

| 宿主观测区间 | n | P50 ms | P95 ms | P99 ms |
| --- | --- | --- | --- | --- |
| 首个PCM写出→首个非空partial | 258 | 1219.0 | 1250.0 | 1312.0 |
| Smart Turn commit→wire final | 258 | 63.0 | 94.0 | 125.0 |
| wire final→Core handoff | 258 | 0.0 | 16.0 | 47.0 |
| reload请求→观察到发布 | 20 | 2484.0 | 2984.0 | 3813.0 |
| 启停prepare（缓存权重/新进程） | 151 | 1890.0 | 2797.0 | 未记录 |
| prepare后runtime启动 | 151 | 0.0 | 93.0 | 未记录 |
| 短idle到stopped | 151 | 468.0 | 500.0 | 未记录 |

Windows Python3.11短间隔计时约16ms量化，观测0ms不表示零延迟。首partial并非人工标注的起声指标，不能据此宣称初始800ms目标通过。

详细验证限制见 `artifacts/m3-issue-result.md`：协议调用端HTTP历史、启停pytest日志开销、一次旧cycle50超时诊断缺失均保留。当前数据不证明长期无限运行、真实设备/LLM/TTS或安装包验收。

## 复现

先按原文档设置宿主四个托管路径，并安装已有Silero/Smart Turn资产。两个仓库使用各自Python环境：宿主3.11、服务3.12，避免把宿主的UV_PROJECT_ENVIRONMENT带入服务。示例（每次用新的输出目录；完整测试需要真实等待30分钟）：

```powershell
# 宿主：连续对话/热重载
$env:NEKO_ASR_M3_SOAK = '1'
$env:NEKO_ASR_M3_SECONDS = '1800'
$env:NEKO_ASR_M3_REPORT_DIR = 'C:\results\asr-m3\host-new'
uv run --no-sync python -m pytest tests/integration/test_local_streaming_asr_soak.py -q -s -p no:logging --no-cov --basetemp artifacts/pytest-m3-host-new

# 宿主：反复启停（达到40轮后继续至1800秒）
$env:NEKO_ASR_REAL_CYCLES = '1'
$env:NEKO_ASR_CYCLE_COUNT = '40'
$env:NEKO_ASR_CYCLE_SECONDS = '1800'
$env:NEKO_ASR_CYCLES_REPORT = 'C:\results\asr-m3\cycles-new.json'
uv run --no-sync python -m pytest tests/integration/test_local_streaming_asr_cycles.py -q -s -p no:logging --no-cov --basetemp artifacts/pytest-m3-cycles-new

# 服务仓库自己的3.12终端（MODEL_DIR/WAV指向原公共模型/录音）
$env:NEKO_ASR_TEST_MODEL_DIR = 'C:\models\paraformer'
$env:NEKO_ASR_TEST_WAV = 'C:\samples\public-speech.wav'
$env:NEKO_ASR_SOAK = '1'
$env:NEKO_ASR_SOAK_SECONDS = '1800'
$env:NEKO_ASR_SOAK_REPORT_DIR = 'C:\results\asr-m3\protocol-new'
uv run --no-sync python -m pytest tests/test_soak_real.py -q --no-cov --basetemp artifacts/pytest-m3-protocol-new
```

当前service脚本包含后续bounded HTTP观察器改进，完整run2的精确旧源副本另存；本评论不将15秒改进脚本pilot冒称30分钟验证。独立same-Core功能补充用`NEKO_ASR_SAME_CORE=1`运行`tests/integration/test_local_streaming_asr_same_core.py`。

最终路径：宿主`artifacts/m3/{aggregate.json,preflight.json,source-verification.json,memory-timeline.png,same-core.json}`、`host-30min/{report.json,junit.xml,summary.json,source-end-verification.json}`、`cycles-semantic-30min.json`、`cycles-semantic-30min-junit.xml`、`cycles-semantic-30min.stdout`、`cycles-semantic-30min.stderr`；服务`artifacts/m3-protocol-1800-run2/{report.json,samples.jsonl,resource-analysis.json,pytest-result.json,harness-run2.py,harness-provenance.json,source-freeze-final.json}`，另有observer复现与旧run限制记录。
