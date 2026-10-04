# #3149 剩余改动范围

本 PR 基于 main `9cdc4dfce7ae3203673ad4846a02ae1c7fbe8ad0`。当前差异只包含尚未由主线提供的行为和配套回归；来源 PR 的功能不再作为本 PR 的新增内容。

## 主线已提供的能力

| 主线能力 | 已合入来源 | 本 PR 的使用边界 |
| --- | --- | --- |
| 声纹激活、off/enforce 门控、录入生命周期、采集预检与资源诊断 | #3078、#3172、#3220、#3282 | 复用既有入口、上传协议、预检脚本及资源归属 |
| 会话交接、迟到回调退休、取消收尾、final 预留与背压、静音恢复 | #3089、#3130、#3201、#3219 | 保留主线生命周期与单一预留账本 |
| 近音唤醒、首轮句首纠错及歧义词整句匹配 | #3103、#3165、#3222 | 不重新带入词表或旧目录实现 |
| Qwen 默认 provider 判停、本地活动观察及 finish/reconnect fallback | #3221 | 默认路由与供应商端点权属沿用主线 |

`docs/design/asr-client-phase1.md` 与 `tests/README.md` 使用主线版本。历史 Qwen 默认 manual 的说明不属于当前实现。

## 本 PR 仍需审查的差异

| 剩余行为 | 主要实现 | 与主线的差别及风险 |
| --- | --- | --- |
| 候选准入与转写证据 | `voice_turn/admission.py`、`transcript_admission.py`、`asr_client/endpointing/admission_gate.py`，以及 ASR／Core 输入衔接 | 准入前不准备或上传新回合；证据不足的孤立填充词在回复副作用前结束交付。字段可缺省，既有无证据路径兼容；需审短指令误拒及重叠回合身份 |
| 采样归属与输入交接 | `asr_client/audio_ranges.py`、runtime、detector runtime | 固定候选采样范围、有序交接重采样尾音及跨 await 输入；拒绝的前驱不混入后继。需审迟到检测、容量淘汰和取消 |
| 故障恢复、连接退休与显式结果保留收尾 | `asr_client/recovery.py`、`connection_cleanup.py`、共享 Session 和 Qwen／Step／OpenAI worker | 可信读断连或 final 超时使用有限替代连接预算；旧音频不重传。已接受但尚未入队的 final 阻塞 idle；显式 `finish_and_drain()` 等待供应商确认及 final 交付，普通 close 保持取消语义 |
| 声纹验证结果及激活状态衔接 | `voice_identity_runtime.py`、voice identity service、页面脚本／模板／样式 | 显示验证结果卡片及匹配分数；启动中的 blocked 路由可重试，未完成 DSP／激活转换保持重试资格。主线预检与录入生命周期继续使用 |
| 采集约束、诊断和恢复提示 | `microphone-input.js`、音频捕获／WebSocket 消费、`audio_processor.py`、激活诊断及八语言文案 | 48k 正式录音和声纹预检／录入关闭浏览器 AGC，使用后端 AGC；16k 正式采集绕过后端 DSP，保留浏览器 AGC。设置试麦遵循正式采集策略，采样率在开麦前固定，设备回退沿用。后端 AGC 按块时长换算。恢复状态绑定归属，READY 不开麦；诊断不记录原始音频或转写 |

上述实现共同约束“采集 → 准入 → ASR → final → Core → 页面反馈”。身份、已接受结果和恢复通知需要在同一整合树验证。当前差异为 106 个文件；排除新增文件、语言包及测试文件后，改动 31 个既有文件，仍是大 PR。请重点 review 跨层身份、取消／退休顺序及未启用路径。

详细契约分别见 [候选准入](voice-admission-integration.md)、[短语音实验](short-voice-admission.md)、[采样交接](design/asr-sample-handoff.md)、[故障恢复](design/independent-asr-fault-recovery.md) 和 [输入诊断](voice-input-diagnostics.md)。`benchmarks/` 与 `records/` 中的日期文档是当时基线的实测记录，不代表当前整合验收。

## 验证与边界

以 `6366f2279` 为修复基线，对照 main 确认两处前端回归：声纹 runtime off 时仍会启动恢复轮询；16k 正式采集关闭浏览器 AGC 后没有后端增益补偿。本轮在轮询入口和等待后检查运行状态，并让正式采集／设置试麦按固定的目标采样率选择 AGC。设备回退、Worklet 重采样与 NEKO 报头使用同一快照；声纹预检／录入仍使用原 48k 契约。

本轮 Node 回归 357 passed，Python 设置契约及音频处理生命周期回归 50 passed。新增 8 个用例覆盖关闭模式、等待中关闭／关窗、正常恢复，以及采集参数／报头和设备回退。两处生产 JavaScript 的新增行 V8 覆盖为 22／22，不代表整个采集模块覆盖率。移除轮询入口检查、等待后检查或采样率快照的三种变异均被测试检出。以下较大回归报告属于此前后端运行时基线，本轮没有修改 Python 运行时代码，也没有重新执行全量套件。

`44344a237` 的 CI 串行分片暴露了遗漏同步的静态生命周期测试：它仍以旧的四参数签名提取 Worklet 函数体，并要求旧的调用形状。本次同步五参数签名及采样率快照传递，保留取消、选择归属及提交顺序断言；该文件 7 passed，完整串行分片 158 passed。移除入口／提交取消门控或漏传采样率快照的三种变异均被测试检出；Ruff 通过，生产运行时代码未再修改。

此前验证版本 `22a8e7832` 的语音及主线相关回归为 4561 passed／6 skipped，串行回归 158 passed，Node 回归 190 passed。ASR runtime、transcript dispatcher、detector runtime、Qwen worker、声纹 service、voice-input registry 和唤醒转写七个模块合计语句覆盖率 81.12%；Ruff、Core 契约、模块分层、异步阻塞和八语言同步检查通过。本轮重跑上述静态守卫及修改的 Python 测试文件 Ruff，通过。

全量非串行回归为 31272 passed／187 skipped／21 failed。同环境 main `9cdc4dfce` 对照为 30856 passed／187 skipped／22 failed，复现上述全部 21 项失败：bilibili_api 导入 15 项、Windows 符号链接权限 2 项、Mount 路由测试 1 项、#3078 历史审计 1 项，以及存储与 takeover 日志捕获各 1 项。main 另有一项 ASR adapter 用例失败，本 PR 不声明修复；不宣称全量通过。

本轮重跑 Electron 41.2.0，使用真实声纹页面与 AudioWorklet、模拟音频输入及本地 API，验证设备回退重试、试录释放与重新采集、取消后的迟到授权释放、正式录入重开和服务端契约检查、增益修改失效及实际采集 AGC 参数。PCM 传输为 288000 字节。新增 Chromium 正式采集验收进入 CI：实跑 48k、16k 和授权期间平台判定变化／设备回退，核对实际轨道 AGC、真实 Worklet PCM 报头及资源释放；真实麦克风和云端 provider 未参与。

独立 ASR 准入实验默认关闭，策略在会话边界固定；partial 不升级为 final，关麦和接管优先于恢复。#2980 的 ASR 前旁人音频拦截研发、模型阈值调整、声纹算法、provider fallback 顺序及原生唤醒库构建产物在本 PR 范围之外。旧声纹档案与后端 AGC 修正后的实际匹配分数尚无真实录音对照，本轮保留档案和原版本契约；不根据合成音频或模拟分数强制重录。真实轻声／短指令、实际 provider、Electron 多窗口及游戏高负载仍需专项验收。PR 保持 Draft。
