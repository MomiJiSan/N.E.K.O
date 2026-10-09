# 多人语音拦截：实施交付与验收边界

本交付是 [PR #2980](https://github.com/Project-N-E-K-O/N.E.K.O/pull/2980) 的堆叠增量，
基础版本为 `f9a0f36dbab96b26b76b464ce2ce86dc8f8c598a`，分支为
`codex/voice-interception-implementation`。工程修复、候选身份、模型研究、应用验收分别结算。
#2980 提供已有门控、应用注册表、TSE/PVAD 实现与离线校准工具；本层不重复交付这些基础。
目标仓库的 `codex/stack-pr-2980` 固定到该基础版本，供增量审查；#2980 合并后再将本 PR 改为 main。
基础版本后续更新时必须显式同步堆叠分支并重新验证，不能把旧 head 的结果用于新版本。
本交付按研究阶段 Draft PR 审查，生产工厂未启用。
工程通过不能代替声学与设备验收。

## 处理合同

适用范围是会话激活后的持续音频，包括首次激活回放的共同输出。
Silero 提供语音活动证据；TSE 或通用分离提供匿名候选；声纹判断候选身份。
只有唯一明确本人候选的已确认区间能够交给独立 ASR。
评分 PCM、交付 PCM、原始区间和会话／模型／参考版本必须一致。

首版允许零／单／双候选。零匹配、全旁人、不确定、多匹配均形成缺口，
候选编号不能当成固定人物。本人连续确认在缺口处重置。
失败、超时和旧身份失效不能回退原始混音，也不能通过原混音低分硬性拒绝有用候选。
候选身份通过不能证明没有旁人内容残留；残留仍需人工标注的声学验收。

## 交付一：区间与生命周期

`interception_runtime.py`、`prewire_gate/` 负责能力校验、评分提交、区间终结、
顺序交付和物理退休。新增 `ScorerCapabilities` 声明采样率、最低长度、
支持长度与模型版本，模型能力和校准布局必须同时支持当前配置。
确认缓存至少容纳 `window + (streak - 1) * step`，部署还需测量推理等待和抖动。
禁止补零、借其他句子的身份、降低全局声纹阈值来绕过长度不匹配。

已完成 UNCERTAIN 使用显式 `gap_finalized` 终结，不伪造 `event_ended`，
未完成评分仍按提交时的绝对期限等待。评分错误直接报告并结束本次运行。
正常 finish 共享总期限，正确处理 Plan → claim → End；不支持的短尾形成缺口，
已确认且已交付的记录保留。硬停先撤销产出权限，旧结果不能复活。

异步任务取消不等于本地 native 评分线程退出。运行时跟踪实际评分任务，
确认旧资源物理退出后才允许替换工厂；清理任务保持本次 owner。
三个历史实现变异反证分别捕获 UNCERTAIN 堵塞、吞评分错误和错误收尾。

## 交付二：候选身份

`candidate_identity.py`、`candidate_source.py` 和
`prewire_gate/candidate_contracts.py` 提供有界候选评分与不可变数据合同。
单路 source 接收实际 TSE chunks；双路 source 提供与 spec 精确绑定的 batch。
真实 runtime／gate／scheduler 测试证明更换候选 PCM 会改变判断，
编号交换会重新识别，并且候选模式没有原始混音二次授权。

`CandidateCalibrationContract` 可以严格持久化评分模型、分离模型、参考、
前处理、配置、参数／校准摘要、支持长度与决策布局。
它不拟合阈值、不注册权重，也不证明效果达标。
旧 raw／terminal 校准不能重新标记成候选校准。

当前没有已注册的候选校准发布包。
`PrewireInterceptionFactory.for_production()` 完成已有前置校验后明确拒绝
`candidate_calibration_release_unavailable`；研究构造函数仅用于显式依赖注入。
应用 registry 默认 factory 为 None，未开启功能不创建模型或恢复 watchdog。

单路旧 `TARGET_ABSENT` 缺少原始范围，当前安全停流；不能猜测它对应哪个区间。
精确 `CandidateBatch(spec, candidates=())` 可以结算缺口并继续。
双路匿名分离的合同已验证，实际通用分离模型的流式部署尚未验收。
新工作树尚未装配真实 TFMap 流式 worker 与获准发布的候选评分／校准适配器。
研究接口和受控候选链已贯通；外部 TFMap 实验不等于该 worker 已进入应用。

## 交付三：模型与评价

`scripts/voice_interception_evaluation.py` 提供 dataset、freeze、evaluate、receiver、
resources 命令。它检查人物／会话／原始录音来源划分、冻结合同、人工标注、
覆盖场景、本人保留和旁人误入，以及真正接收端的 PCM／区间／摘要。
全拒声音不能判胜，writer 承接不能冒充 receiver_read 或 provider 确认。
资源采样记录指定进程树；无法查询和进程失踪不能当作零资源消耗。

本机 RTX 4080 SUPER 实际执行 TFMap mask-head 16 步微调，并完成七组
before／after 的十四路本地 ASR 解码。训练源和原始检查点未改动。
这是已知人物诊断，不是未见人物验收；无人工词标签不能计算可信词误入率。

| 诊断 | 基线 → 微调 | 结论 |
|---|---|---|
| 长重叠本人 SI-SDR | 7.83 → 5.92 dB | 本人损伤增加 |
| 同场景旁人相关增益 | −15.58 → −20.04 dB | 更压旁人，不足以抵消本人损伤 |
| 本人独说 SI-SDR | 25.17 → 17.30 dB | 独说也退化 |
| 六个含本人场景 | 六个全部 SI-SDR 下降 | 拒绝该训练候选 |

该实验权重仅保留在外部研究产物，不进入产品资源或注册列表。
后续仍须用独立人物、会话及原始录音划分校准／验收数据，
在调参前冻结本人漏词、旁人误入和不可评分门槛，比较 TFMap 与通用分离路线。

## 交付四：共同出口与接收证据

Core 的 activation output 使用同一个 bridge，首次回放和持续 ACTIVE 输出都经此出口。
交付 sidecar 随 retained source spans 经过独立 ASR、dispatcher、session 和 worker。
交付事件定义在中立的 `voice_turn/interception_events.py`；旧 voice_input 路径仅兼容导出。
ASR 底层不导入上层 voice_input。核心守门仅准入两个明确的音频边界方法，
新增真实源码与反向依赖反例，其他公共方法和上层依赖仍被拒绝。
LOCAL_ACCEPTED、QUEUED、TRANSPORT_WRITTEN、TRANSPORT_OWNED、UNKNOWN 分别保留。
UNKNOWN 不能重投；本地写成功不生成 provider 确认。

缺口／End 使用独立 ASR manual／SmartTurn 的明确封口能力，保留已接受前段 final。
封口期间的新本人音频暂存为后继；下一次缺口共用原截止时间等待真实前段 final
和后继 writer 接管，再封住后继。ACTIVE 状态本身不是 writer 已准备好的证据。
超时、取消和换身份不能封住新流。两个历史问题都有实际反证。

native、Provider VAD、缺少精确源贡献映射的 OpenAI 重采样路线没有准入。
当前能力拒绝发生在明确边界或 tagged resampling 处，不能宣称这些路线可用。
本轮没有应用级生产路由装配；在未来生产装配中必须先检查能力再接受首帧。

真实 TCP 接收端测试贯通候选 selector → runtime → Core → IndependentAsrRuntime.submit
→ dispatcher → RealtimeSession → worker socket → receiver read，未 mock submit/stream_audio。
本人正例接收 800 samples，纯旁人接收零 samples，独立摘要审计通过。
算法及 provider final 是受控研究 fixture；它证明交付合同，不证明真实 ASR 准确率。
模型微调的本地真实解码结果是另一项证据，尚未与完整设备链联合验收。

## 验证与仍未通过的条件

本次堆叠组合重新验证为 4089 passed、18 skipped（主批 3013，补充 voice_turn／voice_identity／
根目录 ASR 测试 1076），总体覆盖率 85.81%，统计 106 个源码文件，完整保留原 95 文件并
纳入继承的 TSE/PVAD／进程启动和校准工具；保持 80% 门槛。两批模块来源审计均无越界源码。
最初主批覆盖率 75.37% 未达标；补跑现有遗漏测试后追加统计，没有缩减分母。
语音相关 Node 213 项和 Electron 41.2.0 六个受控采集场景通过，15 项静态守门及 Ruff 通过。
额外前端 glob 为 485 passed、9 failed：三项缺少本地 jsdom，另六项在基础提交也复现失败。
四个相关 workflow 已允许 `codex/stack-pr-*` 作为 PR base，保留 main 和原门槛。
当前图检查覆盖 63 个变更路径、806 个符号，16 个完整批次，没有 partial／truncated。
临时隔离启动器缺少 Windows spawn 入口保护造成的两轮阻塞已修正，不计作通过证据。

原独立提交执行过扩大 Python 回归、前端 Node 回归和 Electron 41.2.0 六场景。
原记录为 3522 passed、14 skipped，耗时 290.11 s；该记录不是本次堆叠提交的验收结果。
原八个声明范围的 93 个源码文件完整保留，新增中立交付事件与核心守门脚本后
统计 95 个源码文件，总体覆盖率 85.30%，保持原 80% 门槛。
变更可执行行另计为 87.69%，没有替代总体分母。
其中 `core.asr_runtime` 全模块覆盖率为 77.94%，不能宣称每个模块都超过 80%。
前端 Node 回归为 139 passed。
覆盖率工具的 dotted source 声明会预导入应用包并干扰测试状态；
共享依赖环境曾带入旧研究路径；原审计只覆盖 app／main_logic，遗漏了 scripts。
后续发现校准测试从旧工作树加载了未提交脚本，原独立提交的 CI 因此收集失败。
堆叠后该脚本由 #2980 提供，但仍须排除旧源码搜索路径，重新验证本层与基础层组合。
覆盖率统计保持原文件集合和 80% 门槛；旧报告不能代替本次组合验证。
Ruff、分层、核心合同、异步阻塞和启动导入等静态守门均通过；
原独立提交还完成过最新 main 的存储测试 180 passed、2 skipped；本层不带入无关 main 更新。
Electron 使用实际 getUserMedia／AudioWorklet 与 fake device，覆盖 `/`、`/chat`、
正常 stop 的 capture_end 在撤权／pause 前发送、硬停不 flush、设备释放与采样率。
它不是用户真实麦克风、全应用服务或最低机器测试。

仍缺未参加调参的多人房间录音、人工本人／旁人词和区间标签、候选发布校准包，
以及最低目标笔记本配置。需要在该设备完成 30 分钟连续场景及两小时稳定性测试，
对齐基线测量说完到首次可听回复的新增等待（目标 300 ms，上限 500 ms）、
相关进程树新增内存（250 MB）、冷启动、首次身份证据等待与持续处理成本。
本机 GPU、每块推理耗时或单进程内存均不能替代这些条件。

核心模块 review 重点是跨层生命周期、默认关闭、首次回放、封口期间输入、
迟到结果、部分写入、未知交付不重投及同类 provider 路径。
本轮改动应分工程、候选、评价、接收与应用边界分别审查；
生产启用与模型替换需独立效果／性能验收，不随工程测试通过自动放行。

修改前已逐符号分析调用影响；共同出口和工厂入口涉及 HIGH／CRITICAL 风险。
原独立提交的图检查覆盖 78 个变更路径、1770 个变更符号。
堆叠增量另按 #2980 基础版本检查，并以当前实际 diff 复核文件和符号范围。
执行流索引仍有深度与数量上限；“零受影响流程”不能解释成没有风险，
因此还结合入口／消费者阅读、真实交付回归与历史故障反证。

集成检查、模型实验和接收审计的原始产物保留在本地，不随仓库发布，亦不是运行或部署依赖。
