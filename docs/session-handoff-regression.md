# 会话交接修复回归与交付记录

工作树：`E:/Work/CODE/N.E.K.O-session-handoff-lifecycle`。分支：`codex/session-handoff-lifecycle`。基准为上游 main 的 `868ab5af43b01fc710a3b7009e0ec9c794683980`，日期 2026-09-09。原工作树的未提交改动未合入。本修复仍需实机验收，不应据自动回归结果宣称已完成上线验收。

## 改动、必要性与前后行为

旧实现把“当前已经 inactive”当成交接完成，后台结束、输出回调和启动 finally 仍可能访问随后安装的连接、TTS 队列、输入缓存及通知状态。修复建立启动操作、LLM 连接和 TTS runtime 三层身份，并由 manager 持有退休记录，分别跟踪交接安全与真实资源退出。结束请求在接受时捕获目标，重复请求复用该目标的清理任务；锁内只处理归属，网络关闭及线程等待在锁外。

新启动共用 15 秒预算，覆盖交接、记忆隔离、容量、连接及 TTS / ASR 就绪。超时不会伪装释放仍存活的资源。主 LLM 和主 TTS 默认各最多两个资源；连接中、预热及未退出的退休资源均占容量，provider 能力声明可收紧为串行。fallback 使用同一主 TTS 容量限制。正常健康 TTS 复用及独立游戏预加载继续保留。

成功确认在输入锁、上下文和 ready 完成之后发送。缓存输入回放归已安装连接持有，长文本回复不会延长启动预算；实时输入排在预约回放之后。取消只恢复尚未派发的后缀，已经提交或提交结果不明的输入不会盲目重放。旧启动失败和 finally 不再修改后继启动槽。

文本、音频、SID 轮换、TTS 通知和输入回放均参加归属检查及在途任务收尾。嵌套回调不能提前移除外层回放任务的登记。热切换关闭也排空旧输出；provider 内部同对象 close/reconnect 保留原语义。TTS header / 二进制帧按同一帧收尾，阻塞的队列消费者确实退出后才清队列，避免退出哨兵被提前取走而留下 executor 线程。

准备、成功、失败通知携带请求关联，并保留发起窗口、麦克风租约窗口和观察窗口的不同语义。暂停麦克风不解绑聊天 WebSocket。普通记忆结束保持“旧输出 → session end → 新输出”的队列顺序且不等待落盘；有隔离回调时保留独立屏障，超时回调不冒充实际结算完成。

## 基线与行为回归

P0 在生产修改前保存原始文件并运行断言正确行为的测试：后端新增回归 **8 failed / 3 passed**，既有健康对照 **35 passed**。失败涵盖慢关闭占据当前槽、锁等待后误关后继、调用者取消丢失清理责任、启动取消后重新激活、旧 finally 清除新启动计数、快速重开以及两种旧输出越过记忆边界。原始输出保存在 `output/session-handoff-lifecycle/p0-behavior.txt` 和 `p0-baseline.md`。

测试使用真实 manager、锁、队列、输入处理和生命周期逻辑，只替换配置请求、记忆外部服务及 provider；事件屏障固定竞态顺序。TTS 使用实际线程和队列，断言线程退出、队列内容和 socket 写入。进程必须正常退出，单个测试显示 PASS 后仍挂在线程收尾中不算通过。

新增两个归属模块的最终覆盖率测试 **153 passed**：`session_lifecycle.py` **97%**，`tts_lifecycle.py` **90%**，合计 **95.21%**。这是这两个模块的行覆盖率，不代表整个仓库达到相同比例。详细记录为 `output/session-handoff-lifecycle/coverage-final.txt` 和 `coverage.xml`。

此前已完成的 47 文件扩大核心回归为 **1498 passed / 0 failed**，53.03 秒，进程正常退出；5 条 warning 为既有 websockets / FastAPI 弃用提示和 pytest 标记提示。包含 ASR 交接、预热热切换、游戏记忆路由、TTS fallback、语言和麦克风租约契约；日志为 `output/session-handoff-lifecycle/core-regression-final.txt`。按用户最新要求，后续验证只运行改动命中的定向用例，不再扩大或重复整组回归。

本次主要边界的定向重放示例：

```powershell
$env:UV_PROJECT_ENVIRONMENT='E:/Work/CODE/N.E.K.O/.venv'
uv run --no-sync python -m pytest tests/unit/test_session_handoff_lifecycle.py tests/unit/test_session_handoff_startup.py tests/unit/test_session_handoff_boundaries.py tests/unit/test_tts_handoff_ownership.py tests/unit/test_session_start_input_flush.py tests/unit/test_session_notification_handoff.py -q
```

## 移除保护的反证

`tests/session_handoff_mutations.py` 是显式启用的 pytest 插件，仅在内存替换当前函数，测试结束恢复。通过 `NEKO_HANDOFF_MUTATION` 选择容量、记忆屏障、旧 finally、旧输出、清理 shield、旧槽退出、交接期 PCM、ready 输入、已提交输入、实时输入顺序及嵌套回调保护；重放测试名及失败断言见 `output/session-handoff-lifecycle/mutation-report.md`。变异应用失败不算护栏被验证。

TTS 与热切换反证可执行 `uv run --no-sync python scripts/check_session_handoff_mutations.py`。脚本核对导入的是本工作树，每项必须触发行为断言失败，导入错误、语法错误和无关异常均不算通过。前端三个反证已经包含在 `tests/unit/test_session_notification_handoff.py` 的正式参数化测试中。

最终反证结果分别为：生命周期 **12 类 / 17 参数案例**触发预期行为失败，TTS / 热切换 **6 / 6** 由断言杀死变异，前端 **3 项**通过反证检查。恢复原保护后相关正式回归均通过。额外使用 ack 尚未返回的场景移除实时输入顺序保护时，变异曾存活，因为启动计数仍在保护输入；此结果未计为有效反证，改用启动槽已释放但回放尚未执行的原场景后得到预期断言失败。

## 风险、实机记录及性能

本次改动集中于核心生命周期，仍需主开发者 review；重点检查取消传播、输出在途任务、启动成功时序、输入回放、预热热切换和游戏完成回调。调用图已确认 `_close_detached_pending_session` 为 HIGH 风险，并检查其清理、预热、启动、最终切换及 ASR 交接调用。动态 mixin 调用不能仅凭图中的零调用者判断安全。

最终 GitNexus 变更映射覆盖 **37 文件 / 360 符号**，风险 **HIGH**，列出 6 条受影响流程；CLI 正常退出，完整结构化结果未标记 `partial` 或 `truncated`，见 `output/session-handoff-lifecycle/graph-changes.json`。索引构建本身仍报告流程枚举及动态调用上限，刷新重跑后仍存在，因此这 6 条不是影响范围的完整枚举，也不能把未列出的后端调用当成不受影响。已结合实际调用点和上述行为回归核对，未将此报告表述为全部调用图覆盖完成。

浏览器和 Electron 的正式实机样本目前均为 **0**。此前仅启动隔离记忆服务和采样后端，两者处于存储 `selection_required` 初始化状态，没有实际会话；随后已停止并释放 48911 / 48912。`resources.jsonl` 中空 manager 的基础进程采样不是生命周期验收，也不是性能对比。

开麦到确认、TTS ready、首段有效音频的实机基线和修复后数据均未取得，不承诺加速。操作及采样命令见 `docs/session-handoff-acceptance.md`：网页和 Electron 各做空闲 / 正在说话两组，每组 0、50、100、300、1000 毫秒各十次，至少 200 个有效样本；provider 故障、跳过和无音频的样本独立标记。需要审核正常资源回落、异常资源有界、无误关误清及过期通知覆盖。

## 整体回退

代码、回归及通知协议作为一个修复单元交付。尚未提交、推送或创建 PR。当前可以停止修复工作树的服务，回到原工作树运行；验收使用的隔离数据目录不会替换原始记忆。后续若提交，应整体 revert 该修复提交并重启后端、重新加载前端，不拆分部署 P1–P3。
