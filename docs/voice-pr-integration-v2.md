# 语音 PR 整合记录

## 固定版本

| 项目 | SHA / 路径 |
| --- | --- |
| upstream/main | `9108054fdc9840a9be27526362b4920d31fd270d` |
| PR #3078 | `2ef746ca4289f2f653766d24da1f66f9a6fe5b24` |
| PR #3089 | `b3b56f5a6286fd30b3088bc4426652f5e70ca61a` |
| PR #3103 | `abe52cad7e18cf11188aba971eed72819dae66fd` |
| PR #3130 | `0e58bcc9ca9d8a270de644954b5a9ad750daaba8` |
| 旧整合分支 | `6099ce511` |
| Electron checkout | `C:/Users/ALEXGREENO/Desktop/CODE/N.E.K.O.-PC`, `8ef610d98e7ca833483dc97ce6b75fe97ad119a8`（有未提交改动） |
| 新整合 worktree | `C:/Users/ALEXGREENO/Desktop/CODE/N.E.K.O-voice-integration-v2` |

各 PR 对其 main 目标的共同祖先：#3078 为 `20dff4562`，#3089 为 `868ab5af4`，#3103/#3130 为 `d6f49ff5c`。#3103 的唤醒增量为 `5775d6411..abe52cad7`；此前提交包含旧版 #3078 内容，不重复移植。#3130 包含 `1309201f0` 和新增修复 `0e58bcc9c`。

## 运行环境与验收边界

本次所有源码、依赖安装和自动测试均从新整合 worktree 执行。Electron 是单独的 `N.E.K.O.-PC` checkout，其工作区现有改动不属于此整合提交。启动前 48911、48912、48915 端口均未监听；没有把旧后端的运行结果记为新分支实测。用户配置快照中 `independentAsrEnabled=false`、`noiseReductionEnabled=true`、`voiceInputResourceOptimizationEnabled=true`；声纹启用值未在此份偏好配置中找到，记为未确认。不记录密钥或会话内容。

网页和 Electron 真人麦克风、Provider 识别与回复须在同一新分支后端启动并确认健康实例 ID 后验收。#3103 的唤醒模型及修正版运行库也要分别校验版本和实际加载。

## 阶段检查点与冲突决策

1. `505fc122d`：最新 main 合入完整 #3078，自动合并，无文件冲突。原生 ASR 与激活相关 Python 794 项、前端 76 项通过。GitNexus 变更分析：194 文件、784 符号、12 条执行流程，高风险。
2. `e32fa9364`：移植 #3089 会话交接。其原始增量不包含 `asr_runtime.py`，整合后也未修改该文件。`lifecycle.py` 的热切换保留 #3078 激活票据、截止时间和 CAS 检查，同时将所有会话 close 改为 #3089 的归属关闭；`end_session` 改为绑定请求并由受管任务完成。`tts_runtime.py` 使用 #3089 的 worker 退休登记，保留 #3078 的游戏语音预载清理；会话清理保留请求图片账本裁剪。旧单体测试已经由 #3078 拆分，#3089 的断言迁到拆分后的对应测试。Python 1151 项、WebSocket 144 项、相关前端 330 项通过，交接故障变异 6/6 被断言检出。GitNexus 变更分析：48 文件、157 符号、5 条执行流程，中风险。
3. `00237aea1`：只移植 #3103 的唤醒增量 `5775d6411..abe52cad7`，保留 #3078 的新版基础实现。`asr_runtime.py` 只增加唤醒证据可接受的条件，不替换整个音频路由。合并 `uv.lock` 时保留 main 的 `pywinauto` 标记，`uv lock --check` 通过。唤醒专项 176 通过、3 跳过，语音回归 1168 通过。GitNexus：42 文件、56 符号、0 条执行流程，低风险。后续扩大回归发现 #3103 一个核心测试仍引用 #3078 已拆分的单体模块，在 #3130 检查点中修复为拆分后的夹具路径；该测试 5 通过、1 跳过。
4. `ef87803ab`：移植 #3130 的两次提交（含 `0e58bcc9c` 竞态修复）。`asr_runtime.py` 的同一通知尾部同时保留 #3078 的交付失败去重和 #3130 的 READY/FAILED 恢复状态；`app-websocket.js` 同时保留声纹激活状态处理和恢复事件。#3130 对旧单体 ASR 测试新增的三个断言已迁移至 `tests/unit/asr_runtime/test_provider_contract.py`。恢复前端 12 通过；当前语音后端相关回归 905 通过、10 跳过。GitNexus：8 文件、38 符号、5 条执行流程，中风险。

最终 `asr_runtime.py` 与 #3078 head 相比仅增加 85 行、删除 1 行（唤醒条件一行，加上 #3130 恢复逻辑），而非旧整合中大量删去 #3078 核心路径的状态。此差异证明新版保留了基础实现，但不代替真人识别闭环验证。

全量前端测试中 `api_key_secret_masking.test.cjs` 有 5 项失败，错误为 `_secretDisplayCache is not defined` 等；该测试与 `static/js/api_key_settings.js` 相比固定 main 均无变更，记为基线外问题。全仓 pytest 曾启动，但测试规模很大，约 4% 时停止；不能据此声称全仓通过。

## 运行与构建核对

本 worktree 的 `build_frontend.bat` 成功完成 PNGTuber/Live2D 资源解包、plugin-manager 和 react-neko-chat 构建。前端 Node 全量测试 351 项中 346 通过、5 项为上述固定 main 已存在的 API 密钥遮罩错误。

源码启动命令是 `uv run python -m app.memory_server`、`uv run python -m app.main_server`、`uv run python -m app.agent_server`（README 中旧的 `app/memory_server.py` 路径已不存在）。本轮三个服务运行于 48912、48911、48915，其 `/health` 均返回 `voice-v2-c14dae66e5774cf7895cf258d569385b`。从 `http://127.0.0.1:48911/static/app/app-audio-capture.js` 返回的字节 SHA-256 与本 worktree 文件一致：`91BE1D436AE4010D2F865F84CE6F8E548FE51989B09FBFB2E49CF8DA0EB816ED`。

Electron checkout 为上表所列的独立工作区，在该目录执行 `npm start` 已成功通过 Forge 启动、加载 `http://localhost:48911/`、创建 Pet 和 React Chat 窗口；没有复制或覆盖其现有未提交改动。其页面连接的是上述健康实例。尚未由真人在该窗口完成说话、Provider 识别、回复的闭环，因此不能将启动联通冒充语音实测。

`uv run python scripts/check_session_handoff_mutations.py` 复测 6/6 变异均被断言检出。第一次扩大 Python 回归的 125 个错误均发生在 Windows 默认 `%TEMP%/pytest-of-ALEXGREENO` 建立阶段；指定 `--basetemp=.pytest_tmp_pr3130` 后同组为 905 通过、10 跳过。

#3103 的固定哈希唤醒模型已由 `scripts/provision_wake_word_model.py` 下载并校验到本 worktree 的 `.codex-integration-run/wake-model`。当前 worktree 未安装要求的修正版 `sherpa-onnx 1.13.8+neko.kws2`，运行服务也未设置 `NEKO_WAKE_WORD_MODEL_DIR`；本机 `vswhere -latest` 没有找到 Visual Studio 安装，因此无法在此机执行仓库的 Windows 原生运行库构建脚本。模型推理与真人唤醒尚未验收；双开关关闭的原生路径不应依赖这些资源。不能用上游普通 `1.13.8` 代替修正版宣布唤醒通过。

`uv run python scripts/prepare_speaker_model.py` 已完成 CAM++ 固定资源校验；真实模型单测 2 通过、1 项因需要显式提供真实录音样本而跳过。声纹四段真人录入仍需实机完成。

## 依赖与回退

顺序为 `9108054fd` → 完整 #3078 → #3089 会话交接 → #3103 唤醒有效增量 → 最新 #3130 恢复。可分别回到上表的四个检查点；旧整合分支 `6099ce511` 保留为故障对照。回退或重启后先清理现有 48911/48912/48915 进程，再从目标检查点所在目录启动三服务，核对 `/health` 的共同实例 ID，并从同一目录重新执行 `build_frontend.bat`。不要用另一个工作区的静态资源或 Electron 内置后端代替目标检查点。

## 实机验收记录表

以下四行的“自动测试”覆盖路由选择、激活门控、静音恢复、连接回执和交付归属；“真人闭环”仍须在同一个上述健康实例下分别于网页和 Electron 完成。每行记录开始/暂停重开、连续两句、静音取消、断线重连，前后至少各一轮。快速启停和多窗口接管分别重复 10 次。

| 声纹 | 独立 ASR | 应走路径 | 自动测试 | 网页真人 | Electron 真人 |
| --- | --- | --- | --- | --- | --- |
| 关 | 关 | Core 原生 ASR；声纹/唤醒模型缺失也不拦截 | 相关回归通过 | 待测 | 待测 |
| 关 | 开 | 独立 ASR；不等声纹确认 | 相关回归通过 | 待测 | 待测 |
| 开 | 关 | 声纹确认后交 Core 原生 ASR | 相关回归通过 | 待测 | 待测 |
| 开 | 开 | 声纹确认后交独立 ASR | 相关回归通过 | 待测 | 待测 |

实机记录阶段按 `采集 → 后端接收 → 路由选择 → Provider 发送 → Provider 返回 → 识别文本 → 回复`，只记开关、路由、会话代次、错误码和音频计数，不保存原始音频或文本。声纹录入另记录四段录音、保存、重开后档案可用；唤醒另记录关键词、长静音后的首句完整交付及不重复播放。当前没有这些实机记录，不能据自动测试推定已经通过。

## 待完成验收

- 四种声纹/独立 ASR 开关组合的真人音频闭环；资源缺失旁路、快速启停、静音恢复、迟到回执、首次语音只交付一次已有相关自动测试，但仍需网页和 Electron 实机核对。
- #3103 修正版运行库的构建安装、已校验模型的实际加载、真人唤醒与长静音后首句完整性。
- 全仓 pytest 与覆盖率未完成；本记录不把相关专项测试当成全仓通过。
- 旧分支与新分支在同一真人输入下的故障复现对照未完成，因此当前不下最终根因结论。
