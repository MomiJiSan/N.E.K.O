# Windows 本地 ASR 离线部署验收

本轮验收统一跟踪于 [ASR 主线 #1](https://github.com/MomiJiSan/N.E.K.O-ASR/issues/1)。
离线范围是本地 ASR 推理、服务管理和桌面安装运行；Qwen/CosyVoice 等云端生成或语音服务仍需网络。本轮不使用真实麦克风、摄像头、屏幕采集或实体播放设备。

## 冻结服务的宿主配置

宿主保留 Python 模块启动和外部 loopback 服务两种原有模式，新增独立 Windows 可执行文件模式：

```powershell
$env:NEKO_ASR_SERVICE_EXE = 'C:\NEKO\asr\neko-asr.exe'
$env:NEKO_ASR_SERVICE_ROOT = 'C:\NEKO\asr'
$env:NEKO_ASR_MODEL_DIR = 'C:\NEKO\models\paraformer'
$env:NEKO_ASR_SMOKE_WAV = 'C:\NEKO\models\public-speech.wav'
```

上述路径须为绝对路径；使用 EXE 时不能同时设置 `NEKO_ASR_SERVICE_PYTHON`。部署须复制整个服务 onedir，不能只复制 exe；模型权重单独分发。服务依然由现有 manager 执行 prepare、租约、闲置卸载、实例鉴权和 Windows Job 回收。子进程不继承 `PYTHONPATH`、`PYTHONHOME`。

候选模型仍使用成对的 `NEKO_ASR_CATALOG` 与 `NEKO_ASR_STATE_DIR`，沿用既有可信 catalog、不可变 stage 和逐句切换，不在正在映射的目录覆盖模型。微调只需输出同一运行时契约的受信 bundle；本轮没有训练或导出模型。

## 桌面资源布局

本轮 PC 集成保存在验收源码快照与补丁中，未覆盖原 PC 工作区。打包态 Windows 从安装后的实际 `resources` 定位以下文件：

```text
resources/bin/projectneko_server.exe
resources/bin/_internal/...
resources/bin/asr/neko-asr.exe
resources/bin/asr/_internal/...
resources/models/asr/paraformer/encoder.int8.onnx
resources/models/asr/paraformer/decoder.int8.onnx
resources/models/asr/paraformer/tokens.txt
resources/models/asr/public-speech.wav
```

任何操作者显式设置的 ASR 配置优先，包含不完整配置，不能静默替换。开发态不注入安装路径。可选 catalog 存在时，其状态目录放在 userData；默认单模型包不声称已提供模型选择界面。

## 可复核的无人值守验收

本机证据位于 `artifacts/windows-offline-20261105/`，原始日志与二进制不提交 Git。最终摘要和文件指纹另在主线 #1 记录。验收使用真正 Windows Sandbox，而不是只设置离线环境变量：禁网络与所有采集重定向，仅共享公共部署载荷和指定证据目录；来宾没有 Python、uv、Node 或 npm，音频服务停止并禁用。

1. 独立冻结 ASR 解压到含中文、空格的路径，在无开发环境的来宾完成 prepare、公开 WAV 流式识别、唯一 final、取消与隔离；缺失、损坏模型须明确拒绝，不能联网补下载。
2. NSIS 按用户安装到中文、空格路径。验证注册表版本、安装位置、所有预期文件长度及 SHA-256，只允许安装器生成的确切卸载程序不在原始文件清单中。
3. 安装后的冻结宿主通过真实 manager 和 WebSocket 消费公开 PCM，核对 partial、final、消费样本数与 final 文本哈希，并确认自有服务退出。
4. 启动真实安装的 Electron，保持 `app.isPackaged=true`，清除显式 ASR 路径，验证三服务同实例以及由安装资源自动配置的 ASR prepare。初次存储选择通过既有 API 完成；默认 free Core 的能力门控不能绕过。测试仅在来宾中配置无密钥的可独立 ASR Core，不调用云端生成。
5. 覆盖安装、公共资源变体更新/回切和卸载分别保存前后版本、文件清单与合成配置保留证据。相同代码和模型的资源包回切不代表跨代码版本或跨 schema 回滚通过。

QA 桌面包仅在 `NEKO_OFFLINE_ACCEPTANCE=1` 下启用早期采集拒绝和所有 WebContents 静音保护。此保护不替换 ASR、后端或传输。启动测试结束时先尝试关闭窗口，必要时回收持有句柄的自有进程；强制清理不能记为应用正常退出验收。全部测试结束后销毁本轮自建 Sandbox。

## 交付边界

当前是未签名的本地 QA 安装包，宿主采用 PyInstaller，Electron 验收运行时为 41.2.0。它不是仓库正式 Nuitka、签名发布流水线，正式分发文件指定的 Electron 41.10.1 尚需另验。离线运行通过也不代表能够在完全离线的构建机重新构建；构建工具链和 wheelhouse 未全部封装。

发布前仍需模型许可与再分发确认、代表性中文准确率、普通低配 CPU/慢盘基准、正式平台与更新策略，以及有人在场时的真实采集和听音验收。尚未验收的可选 embedding 不用占位文件冒充齐备。

## 正常退出与 Portable 验收方法

普通关窗可能保留托盘。正常退出从既有托盘“退出”进入；出现挽留弹窗时选择“今天先走”，等待现有告别动画和认证后端关闭流程。验收工具只启动、观察和记录进程，不能调用 Quit IPC、`CloseMainWindow` 或强杀来替代真实界面退出。

先真实 prepare 并观察模型子进程仍驻留，再执行正常退出。分别核对 Electron、冻结宿主、ASR supervisor/native worker 以及 Portable 外层启动器的创建身份、退出码和时间，检查端口关闭、认证关闭原因以及是否触发产品内部超时强杀。进程清单还要递归覆盖已持有父进程创建的系统辅助进程，不能只按安装目录过滤。

ASR v2 修复了一项 ACK 后的关闭竞态：模型子进程回复 shutdown 后，管理器给它最多 1 秒自行退出，再沿原有 terminate/kill 兜底。失败、超时或取消未收到 ACK 时保留原回收行为。Windows multiprocessing 的 65536 是 terminate 使用的退出码，不能单独用它判断 shutdown 是否成功，也不能把它记为自然退出码 0。

Portable 必须运行真实 SFX 文件，验证实际解压的 Electron 和资源路径，退出后同时等待内层应用、外层启动器及临时解压目录完成清理。相同包在两个含中文、空格的路径顺序运行；每次启动前排除已有应用与后端，防止单实例复用掩盖新路径启动失败。

electron-builder 的 Portable 环境变量本身不使用户配置随 exe 携带。当前测试使用同一用户的显式 QA userData 与默认后端存储，只验证同用户换 exe 路径后的配置保持；不代表 USB 用户数据迁移。检查无新增产品安装/卸载登记也不等同所有注册表零写入。最终实测结论和失败记录统一以主线 #1 为准。

## 0.9.2 检查点

2026-11-05：服务 v2 已进入 0.9.2 QA 包，干净断网 Sandbox 安装及安装后真实宿主流式 smoke 通过。安装后 Electron 正常退出、Portable 运行尚未验收完成；自动化中断和清理不作为通过证据。源码与 PC 接入补丁见 [首版检查点](local-streaming-asr-checkpoint.md)。
