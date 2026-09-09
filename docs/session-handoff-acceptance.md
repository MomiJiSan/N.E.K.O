# 会话交接实机验收

脚本为 `scripts/session_handoff_acceptance.js`，在真实麦克风归属窗口的 DevTools Console 粘贴执行。网页使用 `/`；Electron 选择加载 `appState`、持有麦克风的主窗口，而不是观察窗口。先开启后端并确认麦克风权限、音频设备及 provider 可用。脚本不代替实机验收，也不生成“通过”的伪记录。

## 执行

### 确认运行的是修复工作树

本机修复目录为 `E:/Work/CODE/N.E.K.O-session-handoff-lifecycle`，分支为 `codex/session-handoff-lifecycle`。原目录 `E:/Work/CODE/N.E.K.O` 未被修改，从原目录启动不能验证本次修复。定版前的开发测试不能充当最终验收，代码变动后须重启后端并重新加载前端。

使用正式启动器启动整个修复后端。它负责三个服务的共同实例身份、待处理的存储迁移和重启；不能用独立采样服务入口代替这些启动职责。在旧测试服务已停止后，一个 PowerShell 终端执行：

```powershell
Set-Location E:/Work/CODE/N.E.K.O-session-handoff-lifecycle
$env:UV_PROJECT_ENVIRONMENT='E:/Work/CODE/N.E.K.O/.venv'
$env:DO_NOT_TRACK='1'
uv run --no-sync python launcher.py
```

Electron 要求 main / memory / agent 三个服务均健康，并且返回同一个 `instance_id`；只启动主服务和记忆服务会导致 `NEKO_BACKEND_HEALTH_TIMEOUT`。同时须检查 `/api/storage/location/bootstrap` 的 `blocking_reason` 为空，不能只看 `/health` 返回 200。

2026-09-09 22:36 已通过正式启动器完成用户在页面选择的目录切换，当前数据目录为 `E:/Work/N.E.K.O`。源测试目录保留为迁移备份。主、记忆和 Agent 服务统一就绪；`selection_required`、`migration_pending`、`recovery_required` 均为 false。此前独立服务入口不处理待迁移事务，存储重启后不会自动拉起，不再作为首选实测启动方式。

浏览器访问 `http://127.0.0.1:48911/`。Electron 连接同一个后端，在实际持有麦克风的窗口测试。结束时在启动器终端按 Ctrl+C；若 Electron 已因后端未就绪而超时，在启动器就绪后重新打开它。

此前独立采样记录为 `output/session-handoff-lifecycle/resources.jsonl`。改用正式启动器后该文件不再自动更新，不能用旧样本评价后续测试。定向功能实测可先保留前端导出与当前后端日志；性能和资源验收须另行接入采样。不要在两份程序同时占用 48911 / 48912 / 48915 时开展对比。

### 操作矩阵

先设置声纹关闭，记录 provider、TTS 路由、ASR、语言、输入和输出设备。保留同配置的旧版本基线，再运行修复版本。浏览器和 Electron 分别执行以下两轮；替换 `environment`，不要同时运行：

```js
await sessionHandoffAcceptance.run({environment: 'browser', phase: 'idle', configLabel: 'voiceprint-off'});
sessionHandoffAcceptance.export();
await sessionHandoffAcceptance.run({environment: 'browser', phase: 'speaking', configLabel: 'voiceprint-off'});
sessionHandoffAcceptance.export();
```

默认每轮包含 0、50、100、300、1000 毫秒，各 10 次。脚本点击实际 `micButton` 开麦，点击实际 `muteButton` 关麦。GitNexus 与源码核对的调用链为 `muteButton → stopMicCapture → stopRecording → pause_session`；`stopButton` 是停止屏幕共享，不能用于关麦。

`speaking` 每次在 Console 提示后，请对麦克风说话引发真实回复。脚本只在实际 `appState.isPlaying` 为真时关麦；30 秒没有观察到说话状态，记为 `skipped`。重开确认后每次也请触发回复，以记录首段音频。15 秒没有音频记为 `incomplete_audio`，不会算作通过。`idle` 先等待不在播放，再关麦。

中断使用 `sessionHandoffAcceptance.stop()`，导出使用 `sessionHandoffAcceptance.export()`。停止不会主动关掉已经恢复的麦克风，请用正常 UI 结束。启动失败或超时立即中止整个矩阵，不进行故障重试。`operator_stop` 是操作员中止，不计生命周期失败。

## 记录与审核

导出的 JSON 仅包含时间、状态、请求关联和二进制字节数；不保存文本、音频正文、配置密钥。记录 `requestId`、通知关联、开麦到 ack、录音就绪和首二进制音频耗时，以及请求间隔和 `performance.now()` 实测间隔。浏览器定时器不是精确调度器，0 毫秒也会产生调度延迟，必须使用 `actualGapMs` 分组比较。

目前前端没有独立的 TTS runtime ready 事件，因此 `ttsReadyMs` 为 `null`，不能拿 ack 冒充 TTS 就绪时间；需结合后端日志测量。首音频只关联“本请求 ack 之后的 audio_chunk header + 二进制帧”，协议没有在音频帧上回显启动 request ID，因而这项数据不能独自证明音频确实属于新会话。人工听音与后端 runtime 日志仍须核对，发现旧音频要计为失败。脚本的 `observed_requires_review` 表示已采样，并不表示验收通过。

同步记录后端线程数、LLM 连接中/预热/当前/退休数量和 TTS runtime 数量。正常结束后观察资源回落；异常未退出资源仍需占容量。provider 故障单独归类，保留原始错误码和时间，不记成生命周期误关。对每个环境/状态/间隔，缺少 10 条可审核记录的档位必须补验；不能把 skipped 或 incomplete 算作样本完成。

完成声纹关闭原场景后，再分别复验正常 TTS 复用、独立 ASR、预热、游戏语音、语言切换和多窗口发起/麦克风租约/观察窗口通知。最终审核应确认无误关、误清、过期通知覆盖或无限准备状态，并比较同配置基线的 ack、TTS ready、首音频分布。将本次修复作为整体回退。

## 可重放保护反证

执行 `uv run python scripts/check_session_handoff_mutations.py`。脚本在隔离子进程内依次移除 runtime 归属、容量、cleanup shield、通知归属、热切换 callback 排空和队列消费者退出保护；不修改生产源码。每个正式回归必须因行为断言失败才计为 `KILLED_BY_ASSERTION`，导入、语法、超时和取消异常不计为有效反证。六项均被断言杀死时命令返回 0。脚本显式绑定所在 worktree 的模块路径，不依赖 `PYTHONPATH` 或 editable install。正常生产测试仍须单独全部通过。
