# M3 真实麦克风与双端验收状态

2026-11-05 回复终态修复：共享adapter与WebSocket文本收尾按回复段/队列version封口全部已有气泡，同版本延迟尾句直接创建为sent。新增9项回归、相关77项Node测试及75项websocket_static测试全部通过，独立review通过。网页和真实Electron各验证无标点尾句、多句延迟队列，共8个目标气泡最终全部sent、44个实际音频source自然结束，文字完整、队列清空；两修复JS的实际HTTP served SHA与冻结源码一致。原文所述Full长期streaming问题已在本轮修复，前次结果保留为历史。运行当时代码尚未提交/推送。

本轮详细结果为 `artifacts/m3/streaming-terminal-fix-20261105/issue-result.md`，以及[issue实测记录](https://github.com/MomiJiSan/N.E.K.O-ASR/issues/7#issuecomment-6647375409)，运行 `20261105-013618-4b9fd9`；网页另有启动greeting单列，不冒充目标测试，也不以总cancel=0替代逐目标判定。保持禁mic/camera、无声sink及前后端屏幕采集拒绝；自有进程已清理。真实麦克风、ASR→Realtime WSS、物理听音和安装包验收范围不变。

2026-11-05 更新：网页与真实 Electron 已分别完成两轮文本 → 真实 Qwen HTTP → 现用 CosyVoice → 客户端静音解码/调度/自然结束。独立审计四句共23个 native source 全部成对结束、无播放区间取消、无重复播放、终态队列清空、采集轨道为0。10个生产链路文件起终SHA一致，运行当时代码尚未提交/推送。这是文本/LLM/TTS静音软件链路通过，不包含真实麦克风、本地ASR→Realtime WSS、物理听音或安装包验收。

最终运行 `20261104-170309-0166dc` 使用独立资料与明确禁采集/无声sink。首轮观测keepalive停更、Electron临时入口门问题已仅在测试artifact修正并重跑。run-b曾被原有中文首次默认的主动视觉功能触发非空截图响应；发现后停止实例，在最终运行关闭主动功能并明确拒绝前后端屏幕采集，最终全部page没有截图请求/响应。不能声称早期截图仅留本机；脱敏事实与归属限制见本地报告和issue。Full回复实际显示且音频结束，但DOM `data-message-status` 仍为 `streaming`，单列终态标记限制。

完整新结果为 `artifacts/m3/real-device/silent-e2e-result.md`；逐句metadata、独立audit和清理在最终run及Electron c目录。截图仅保留可分享的验收应用区域；密钥、raw日志、音频和桌面图像不上传GitHub。下文为前次尝试的历史记录，未完成项已由本次文本静音子项进展覆盖，真实设备项仍待现场。

---

2026-11-05，本轮真实设备/网页/Electron整链路尚未完成。用户不在公司，停止实体麦克风和发声操作；之后界面自动化因不能可靠确认Windows浏览器URL被安全保护中止。

已实测Qwen conversation文本HTTP200、现用CosyVoice生产worker返回可解码Opus；实际main/memory/agent健康、业务资料隔离及禁麦/摄像头响应头通过。没有把这些后端结果当作前端播放通过。当前React已用真实源码与现有依赖构建；真实Electron的独立profile与静音/禁麦临时入口已准备，但Electron本轮未启动。

默认输入是实体 USB 麦克风，默认多媒体输出是虚拟音频端点；现场需先确认实际物理输出。设备probe只读metadata，meter samples=0，没有capture/render流、录音或声学回环。三服务已按owned身份关闭并核对端口停止。

完整结果、失败尝试、观察范围和清理证据见本地 `artifacts/m3/real-device/device-stage-result.md`，对应issue #7。安全bootstrap只用于本轮无人值守静音检查，不能用于宣称真实mic通过。

现场验收仍需分别在网页与Electron跑通：实际getUserMedia硬件输入、worklet PCM、本地ASR partial/唯一final、真实realtime Core与所选TTS、客户端实际解码调度/播放结束、真实可听输出、关麦和退出销账。不得以fake media、虚拟端点、后端audio_done或模型API smoke替代。
