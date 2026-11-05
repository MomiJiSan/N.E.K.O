# 本地流式 ASR 首版源码检查点

2026-11-05，源码分别交付到服务仓库 `feat/m1a-streaming-service` 与宿主仓库 `feat/m1b-local-streaming`。唯一进度与验收入口是 [ASR 主线 #1](https://github.com/MomiJiSan/N.E.K.O-ASR/issues/1)。这是工程原型检查点，正式 main 和发布流水线仍需后续审阅与合入。

已实现 Streaming Paraformer INT8（默认 CPU 单线程）、流式 partial/唯一 final、取消与逐句隔离；复用宿主 Smart Turn、字幕和 Core 交付。prepare、冷启动、进程归属、租约与空闲释放，以及可信候选校验、预热、逐句切换和失败回滚均有回归。模型包接口允许后续同契约导出产物接入；本轮不包含训练或自动采集。

本次回归：独立服务 293 passed / 11 skipped；宿主相关 Python 582 passed / 1 skipped；前端 Node 88 passed；设置静态检查 24 passed；无设备 DOM 回归 13 passed；PC 补丁套件 11 passed。真实模型及长稳测试按环境变量单独启用，不能将本次无模型测试计数当成再次完成真实设备验收。宿主首次测试遇到系统临时目录 ACL 问题，使用本轮专属 basetemp 后完成回归，没有修改系统权限。

Windows 0.9.2 未签名 QA 包已在干净断网 Sandbox 完成安装及安装后真实宿主 ASR 流式 smoke；安装后的 Electron 正常退出、Portable 实际运行验收仍未完成。此前清理或自动化中断不计为正常退出通过。服务 v2 在收到 shutdown ACK 后给 worker 最多一秒自然退出，超时仍沿用原有强制回收。

桌面 QA 的五个接入源文件与两项测试套件以 [PC 接入补丁](patches/local-streaming-asr-pc.patch) 保留，包含打包资源路径注入及仅在显式 QA 开关下启用的静音/拒绝采集保护。补丁从验收快照提取，只包含接入块和新文件；未修改原 PC 工作区，未声称已提交到 PC 仓库，也不包含安装器配置、构建产物或完整桌面发布工程。应用补丁前需在目标 PC revision 复核上下文。

权重、WAV、二进制、私有配置和原始日志不随源码提交。发布仍需代表性中文准确率、低配 CPU/慢盘基准、模型许可与再分发、正式签名/平台矩阵，以及有人在场的真实麦克风、Realtime LLM/TTS 与听音验收。历史长稳和静音双端证据及限制见主线 #1 与相邻验收文档。
