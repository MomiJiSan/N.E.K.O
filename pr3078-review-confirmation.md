# PR #3078 差异复核

比较基线：16a48f711f6842e3fbd181c41675459bb3c025bc。
PR HEAD：299c0b4b8c14eb386c645ad180241c6756e0b5c1。
GitHub API 返回实际 changed_files=109。watch-together、图片生成已在基线中，不属于本 PR 独有改动。

## 已确认

| 场景 | BASE | HEAD |
| --- | --- | --- |
| C1：声纹和独立 ASR 关闭，Gemini 发送遇到 ConnectionClosedOK 后完成热切换 | 新会话收到 320 字节 | 新会话收到 0 字节；session_closed_by_server 仍为 True |
| C9：取消热切换后，替换会话 close 抛异常 | 吞掉关闭异常；pending cleanup 和 reset 各执行一次 | 关闭异常传播；两项清理均未执行 |
| C9：第二次取消已排队，进入替换会话清理 | close 同步前缀执行 | 新建 close task 在首步前被取消；close 未开始 |
| C4/C7：schema 1 档案、requested_enabled=False，升级 | has_profile=True，reason=disabled | has_profile=False，reason=profile_incompatible；启动时不阻断 PCM，但重录后 requested_enabled=True |
| schema 1 档案、requested_enabled=True、enforce 模式 | 旧初始化错误分支不下发阻断权限 | 两个注册 manager 的 native PCM 均被阻断 |
| schema 1 档案、requested_enabled=True、off 模式 | — | 两个注册 manager 的 native PCM 均可提交 |
| 未配置激活 factory，健康 native/independent 输入各三帧 | 960 字节，无新状态通知 | 960 字节，无新状态通知；无 protected prefix |

C1 根因：_stream_audio_gemini 新增异常传播，Core 在捕获正常关闭后设置 manager 级关闭标志；热切换更换 session 不复位该标志。连接代次校验不解决 manager 标志残留。

C9 根因：_retire_replacement_session 无 handoff ticket 时直接 await 新建 close_task，没有保留调用点原先的 Exception 隔离，且增加一次调度/取消窗口。这里不是 activate(None) 的问题。

## 撤回的错误判断

此前“关闭降噪导致永远无法激活”的 P1 不成立。原始 HEAD 的 VoiceInputAudioPipeline.process 返回 evidence.peak，而非局部 probability；RNNoise 不可用时 evidence.peak 已经是 None。以原始 HEAD 方法和真实 AudioProcessor(nr_enabled=False) 验证：静音 voice_activity=False，有声 voice_activity=True。之前直接给辅助函数传 0.0 的实验没有复现生产路径。

基于该误判添加的 audio_input.py 本地修改已经撤回。原有 core/asr_runtime.py 的 runtime 构造失败状态通知补丁仍在本地，未提交；它不修复上述 C1/C9。

## 证据与边界

tests/unit/test_pr3078_review_evidence.py 含 11 项固定提交对照探针。通过 git show 读取原始方法 AST，以同一组模块依赖加载 BASE/HEAD 方法；执行真实 Core 路由、完整热切换方法、旧格式加密档案读取和 registry 门控。Provider 网络、模型推理及部分外围操作使用测试替身；普通 independent 测试只证明 Core 提交边界，未替代实际 ASR worker、最终转写或服务器实测。

11 项探针全部通过，含对 HEAD 错误行为的明确断言；这不是“PR 已修复/验收通过”。现有大批测试全绿也不能抵消这些差异。

GitNexus 查到主工作树的历史调用图，但索引比 review HEAD 落后 23 个提交，且没有该 review 工作树的独立索引。图结果仅作历史定位，未作为无影响证明。本轮没有修改 C1/C9 生产实现、commit、push 或 GitHub 评论。

## 本地修复状态

review 工作树已加入 C1/C9 最小修复：成功 promote 后清除 Gemini 关闭闩锁；replacement close 的普通异常隔离并继续收尾。新增当前 HEAD 正向断言后，语音回归与差异探针共 114 项通过。修复尚未 commit/push，PR 远端仍为原始 HEAD，故当前 PR 仍不满足零回归/零行为变化要求。

## 推送后的补充修复

补充修复 `60c78164f`：录入开始时保存用户原有 `requested_enabled`，因此声纹关闭用户完成首次录入或重录后仍保持关闭，不会被隐式开启。该修复及对应回归测试已推送至 PR。
