# Owner 声纹录入与独立验证合同

> **状态：Current contract。** 本文记录桌面端 Owner 声纹录入、音频处理域、独立验证和 Profile 提交的当前行为。代码与测试始终具有最终权威；历史 Phase 4C 文档只描述各阶段当时的边界，不覆盖本文。

## 1. 目标与适用范围

本合同解决两个问题：

1. 录入 Profile 与桌面运行时候选必须来自同一音频处理域，避免因为采样率、降噪或重采样路径不同而产生错误 `LOW`。
2. 三段参考音频形成声纹后，必须再用一段独立语音验证 Profile，向用户显示本轮最保守的匹配结果；验证通过前不得替换或启用 Profile。

适用范围仅为桌面端 Owner CAM++ Profile。它不承诺不同设备、移动端或不同音频合同之间的 Profile 可互换。

## 2. 用户流程

```text
开始录入
  → Segment 1：3 秒参考语音
  → Segment 2：3 秒参考语音
  → Segment 3：3 秒参考语音
  → 建立三段参考 centroid
  → Segment 4：5 秒独立验证语音
  → 计算 1.5 / 3.0 / 5.0 秒三个检查点
  → 三项全部通过：提交并激活 Profile
  → 任一项未通过：返回最低匹配率并按服务端进度重试或重置
```

前三段显示为 3 秒录音。浏览器为服务端流式重采样保留 100 ms 采集余量，服务端规范化后只使用精确 3 秒。第四段必须精确上传并使用 5 秒，不增加余量，避免越过 5 秒硬上限。

四段共用一次麦克风租约和同一组采集参数。服务端 `next_segment_index` 是唯一权威进度；客户端不得根据本地计时自行跳段或提交 Profile。

## 3. 音频处理域

桌面录入固定使用 `owner-campplus-desktop-v1` 合同：

| 阶段 | 合同 |
| --- | --- |
| 浏览器输入 | PCM16LE、mono、48 kHz |
| 浏览器配置 | 与桌面运行时相同的麦克风设备、客户端增益和 `getUserMedia` 约束 |
| 服务端分块 | 每块 480 samples |
| 服务端处理 | 复用运行时 `VoiceInputAudioPipeline` 的 NR、AGC、Limiter 与 soxr 路径 |
| 模型输入 | PCM16LE、mono、16 kHz |
| 参考长度 | 规范化后的前 3.0 秒 |
| 验证长度 | 规范化后的前 1.5、3.0 和 5.0 秒 |

每个 Segment 创建独立 normalizer，不能复用上一段的 DSP 内部状态。录入开始时冻结当前降噪开关，四段使用同一快照；运行时配置与 Profile 快照不一致时，声纹证据必须为 `UNAVAILABLE` 并 fail-open，不能比较后产生 `LOW`。

浏览器若实际无法建立 48 kHz `AudioContext`，或请求缺少/伪造音频合同，录入直接失败，不允许把 44.1 kHz 或旧 16 kHz 数据标记成 48 kHz。

## 4. API 合同

录入段只使用：

```http
PUT /api/voice-identity/enrollment/segment
Content-Type: audio/pcm;format=pcm_s16le;rate=48000;channels=1
X-Voice-Audio-Contract: owner-campplus-desktop-v1
X-Voice-Identity-Enrollment: <opaque enrollment id>
X-Voice-Identity-Profile: <opaque profile id>
X-Voice-Identity-Segment: 1..4
```

请求体上限按 Segment 区分，`Content-Length` 与流式读取执行相同限制：

| Segment | 用途 | 最大 source PCM | 最大字节数 |
| --- | --- | --- | --- |
| 1–3 | 参考录音 | 4 秒 @ 48 kHz mono PCM16 | 384,000 |
| 4 | 独立验证 | 5 秒 @ 48 kHz mono PCM16 | 480,000 |

Segment 4 的匹配成功和匹配失败都是正常验证结果，HTTP 响应瞬时增加：

```json
{
  "verification": {
    "passed": false,
    "match_percent": 31
  }
}
```

`GET /api/voice-identity/status`、幂等恢复响应和其他 Segment 响应不得包含 `verification`。响应丢失时，客户端只能根据服务端进度恢复，不得构造或缓存一个匹配率。

## 5. 参考一致性与独立验证

前三段分别生成 CAM++ embedding，并执行段间一致性检查；只有一致的三段才能形成 L2-normalized centroid。

第四段与前三段完全独立，生成：

- `score_1_5`：centroid 与前 1.5 秒 embedding 的余弦相似度；
- `score_3_0`：centroid 与前 3.0 秒 embedding 的余弦相似度；
- `score_5_0`：centroid 与完整 5.0 秒 embedding 的余弦相似度。

通过条件固定为：

```text
score_1_5 >= 0.40
and score_3_0 >= 0.40
and score_5_0 >= 0.40
```

用户界面显示三个检查点的最低值：

```text
match_percent = round(
  clamp(min(score_1_5, score_3_0, score_5_0), 0, 1) × 100
)
```

最低值与通过判定保持同一方向，避免界面显示高分但因为未展示的检查点失败。该百分比只描述本轮三个检查点中最弱的声纹相似度，不是身份认证准确率、概率或安全等级。

## 6. 失败、重试与恢复

| 情况 | 行为 |
| --- | --- |
| 第一次独立验证低分 | 正常返回 `passed=false` 和最低匹配率，保持在 Segment 4 |
| 第二次独立验证低分 | 清除本轮参考数据，服务端进度重置到 Segment 1 |
| 静音、削波或真人语音门失败 | 使用既有音频错误合同，不解释成匹配失败 |
| normalizer、Silero 或 CAM++ 不可用/超时 | 不保存新 Profile，不产生伪造 `LOW` |
| 响应丢失 | 读取服务端状态恢复进度，不恢复匹配率 |
| 取消、页面关闭或 TTL 到期 | 停止 track、AudioContext、worklet、定时器和上传；服务端退休 Session |
| 降噪配置与 Profile 不一致 | 撤销声纹激活并返回音频合同不匹配，运行时 fail-open |

录入 Session TTL 保持 45 秒，硬租约保持 60 秒。增加 5 秒验证不能扩大生产 Speaker Shadow、Provider candidate、ASR endpointing 或候选拒绝缓冲；运行时声纹 PCM 上限仍为 4 秒。

## 7. 提交事务与异步边界

每段遵循：

```text
锁内预留 operation
  → 锁外音频规范化、Silero 与 CAM++ 推理
  → 锁内 CAS 校验并提交结果
```

每个关键 `await` 后必须重新校验 session generation、operation nonce、segment index 和 task identity。取消、过期、重录、重复请求或新 Session 接管后，迟到的归一化、验证和推理结果不得写回。

旧 Profile 和启用偏好在第四段验证及完整提交成功前保持不变。通过后才依次 stage Profile、激活 Runtime、保存 preference 并提交加密文件；任一步失败都必须回滚本轮未提交状态。

Profile 使用 schema/AAD v3，保存合同 ID、合同 revision、录入时降噪快照和通过验证后的 reference centroid。旧 schema 明确不兼容，不能静默补字段或跨音频域继续比较。

## 8. 隐私与可观测性

允许持久化的生物特征仅为验证通过后、受现有加密 Profile 保护的 reference centroid。以下内容不得持久化、进入产品日志、诊断快照或状态 API：

- 原始或规范化 PCM；
- 三段临时 reference embedding；
- 1.5、3.0、5.0 秒 holdout embedding；
- 三个原始余弦分数；
- `match_percent`。

临时 embedding 在成功、失败、取消和异常路径都要尽力清零。HTTP 响应不得返回 PCM、embedding、单检查点分数或可用于重放的中间数据。

## 9. 验收边界

解除 Draft/实验门禁前，除自动化测试外还必须完成：

1. Electron 使用实际麦克风走完 3 / 3 / 3 / 5 秒计时和资源释放；
2. 私有真实语料验证 Owner 三检查点不产生 false-LOW；
3. impostor false-HIGH 不高于统一音频处理域之前；
4. 响应丢失、取消、TTL、配置切换和 Profile 回滚在真实桌面链路成立。

自动化测试通过不能替代真实语料与 Electron 验收，也不能把匹配率解释为认证准确率。
