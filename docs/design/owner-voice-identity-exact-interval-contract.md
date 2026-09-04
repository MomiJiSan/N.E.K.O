# Owner 声纹 exact 子回合局部裁决合同

> **状态：Current contract。** 本文记录独立 ASR 在 Provider 给出可信 exact boundary 时，对单个 Provider 子回合执行局部声纹 DROP/FORWARD 的当前行为。代码与测试具有最终权威。

声纹录入与 Profile 提交由 [Owner 声纹录入与独立验证合同](./owner-voice-identity-enrollment-contract) 约束；transport-wide 正式拒绝完成后的恢复由 [Owner 声纹拒绝后的 ASR 恢复合同](./owner-voice-identity-deny-recovery-contract) 约束。

## 1. 目标与语义边界

当两段顺序语音已经被 Provider 识别为两个独立、精确且互不重叠的音频区间时，运行时可以让它们分别接受声纹裁决：

```text
exact A：正式拒绝 → 只 DROP A 的 partial/final
exact B：验证通过 → 只 FORWARD B 的 transcript
```

“局部裁决”只控制 transcript 是否进入 Core。音频在边界到达前已经发送给 ASR Provider，本合同不撤回 Provider 已接收的音频，也不承诺 Provider 不处理该音频。

本文明确不提供：

- 基于标点、字符串、final 文本或词级时间戳的切割；
- 同一个 Provider item 内部的文本二次切割；
- 重叠说话的声源分离；
- 对 unknown、gap、overlap 或歧义边界的局部授权；
- 声纹阈值、模型、检查点时机或 Provider 映射的改变。

## 2. 父 Speaker Lease 的证据合同

父 `SpeakerCaptureLease` 使用以下内部状态：

```text
COLLECTING + HIGH
  → HIGH_SEEN

HIGH_SEEN + CaptureClosed
  → ALLOW

FIRST_LOW + CaptureClosed
COLLECTING + CaptureClosed
SpeakerUnavailable
  → UNAVAILABLE → FORWARD

FIRST_LOW + SECOND/COMPLETION LOW
  → DENY_LATCHED

FIRST_LOW + HIGH
HIGH_SEEN + LOW
  → MIXED_DENY_LATCHED
```

`HIGH_SEEN` 不是立即终态，因此一个子回合的 HIGH 不能担保同一父 lease 中后续子回合。`DENY_LATCHED` 与 `MIXED_DENY_LATCHED` 都永久粘滞，后续 HIGH 不能复活。单次 LOW 后 capture close、声纹后端 unavailable 与普通 Owner HIGH 的既有 fail-open/allow 策略保持不变。

可能产生 DROP 的父 lease 转换使用两阶段所有权：

1. Admission 在同一 FIFO 和锁内 prepare transition，返回携带 logical revision 的 terminal claim；
2. Runtime 在无 `await` 的线性化点设置 `DENY_FENCED`；
3. Admission 以 CAS 校验 revision 后 commit，并由 coordinator 单写 terminal fan-out。

claim 只能消费一次。prepare 后父状态、generation 或 owner 发生变化时，commit 必须返回 stale/conflict，旧 claim 不得关闭新 session。

## 3. exact 资格证明

局部能力只有在下列事实同时成立时才能启动：

- Provider boundary 含合法 start/end，且属于当前 Provider timeline generation 与 key；
- Detector、detector epoch、session epoch、Runtime identity 和 ingress token 当前有效；
- PCM sequence 连续，区间无 gap、overlap 或未知 ownership；
- Speaker Shadow buffer、evidence lease 与 candidate generation 一致；
- 目标 Provider child 是父 lease 唯一且最后一个 child；
- 尚无 successor child 开始或绑定；
- Admission child 尚未 terminal，partial/final 仍受 Admission 控制；
- active exact transaction 与 pending transaction 的总量未超过既有容量上限。

如果 exact target 与父 candidate 不同，父 lease 必须仍为无历史证据的 `COLLECTING`。已经存在 FIRST LOW 或 HIGH 的父证据不能迁移到另一个 candidate；该情况返回 conflict 并退回父组处理。

相同 key、相同 boundary 的重复通知幂等合并。相同 key 的不同 boundary 是安全冲突，不得把先到的 exact proof 降级成 unknown 后继续局部放行。

## 4. 可撤销的跨层事务

exact 建立严格按以下顺序执行：

```text
Runtime 安装 pending fence
  → Detector prepare
  → Admission promote tail child
  → Admission activate exact hold
  → Detector commit
  → Runtime 无 await 发布全部 alias
  → 记录 Provider exact proof
```

### Detector prepare

Detector 在自身有序锁/队列内验证音频与 evidence ownership，预留 target candidate 和可选 suffix candidate，冻结相关 Speaker Shadow buffer，并返回不可伪造、可撤销的 reservation。prepare 不发布 completion callback，也不改变 Admission ownership。

prepare 期间继续到达的 PCM 只能进入有界 provisional suffix scratch。abort 必须把 staged PCM 按原顺序恢复到父 candidate；worker 不可用、队列满、身份变化或恢复失败时 abort 返回失败，不能擦除 PCM 后声称安全回滚。

### Admission promote 与 activate

coordinator 在同一把锁内校验父 lease token/revision、child generation/revision、Provider key、boundary proof、唯一尾部 child 和 candidate ownership。promotion 原子地：

- 从父 `child_bindings` 移除 target child；
- 将 child 放入 exact HOLD；
- 把 suffix 安装为父 lease 的新滚动 candidate；
- 返回一次性 typed receipt。

activation 只把已迁移的 evidence 投影到 exact child，仍不触发 transcript resolution。父 terminal fan-out 若先完成，promotion 必须 stale；promotion 若先完成，父 fan-out 不得再包含该 exact child。

### Detector commit 与 Runtime alias

Detector commit 后，target/suffix 的 Speaker Shadow ownership 与 PCM 分割生效。Runtime 在下一次 `await` 前一次性发布 provider key、target candidate、turn、parent lease、suffix evidence 和 proof 的全部 alias，避免紧随其后的 completion callback 观察到半切换状态。

suffix 不绑定到旧 target turn。后续 Provider started 必须为它建立新的 child/turn identity。

## 5. pending FIFO、取消与接管

exact prepare/promote/activate 期间到达的 ordered endpoint 和 final 放入有界 FIFO，事务完成后按原顺序回放。final 先于 ordered 到达时使用已经建立的 exact boundary/proof 封口，不制造 unknown boundary。

所有跨层 `await` 后重新校验 Runtime、session、transport、Detector/epoch、ingress、Provider key、parent/child revision、dispatcher 与 receipt ownership。取消遵循以下规则：

- Detector commit 前：shield 已接纳的 Admission 命令，取得确定结果后执行对偶 abort；
- Detector commit 后：不能退回 unknown，必须以 `SpeakerUnavailable` 推进既有 fail-open 终态，或升级父组 cleanup；
- FIFO replay 被取消时仍须有界排空到安全终态，再向调用者重新抛出取消；
- 旧任务只能退出，不能回滚新状态或修改 successor。

只要 pending 或 active exact transaction 仍在 Runtime map 中，Provider namespace reset 就拒绝 reconnect takeover。即使 disposition 已经算出，只要 partial、tombstone、lifecycle 或 correlator settlement 尚未结束，也不能接纳新 session。transaction 完全退休后才允许重连。

## 6. 局部终态与 transcript 安全

exact child 复用父 lease 的 evidence reducer，但使用 `EXACT_INTERVAL` 内部作用域：

- exact ALLOW/UNAVAILABLE 只 FORWARD 对应 child；
- exact `DENY_LATCHED` 或 `MIXED_DENY_LATCHED` 只 DROP 对应 child；
- effects 只包含对应 child 的 `SettlePartial` 与 `ResolveReserved`；
- 不产生 `AbortProviderTransport`，不改变 Provider session generation；
- `AdmissionDisposition` 仍只有 `FORWARD`、`DROP`、`ABANDON`，不增加公开枚举值。

DROP 只有在 transcript dispatcher 返回 `APPLIED`，或 `ALREADY_SAME` 且既存 disposition 明确为 `DROP` 时才安全。`NOT_RESERVED`、既存 `FORWARD`、dispatcher 替换或 ownership 漂移必须升级父组 cleanup；不得把“请求过 DROP”误写成“DROP 已生效”。

exact terminal settlement 完成后才退休 Runtime alias、Provider proof、candidate binding 与 ownership。被 DROP 的 partial/final 永不进入 Core，也不能由迟到 callback 复活；successor 的首帧、partial 和 final 继续使用正常链路。

## 7. 降级与 transport-wide 拒绝

局部能力不能延迟正式拒绝：

- 第二次 LOW 在语音仍进行时到达，立即执行既有 transport-wide cleanup；
- 只有 exact boundary 已建立、child promotion 已提交后产生的 completion-confirmation DENY，才允许局部 DROP；
- exact proof 来晚后不能把已经开始的 `DENY_FENCED`、`RETIRING` 或 `QUARANTINED` 降级为局部处理；
- successor 已开始、proof 冲突、receipt stale、rollback 失败、effect 失败或父组完整性无法证明时，退回整组 cleanup；
- 整组 cleanup 仍无法证明 transport、transcript 与 ownership 安全时进入 `QUARANTINED`。

因此典型可局部裁决的序列是：

```text
A FIRST LOW
  → Provider exact endpoint A
  → exact A completion-confirmation LOW
  → 只 DROP A
  → B 使用 suffix/new child 收集证据
  → B HIGH + capture close
  → 只 FORWARD B
```

如果 A 在 endpoint 前已经得到两次 LOW，仍立即关闭整个旧 Provider transport。这是保留 fail-closed 的刻意行为，不是局部能力缺失。

## 8. 容量、资源与验证合同

- parent child、exact proof、Admission ingress、preseal FIFO 与 Speaker Shadow PCM 使用既有有界容量，不因局部能力扩大；
- 容量耗尽不淘汰 HOLD child、不静默 FORWARD，exact 尝试撤销并退回安全父组；
- prepare、commit、abort、reset 和 close 后，临时 target/suffix PCM、receipt 与 candidate binding 必须对偶释放；
- Detector identity reset 同步清空 exact reservation registry，旧 receipt 在 reset/close 返回前失效；
- 不新增数据库、配置、原始音频日志、字符串切割规则或 Provider 专用策略分支。

自动化至少覆盖：局部 A DROP/B FORWARD、single LOW fail-open、mixed evidence 粘滞拒绝、final-first、pending FIFO、重复/冲突 boundary、容量、gap/overlap、旧 socket、取消、rollback、reset/reconnect takeover、dispatcher tombstone 冲突，以及 local DROP 不关闭 Provider session。

真实验收必须同时验证：非 Owner 独立短句不进入 Core、随后 Owner 独立句进入 Core、local DROP 前后 Provider session generation 不变；没有足够静音或两人重叠时不得声称完成局部裁剪。
