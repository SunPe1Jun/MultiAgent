# 跨场景 smoke 审计

2026-10-10。固定 16 条现有 pilot 样本；scene1/2/3/5 各 2 条，scene4 四个房间各 2 条。采用原始指令解析、ASR 辅助时间、逐 mention Visual、全局 coordinator、30% 时钟保护。全部完成，无进程中断；未开启 region 分割。

## 结果

| 场景 | 样本 | Any-hit | Exact-set | Micro-F1 |
|---|---:|---:|---:|---:|
| scene1 | 2 | 50% | 0% | 40% |
| scene2 | 2 | 100% | 0% | 33.33% |
| scene3 | 2 | 100% | 100% | 100% |
| scene4（四房间合计） | 8 | 100% | 62.5% | 83.33% |
| scene5 | 2 | 50% | 50% | 44.44% |
| 合计 | 16 | 87.5%（14/16） | 50%（8/16） | 67.53% |

这是小样本功能检查，scene4 占一半，不能作为全数据集性能估计，也不是与原 TriModal 的同样本配对比较。上表为本次 v15 端到端推理的实际结果，未用后处理猜测替换预测。

## 解析和候选覆盖

共提取 49 个 mention，46 个有候选。未映射的是 scene2_664 的 clutter，以及 scene5_347 的 passage、ground；场景表没有这些独立对象，不能随意补到某个实体上。全部 40 个 GT 对象都出现在对应指令的候选池并集中：本次没有因类别映射遗漏 GT，但候选覆盖并不等于实例识别正确或 mention 标注完全正确。

确定的解析漏洞：scene1_262 对 `the side door of the small building` 又提取了内部的 `the small building`；scene3_721 对 van/truck 的部件短语又提取了内部父对象，共 3 个重复。已在 mention_mapping.py 做严格的字符区间去重：仅去掉部件内部、文本与 parent_text 一致、候选集合相同的父对象重复项。后面独立出现的 `that van` 仍保留，原 mention ID 和字符位置不重排，抑制记录写入 suppressed_nested_mentions。

已用本次 16 条缓存语言输出回放验证：49→46 个 mention，3 个嵌套重复被去掉，全部候选集合保持不变；34 项单元测试通过。尚未为该去重重新跑端到端视觉推理，因此不宣称其提高上表指标。

## 失败证据与容易推进的修正

1. **重复词组被当成同一时刻。最高优先级。** scene2_474 的两个 chair 分别在字符 [8,18]、[52,62]，ASR 提示窗口也不同，但两个 Visual 都选择 chair1。第二个 mention 的解释竟引用第一个动作 `Arrange that chair`，且证据引用 F1。scene2_664 两个 desk 也都选 desk2。当前隔离的仅是候选表，各 mention 仍看到全部帧、完整指令，并被提示考虑全部帧。建议提供当前 mention 的突出显示、所属动作片段及局部窗口主视图；全视频仅作为辅助上下文。在可靠时间条件下优先比较局部帧，在时间拒绝时使用宽阶段范围并明确不确定性。不能硬编码不同 mention 必须选不同实例。

2. **时间保护退化范围过大，需先校准再拆分。** 12/16 触发 30% 保护，只有 4 条继续保留音频时间辅助；其中 room3 两条仅因 wall_audio_to_video 被拒绝。当前代码同时检查 telemetry_to_video、audio_to_video、wall_audio_to_video，任一越界就禁用音频/telemetry 时序和 coordinator 空间提示。因此它避免严重错误窗口，但也损失不少信息。应分别记录音频→视频和 telemetry→视频的可靠性，再考虑只降级失效链路。目前 telemetry 校准本身依赖音频元数据，不能无证据直接恢复；低于 30% 也只是通过启发式阈值，并不证明校准正确。本轮所有样本实际提取的 RGB 均有变化，未发现所有帧退化为同一张的情况。

3. **最终 coordinator 仍接收全场景 3D 表。** Visual 的候选隔离在最终选择层并未完全延续。建议做“只提供当前指令候选并集+明确的空间关系”对照，再比较仅在重复指代/关系冲突时触发协调。该代码事实说明有干扰路径，但本轮没有消融，不能断言它导致了多少错误。

4. **同类实例仍缺乏可验证的视觉身份。** scene1_695 的 cargo2/platform1 正确，但 person2→person1、building5→building10；scene4_47 curtain1→curtain2、scene4_150 chair2→chair、scene4_11 window→window2。候选类别全部正确。可优先在可信投影下提供当前 mention 的同类候选对照裁剪，并输出各候选区分依据。静态 `near` 关系可计算距离作为弱校验：room1 的 curtain1 到 sofa 约 1.34m，curtain2 约 1.94m，有明确可比较的关系；但移动后目标位置、投影误差和对象中心距离不能当作硬 GT 规则。

5. **here/there 的回指先于新增对象。** scene5_347 的 `there` 最终额外输出 yard_inland，GT 只有 drone3、warehouseD；同时 drone 和 warehouse 的实例也都错了。建议把 here/there 与前文 passage/warehouse 的关系作为待判断的回指链，只有独立证据支持新的实体时才新增 ID。不能简单删除所有 here/there，也不能据模型输出修改这条 GT。

## 建议下一轮小对照

保持这 16 条及 GT 不变，先只验证“局部 mention 主视图+所属动作片段”，重点看 scene2 两条是否仍复用同一 chair/desk，以及其他场景是否回归；再分别验证 coordinator 表范围和时间链路校准。分别报告端到端 Any-hit/Exact-set/F1、每个 mention 的依据帧、时间回退比例和实例混淆。多项策略同时修改会难以判断改善来源。

机器可读明细：summary.json、failure_audit.json。完整日志：../cross_scene_smoke_v15.log。
