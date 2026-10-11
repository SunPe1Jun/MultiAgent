# 扩展 pilot（20 条）结果

本轮使用 C 方案 `V3_instruction_hybrid_asr_gaze_hand_2`，scene1 和 scene2 各 10 条，排除了原 smoke 的 8 条。样本按指令模式分层后固定随机填充，用于诊断泛化，不是总体准确率估计。

| 指标 | 结果 |
| --- | ---: |
| 样本/可评估 | 20/20 |
| Any-hit | 12/20（60%） |
| Exact-set | 4/20（20%） |
| Micro Precision | 42.22% |
| Micro Recall | 39.58% |
| Micro F1 | 40.86% |
| TP / FP / FN | 19 / 26 / 29 |

| 场景 | Any-hit | Exact-set | Micro F1 |
| --- | ---: | ---: | ---: |
| scene1 | 7/10（70%） | 3/10（30%） | 57.14% |
| scene2 | 5/10（50%） | 1/10（10%） | 27.45% |

逐条预测和完整轨迹位于 `V3/` 与 `traces/`。原始汇总见 [summary.json](summary.json)。

## 发现

1. 新样本 Any-hit 为 60%，低于原 8 条的 75%；原 smoke 结果不能代表扩展样本。
2. scene1 明显高于 scene2。scene2 的 chair、desk、person、PC 同类实例消歧仍是主要瓶颈。
3. Mention 候选映射保持较高覆盖，主要损失来自实例编号错误、多个对象漏选，以及视觉模型在证据不足时的空输出。
4. 发现 1 条视觉响应截断、8 条候选闭集过滤行；这些都已记录在对应 trace，不能简单归为视觉识别错误。
5. 本轮未启用 SAM regions，因此结果只评价 instruction、事件帧、gaze 和 hand 条件视觉。

本轮 20 条全部完成、可评估，`progress.json` 显示 completed=20。下一步应先针对 scene2 的同类实例加入候选 crop/区域跟踪消融，并修复视觉输出截断与候选过滤，再比较 Any-hit 和 Exact-set；暂不扩大到全量。
