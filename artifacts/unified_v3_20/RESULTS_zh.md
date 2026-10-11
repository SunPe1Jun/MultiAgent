# 统一 V3：20 条真实样本实验报告

## 配置与评估口径

- 样本：`pilot20/manifest.json` 的全部 20 条，按场景分层选取，偏向多对象与关系指令。
- 模型：Qwen3-VL-30B-A3B-Instruct；faster-whisper-small.en，CUDA float16。
- 流程：Referential → ASR mention 时间窗/事件回退 → RGB + gaze heatmap → Hand cue → Visual → 多对象集合。
- gaze 与 Hand 只辅助视觉；候选 evidence source 仅有 `visual`。GroundingDINO/SAM 未启用。
- 协调器仍允许明确实例 ID、类型唯一候选直接由语言确定；不能把每个输出都归因于视觉证据。
- 微平均指标使用所有样本的 TP/FP/FN 汇总计算；Any-hit 表示至少命中一个对象，Exact-set 要求集合完全一致。
- 已逐条核对场景、事件与 manifest，并与原始 evaluator 的集合计数交叉验证。

## 总体结果

| 指标 | 结果 |
|---|---:|
| 完成 / 可评估 | 20 / 20 |
| Any-hit | 17/20 = 85.00% |
| Exact-set | 7/20 = 35.00% |
| Micro Precision | 72.22% |
| Micro Recall | 54.17% |
| Micro F1 | 61.90% |
| TP / FP / FN | 26 / 10 / 22 |

## 分场景结果

| 场景 | 样本数 | Any-hit | Exact-set | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| scene1 | 3 | 3/3 | 0/3 | 50.00% | 33.33% | 40.00% |
| scene2 | 3 | 1/3 | 0/3 | 33.33% | 10.00% | 15.38% |
| scene3 | 3 | 3/3 | 2/3 | 100.00% | 83.33% | 90.91% |
| scene4_room1 | 2 | 2/2 | 1/2 | 75.00% | 75.00% | 75.00% |
| scene4_room2 | 2 | 2/2 | 1/2 | 75.00% | 75.00% | 75.00% |
| scene4_room3 | 2 | 2/2 | 0/2 | 80.00% | 80.00% | 80.00% |
| scene4_room4 | 2 | 2/2 | 2/2 | 100.00% | 100.00% | 100.00% |
| scene5 | 3 | 2/3 | 1/3 | 60.00% | 50.00% | 54.55% |

## 时间对齐与证据诊断

- 全部语言 mention：57，ASR 成功匹配 37；有闭集候选的 mention：47，其中匹配 30。
- 最终时间窗状态：`{'asr_aligned': 37, 'event_fallback': 20}`。
- 帧级 gaze 状态：`{'valid': 117}`。
- Hand cue 115 条，正支持视觉证据 36 条，校验拒绝视觉证据 20 条。
- mention 选择路径：`{'visual_ranking': 13, 'unresolved': 19, 'unique_language_candidate': 6, 'explicit_language_id': 19}`。
- 从原始失败响应恢复的调用：1 次；只修复 JSON 分组结构，原始响应与代码修复记录均保留。
- 输出长度重试：1 次；模型及提示词不变，上限从 2,048 提高至 8,192 token，每次重试有独立缓存。
- ASR 匹配成功率不等于时间戳准确率；音频、视频和遥测的内容起点尚未人工校准。

## 逐条结果

| 场景 / 事件 | 预测 | GT | 漏选 | 多选 |
|---|---|---|---|---|
| scene1/scene1_262 | person1 | building6, person1 | building6 | ∅ |
| scene1/scene1_695 | building1, cargo1, person1, platform1 | building5, cargo2, person2, platform1 | building5, cargo2, person2 | building1, cargo1, person1 |
| scene1/scene1_280 | person1 | building6, person1, person2 | building6, person2 | ∅ |
| scene2/scene2_664 | ∅ | desk7, desk8, file1 | desk7, desk8, file1 | ∅ |
| scene2/scene2_474 | chair1, desk1, person1 | chair4, chair7, desk4, person1 | chair4, chair7, desk4 | chair1, desk1 |
| scene2/scene2_123 | ∅ | chair5, desk8, person1 | chair5, desk8, person1 | ∅ |
| scene3/scene3_148 | person1, truck1 | person1, truck1 | ∅ | ∅ |
| scene3/scene3_721 | truck1, van1 | building2, truck1, van1 | building2 | ∅ |
| scene3/scene3_79 | van1 | van1 | ∅ | ∅ |
| scene4_room1/scene4_29 | lamp, sofa | lamp, sofa | ∅ | ∅ |
| scene4_room1/scene4_47 | curtain1, sofa | curtain1, stove | stove | sofa |
| scene4_room2/scene4_57 | door, switch | door, switch | ∅ | ∅ |
| scene4_room2/scene4_150 | chair, table | chair2, table | chair2 | chair |
| scene4_room3/scene4_103 | chair | chair, dining_table | dining_table | ∅ |
| scene4_room3/scene4_104 | cabinet, rug, sofa, vase | cabinet, rug, vase | ∅ | sofa |
| scene4_room4/scene4_26 | bedroom, bench | bedroom, bench | ∅ | ∅ |
| scene4_room4/scene4_11 | curtains1, window | curtains1, window | ∅ | ∅ |
| scene5/scene5_347 | drone1, warehouseC | drone3, warehouseD | drone3, warehouseD | drone1, warehouseC |
| scene5/scene5_433 | crate, person1 | crate, person1 | ∅ | ∅ |
| scene5/scene5_292 | truck | drone2, truck | drone2 | ∅ |

## 结果边界

这是固定 20 条 pilot 的描述性结果，不能代表完整数据集，也不能单凭本次结果判断 gaze、Hand 或 ASR 的独立增益。旧 E1–E6 使用不同融合逻辑，不能当作本次 V3 的消融。

机器可读文件：`summary.json`、`diagnostics.json`、`per_sample.csv`、`per_sample.json`；完整输入与模型响应见 `cache/qwen/`，逐条过程见 `traces/`。
