# V3 与 V3 + SAM 对照

| 指标 | V3 | V3 + GroundingDINO/SAM | 变化 |
|---|---:|---:|---:|
| Any-hit | 85.00% | 80.00% | -5.00 pp |
| Exact-set | 35.00% | 30.00% | -5.00 pp |
| Micro Precision | 72.22% | 78.79% | +6.57 pp |
| Micro Recall | 54.17% | 54.17% | +0.00 pp |
| Micro F1 | 61.90% | 64.20% | +2.29 pp |

空输出变化：`{'base_nonempty_to_sam_empty': 1, 'unchanged': 18, 'base_empty_to_sam_nonempty': 1}`。

| 样本 | V3 | V3 + SAM | SAM 观察 |
|---|---|---|---|
| scene1/scene1_262 | person1 | ∅ | 0 positive evidence, 41 regions |
| scene1/scene1_695 | building1, cargo1, person1, platform1 | platform1 | 0 positive evidence, 52 regions |
| scene1/scene1_280 | person1 | person1, person2 | 16 positive evidence, 65 regions |
| scene2/scene2_664 | ∅ | ∅ | 0 positive evidence, 35 regions |
| scene2/scene2_474 | chair1, desk1, person1 | chair4, desk2, person1 | 14 positive evidence, 149 regions |
| scene2/scene2_123 | ∅ | chair1, person1 | 7 positive evidence, 71 regions |
| scene3/scene3_148 | person1, truck1 | truck1 | 0 positive evidence, 37 regions |
| scene4_room1/scene4_47 | curtain1, sofa | sofa | 0 positive evidence, 10 regions |
| scene5/scene5_347 | drone1, warehouseC | drone4, warehouseB | 2 positive evidence, 64 regions |

## 空输出原因

`scene2_664` 在两次运行中都为空：file mention 的 ASR 回退窗口较宽，SAM 没有生成可与 file candidate 可靠关联的 region；desk 虽有少量 region option，但 Visual 仍全部给零支持。
`scene1_262` 在 SAM 运行中变为空：区域候选主要是未关联的 building/person proposals，Visual 没有把它们确认成闭集候选，因此全部为零支持；另外 Referential 已将 `side door` 映射为 `door`，但该场景的 door 候选集合为空，所以模型输出的 `door1..door3` 被 closed-set 校验拒绝。这是上游候选 schema 覆盖不足，和视觉证据为零是两个叠加问题。
`scene2_123` 从空输出变为 `person1 + chair1`：SAM 区域和 Visual 产生了正支持，但 chair 实例仍选错（GT 为 chair5），说明区域辅助提升了可解析性，却没有完成实例身份对齐。

完整逐条对照见 `comparison.json`；SAM 运行报告见 `RESULTS_zh.md`。
