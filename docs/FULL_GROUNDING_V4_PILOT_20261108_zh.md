# 完整指代 V4 mention accounting 小试（2026-11-08）

本轮在 2026-11-06 的同一批 8 条完整交互、同一组全画面输入、候选空间、视觉预算和四条件上测试新的 mention accounting 合约。没有使用新交互，也没有改变默认流程。

V4 要求模型为每个 parser mention 输出稳定 `mention_id`，每个 mention 恰好一行，并用统一的 `grounded/unresolved` 状态区分视觉定位和无法定位。重复的 literal mention 因而可以独立评价。模型 trace 在每次响应冻结后清除 `processor_inputs.pt` 和 `processor_rgb_*.png`，只保留响应、审计和 `trace.json`。

## 结果

32 个条件、29 次唯一模型调用全部完成，视觉输入消费核验通过。mention 结构结果为：

| 指标 | 结果 |
|---|---:|
| schema 有效条件 | 26/32 |
| mention 覆盖完整 | 26/32 |
| 有效 grounded 行 | 70 |
| unresolved 行 | 20 |
| 遗漏 mention ID | 0 |
| 重复 mention ID | 0 |

6 个不完整条件集中在两个包含关系位置短语的交互：

- `scene2/411` 的 `over there`；
- `scene4_room2/149` 的 `near the small table`。

模型把这些短语输出为 `category: location`，而场景候选词表只接受具体对象类别。它们不是 mention 遗漏，也不是重复 ID；这是 location mention 的协议策略问题。V4 已保留这些原始输出，没有事后修复。

完整集合评价仍然是旧交互回归，不能作为泛化结论：

| 条件 | Micro-F1 | Exact-set | Any-hit |
|---|---:|---:|---:|
| NO_ADDED_HEAT | 0.3704 | 1/8 | 3/8 |
| TRUE_HEAT | 0.2857 | 1/8 | 3/8 |
| SPATIAL_MIRROR | 0.3077 | 1/8 | 3/8 |
| TIME_REVERSE | 0.3571 | 1/8 | 3/8 |

这轮没有显示热图收益。V4 的价值在于把输出覆盖问题暴露并量化出来，使后续视觉线索对照不再被 V3 的空数组和重复 literal 混淆。

## 下一步

V4.1 只处理 location mention 的语义协议：对 `over there`、`near ...` 这类没有独立场景对象身份的关系短语，要求模型输出 `unresolved/non_concrete_reference`；对明确指向可见具体对象的目的地，例如 warehouse，仍允许 grounded。V4 原始 32 条响应保留为回退基线，V4.1 单独建目录，不覆盖 V4。

只有在这个协议稳定后，才进入未见交互，并重新比较 RGB、真实热图、空间镜像、时间反转和 foveation。

## 产物

- V4 代码：[full_frame_grounding_v4.py](../MultiAgent/full_frame_grounding_v4.py)
- V4 runner：[run_full_frame_grounding_v4.py](../tools/run_full_frame_grounding_v4.py)
- V4 测试：[test_full_frame_grounding_v4.py](../MultiAgent/test_full_frame_grounding_v4.py)
- 冻结运行：`.local-sync/full_interaction_referring_20261108/grounding_v4/`
- 存储清理记录：[STORAGE_RETENTION_20261108_zh.md](STORAGE_RETENTION_20261108_zh.md)
