# 完整指代 V4.1 位置 mention 策略小试（2026-11-08）

本轮只在 `scene2/411` 和 `scene4_room2/149` 上测试 V4.1 位置短语策略，保留 V4 的 32 条完整运行作为回退基线。输入画面、候选空间、预算和四条件保持一致，共 8 次本地模型调用。

对 `over there`、`near the small table` 这类没有独立场景对象身份的关系短语，要求模型输出 `unresolved/non_concrete_reference`；对明确命名 concrete scene object 的位置短语仍允许 grounded。

## 结果

8/8 条条件的 schema 和 mention coverage 均通过，0 个 mention ID 遗漏或重复。关系短语被单独保留为 unresolved，具体 table 对象仍被模型输出并进入 decoder。

| 条件 | Micro-F1 | Exact-set | Any-hit |
|---|---:|---:|---:|
| NO_ADDED_HEAT | 0.2000 | 0/2 | 1/2 |
| TRUE_HEAT | 0.2500 | 0/2 | 1/2 |
| SPATIAL_MIRROR | 0.2222 | 0/2 | 1/2 |
| TIME_REVERSE | 0.2222 | 0/2 | 1/2 |

样本只有 2 条旧交互，不能用于判断热图收益。V4.1 已解决输出协议问题，但没有解决 scene-ID 选择问题：例如 `scene2/411` 中多个 mention 选择相同或相近的视觉点后，decoder 仍会映射到错误的同类实例。

## 当前含义

主线已经从“模型有没有覆盖 mention”推进到“模型是否在完整交互中给出足够区分同类实例的视觉点，以及近似 decoder 是否能正确恢复 scene ID”。下一轮应优先扩大到未见交互，并把 mention coverage、grounded point、decoder rejection 和最终集合指标分开报告。

## 产物

- V4.1 runner：[run_full_frame_grounding_v41_location.py](../tools/run_full_frame_grounding_v41_location.py)
- V4.1 冻结运行：`.local-sync/full_interaction_referring_20261108/grounding_v41_location/`
- V4 原始回退基线：`.local-sync/full_interaction_referring_20261108/grounding_v4/`
