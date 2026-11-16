# 完整指代 Decoder 与 mention 覆盖审计（2026-11-08）

本报告只审计 2026 年 11 月 6 日已经冻结的完整交互输出，没有新增模型调用，也没有修改输入、提示、候选、响应或预测。已有 GT 指标只作为预测冻结后的事后上下文。

## 当前发现

下一步瓶颈是 mention accounting 和视觉点到 scene ID 的解码覆盖，优先级高于继续调热图或 foveation 的呈现形式。

- V3 的复合坐标结构检查 32/32 通过，但最终只保留 33 条视觉行，累计 63 个 mention hint 没有被覆盖。
- V3 有 14 次把完整对象行错误放入 `unresolved`，另有 4 行类别不在场景词表中。
- V2 保留 120 条视觉行，其中 39 条因视觉点与投影 anchor 中心距离过远被拒，20 条因没有可用相机/候选被拒，6 条因同类 anchor 无法区分被拒。
- V3 的几何拒绝下降到远点 2、无支持 4、同类歧义 2，但 mention 覆盖成为主要损失。

## 结构计数

| 版本 | 条件响应 | 结构有效 | 有效视觉行 | 映射到 scene ID | 未覆盖 mention |
|---|---:|---:|---:|---:|---:|
| V2 | 96 | 88 | 120 | 55 | 不适用 |
| V3 | 32 | 32 | 33 | 25 | 63 |

V3 的“结构有效”不能单独当作任务完成率，因为模型可以返回空的 `referents` 和 `unresolved`，同时遗漏全部 parser mention。

## 对主方向的含义

主任务仍然是完整时间顺序交互指代：解析原始指令中的每个 mention，查看完整 RGB 交互和可选注意线索，让模型选择每个具体对象，最后评价 referent 集合。当前结果没有支持新的热图收益结论，只暴露出两个更早的协议损失：

1. 重复或关系 mention 只用 literal 文本表示。两个 “that car” 无法稳定区分具体发生次序。
2. `referents` 和 `unresolved` 使用不同结构；模型有时把完整对象行放进 `unresolved`，验证器只能保留原始错误并拒绝它，不能偷偷修复。
3. schema 有效不代表 mention 覆盖。V3 可以返回 `{"referents": [], "unresolved": []}`，但仍遗漏多个 parser hint。
4. V2/V3 的 scene-ID decoder 仍是近似的 anchor-center 代理。远点或同类中心歧义属于解码限制，不能直接当作模型视觉识别错误。

## 下一步实验设计

在再次扫热图/foveation 之前，建议在同一完整画面输入和同一候选空间上测试 V4 mention accounting 协议：

- 为每个 parser mention 分配稳定 `mention_id`，包括重复出现的 mention；
- 强制每个 mention occurrence 恰有一行输出；
- 统一行结构，使用 `status: grounded | unresolved`；
- 只有 `grounded` 才要求 `category`、`view_id` 和 RGB 点；
- `unresolved` 必须给出 literal mention 和明确原因；
- 保留原始非法行，单独报告覆盖率，不用 GT 修复；
- 继续使用同一个 GT-free scene-ID decoder，把 mention 覆盖、视觉定位和 decoder 拒绝分开报告。

这样可以先隔离输出协议和重复 mention 的损失，再进入真正未见交互的 RGB、热图和 foveation 对照，主研究方向不变。

## 产物

- 可复现工具：`tools/audit_full_grounding_decoder.py`
- V4 合约原型：`MultiAgent/full_frame_grounding_v4.py`
- V4 合约测试：`MultiAgent/test_full_frame_grounding_v4.py`
- 汇总：`.local-sync/full_grounding_decoder_audit_20261108/summary.json`
- 逐行决策：`.local-sync/full_grounding_decoder_audit_20261108/row_decisions.json`
- 被审计的冻结运行：`.local-sync/full_interaction_referring_20261106/grounding_v2/` 与 `grounding_v3/`
