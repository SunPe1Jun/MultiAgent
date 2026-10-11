# GT 标注复核报告

本次扫描覆盖 scene1、scene2、scene3、scene4_room1–4、scene5，共 4,000 条指令。扫描使用当前 anchor table、alias table 和 Excel 的 instruction/referents 字段做词法对齐，再人工区分直接对象、空间参照物、部件父对象和可能的隐含区域。词法命中的疑点不是自动判错结果。

## 已确认并修正

| 文件 | ID | Excel 单元格 | 原标注 | 修正后 | 原因 |
|---|---|---:|---|---|---|
| scene4_room1.xlsx | Basic-047 | E48 | `curtain, stove` | `curtain, sofa` | 指令为 `Open the curtain near the sofa.`；stove 与指令关系不符 |
| scene4_room3.xlsx | Medium-104 | E105 | `vase, cabinet, rug` | `vase, cabinet, rug, sofa` | 指令明确包含 `in front of the sofa` |

每个文件都保留了修改前备份和 SHA256：`scene4_room1.xlsx.before.xlsx`、`scene4_room3.xlsx.before.xlsx`。

## 发现的高置信度候选

以下条目表现为 GT 中出现了与指令明显不一致的对象，或漏掉了明确的动作对象；它们暂时保留原值，避免把“空间参照物是否计入 referent”的标注规则误当成错误：

- scene4_room1 Basic-043：`Pick up the cup from the table.`，GT 为 `cup, rug`，疑似 `rug` 应为 `table`。
- scene4_room1 Medium-105：`Pick up the cup from the table, carry it to the cabinet...`，GT 为 `cup, rug, cabinet`，疑似 `rug` 应为 `table`。
- scene4_room1 Medium-138：`Dust the window, arrange the vase...`，GT 为 `cabinet, vase, lamp`，`cabinet` 与明确的 `window` 不一致。
- scene4_room2 Medium-109/135：指令中的主体是 `basin`，GT 同时包含 `sink`；需要确认 basin 是否按其所在容器计入 sink。
- scene4_room3 Basic-054：`Fetch the cushion from the sofa.`，GT 为 `cushion, desk`，疑似 `desk` 应为 `sofa`。
- scene4_room3 Medium-107：`adjust the lamp beside the window`，GT 为 `person, lamp, cabinet`，`cabinet` 与明确的 `window` 不一致。
- scene4_room4 Medium-130：`Wipe the dining table, move a chair...`，GT 缺少明确动作对象 `dining table`。
- scene4_room4 Medium-137：`Pick up a pillow from the bed...`，GT 缺少明确来源对象 `bed`。

## 不能仅凭词法判错的类别

- `washing machine door`、`cargo bay door` 等部件短语是否只标父对象，当前数据中存在统一性问题，需要先确定评测口径。
- `near the window`、`toward the bedroom side`、`in front of room1` 等空间参照物，在不同指令中有时计入 GT、有时不计入 GT。
- scene1/2/3/5 中大量 `by/near/behind/beside` 参照物未进入 GT，这可能是旧标注的 referent 定义，而非逐条错误。

详细机器扫描结果见 `suspected_gt_issues.json` 和 `suspected_gt_issues.csv`；这些文件明确标注了 `needs_review_not_confirmed_error`，没有据此批量修改 Excel。

## 检索规模与人工复核

扫描 4000 条，词法筛选标记 939 条（scene4 为 120 条）。这些数字包括隐含参照物、部件关系和别名误报，不是确认的错误数。人工复核代表性案例见 `reviewed_examples.json`。其他条目只做检索，未批量更改。
