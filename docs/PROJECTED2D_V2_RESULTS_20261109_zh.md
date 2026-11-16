# Projected-2D RGB 区域对照：32 条结果

> **2026-11-09 原始基线核对更正（优先于下文解释）**：本轮 `FULL_RGB` 改了图像、gaze crop、提示、referent任务和坐标协议，不能作为原始 Experiment 2 基线。同32条原始8B预测经原evaluator重放，Point@100=0.2121、Joint@100=0.1970、Joint Exact@100=5/30；本轮FULL为0.0787/0.0630、Exact=0/30。864项原始计数全部复现，主要落差不能归结为样本或评分器。须先恢复原wrapper对照，再做区域辅助的单因素比较。详见 [基线核对](PROJECTED2D_BASELINE_AUDIT_20261109_zh.md)。

独立目录：`.local-sync/projected2d_routed_composite_v2_20261109/`。原面板和本次坐标契约详见 [坐标契约复核](PROJECTED2D_COMPOSITE_CONTRACT_20261109_zh.md)。这 32 条已接触过前一 ID-set 诊断和首版 2D 评分，v2 在发现坐标问题后重跑，属于探索性研究，不是独立留出。

128 条逻辑条件记录中 123 完成、1 条因点落标题拒绝而带警告、4 条为同一上游 parser 失败保留空输出；124 条有效视觉记录均通过输入消费验证，其中 27 条使用完全相同图像及提示的响应复用。所有 254 个有效点通过复合图→源图→复合图算术往返核验；这只验证坐标变换，不证明对象识别正确。

|条件|Temporal F1|Point@100 F1|Joint@100 F1|Joint Exact@100|
|---|---:|---:|---:|---:|
|FULL_RGB|53.54%|7.87%|6.30%|0/30|
|UNIFORM_RGB|55.81%|6.20%|3.10%|0/30|
|GAZE_RGB|51.52%|6.06%|4.55%|0/30|
|MIRRORED_GAZE_RGB|52.31%|7.69%|4.62%|0/30|

32 个事件参与 evaluator，30 个至少存在一个可评价的时空 referent，因此 exact 的可评价分母是 30。GAZE 相对 FULL 的 Joint@100 差 -1.75 个百分点，交互配对 bootstrap 95% 区间 [-7.20, 3.33]；相对 UNIFORM 差 +1.44 点，区间 [-5.55, 8.09]；相对镜像差 -0.07 点，区间 [-6.11, 6.29]。区间均跨零，不能确认真实 gaze 的独立收益。原 evaluator 的 GT 是 3D anchor 投影的 2D 代理，绝对几何未标定，temporal acceptable 由 gaze 与代理距离生成，并非人工时空标注。

GAZE_RGB 在 30 个可评价事件中 26 条同时有漏选和多选，4 条只有漏选；四组各有 1 次显式父子共指、同帧且近距离的保守去重，Joint Exact@100 均仍为 0/30。这个失败分解依赖原 evaluator 的近距离回退匹配，并不等于已核实的语义身份错误。四组 GAZE/FULL/UNIFORM/MIRROR 的源帧点中，P1 分别占 62/66、56/61、59/63、59/64；可接受 GT panel 记录 P1/P2/P3 为 35/41/48，提示第一面板偏好，不能单凭提示声称已有有效时间选择。

四个预先固定样本的原图模型点与事后投影代理对照：`.local-sync/projected2d_routed_composite_v2_20261109/fixed_2d_points_review.jpg`。例如 scene2_369 的模型将 person/laptop 点在左侧可见实例，原代理指向另外的实例附近；scene5_712 的模型证据把前景手状物描述为 drone，存在模型视觉语义错误。个例不足以确定全部失败原因，也不能证明代理点正确。

按 mention 聚焦与阶段面板重排的小试及帧号标识复核均已完成，详见 [标识聚焦结果](PROJECTED2D_LABELLED_FOCUS_20261109_zh.md)。全部冻结点和事后GT代理可通过 [32条逐样本查看器](../.local-sync/projected2d_routed_composite_v2_20261109/index.html) 复核。
