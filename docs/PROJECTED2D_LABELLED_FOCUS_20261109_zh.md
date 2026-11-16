# Projected-2D：复用 referent 与软时间阶段的聚焦对照

> **2026-11-09 原始基线核对更正（优先于下文解释）**：本轮 `FULL_RGB` 改了图像、gaze crop、提示、referent任务和坐标协议，不能作为原始 Experiment 2 基线。同32条原始8B预测经原evaluator重放，Point@100=0.2121、Joint@100=0.1970、Joint Exact@100=5/30；本轮FULL为0.0787/0.0630、Exact=0/30。864项原始计数全部复现，主要落差不能归结为样本或评分器。须先恢复原wrapper对照，再做区域辅助的单因素比较。详见 [基线核对](PROJECTED2D_BASELINE_AUDIT_20261109_zh.md)。

2026-11-09 已完成 8 个事件、19 个 mention、四条件共 76 次本地 Qwen3-VL-8B BF16 视觉调用。复用此前冻结的 mention、对象类型、部件/共指关系和 `plan_frames` / `focus_spec` 软阶段，模型每次处理一个 mention，按面板时间距该阶段中心排序，全部原 2D 面板仍可选择。提示与画面均明确真实 P1/P2/P3 和时间；模型输出整张复合图坐标，脚本严格映射到源面板 2D 点，使用原 Experiment 2 evaluator。

样本由已评分 32 条开发集每子场景选择行号最小、parser 成功且至少两个 mention 的事件：scene1/495、scene2/369、scene3/161、scene4_room1/1、scene4_room2/19、scene4_room3/108、scene4_room4/72、scene5/226。选择按地址与语言解析，不按 GT 或指标，但发生在 32 条 2D 结果之后，因此不能作为独立验证。逐 mention 拆分、图像顺序及调用预算一起改变，不能将前后差异单独归因于时间策略；四组仍保持相同聚焦策略和预算。

早期聚焦版只在提示提供重排后的面板表，图片未直接标 P1/P2/P3，帧号关联存在歧义，保留在 `.local-sync/projected2d_mention_focus8_20261109/`。最终版仅在相同四组图片的标题栏增加真实 panel ID 与视频时间：96/96 张图的 RGB 和细节像素保持一致。最终版目录为 `.local-sync/projected2d_mention_focus8_labelled_20261109/`。

## 结果与成本

|条件|Temporal F1|Point@100 F1|Joint@100 F1|Joint Exact@100|
|---|---:|---:|---:|---:|
|FULL_RGB|56.25%|6.25%|6.25%|0/8|
|UNIFORM_RGB|56.25%|0.00%|0.00%|0/8|
|GAZE_RGB|60.61%|6.06%|6.06%|0/8|
|MIRRORED_GAZE_RGB|50.00%|0.00%|0.00%|0/8|

GAZE 与 FULL 各只有一个 Joint@100 TP。GAZE 相对 FULL 的差 −0.19 个百分点，8 事件配对 bootstrap 95% 区间 [−20.00,15.79] 点。相对固定/镜像各 +6.06 点，区间 [0.00,16.67] 点；仅一个命中的离散小样本，且下界为零，不能当成可靠阳性。共指保守合并在此八条没有触发，Exact 未改善；不能用无限制合并跨帧点来修成绩。

在同一八条上回算此前整句单次 v2：FULL/UNIFORM/GAZE/MIRROR 的 Joint@100 为 12.90%/0.00%/6.45%/6.67%，聚焦标识版为 6.25%/0.00%/6.06%/0.00%。GAZE 的 Joint TP 仍只有一个；前后均低且波动大，没有发现增加调用改善严格定位/Exact 的证据。每组逻辑请求从 8 增到 19，76/76 调用实际消费视觉输入、无缓存复用，合计 248,064 视觉 token；源整句相同八条每组 8 请求。模型输出前先冻结参数；全部调用及事件响应冻结后才运行本轮评价，构造原2D标注和既有开发评分已在此前发生。

## 帧号关联与错误

未标识聚焦版全部四组的 P1/P2/P3 点分布为 52/3/0；图像写入真实 ID 后变成 15/35/11，明显改变了帧选择。这验证面板标识应直接附于图像，但样本未有人类逐 mention 时刻标注，不能把分布更分散当成时间选择正确。GAZE 的 P1/P2/P3 为 6/7/3；其 Temporal F1=60.61% 高于同八条整句版，但原 temporal acceptable panel 按 gaze 与 projected anchor 距离生成，不是独立时间真值。

76 条调用中 70 完成、6 带 warning；全部 warning 为模型状态标 ambiguous 却仍给点，依既定规则拒绝这些点并保留响应，未通过猜测补点。32 个事件×条件记录中26完成、6带warning，无模型调用失败。61/61 个有效点通过原始复合图坐标→源图→复合图独立往返验证：41 个来自完整图，20 个来自细节区。可逆性不证明落点处对象正确。模型仍有前景虚拟手与 drone/cargo 混淆、同类实例选择和精细定位偏差；原投影代理自身也有未标定限制，不能单凭低分将全部失败归于热图。

## 产物与恢复

[查看最终8条逐样本图像及点](../.local-sync/projected2d_mention_focus8_labelled_20261109/index.html)、[32条主对照](PROJECTED2D_V2_RESULTS_20261109_zh.md)。运行目录内 `calls/`、`mention_calls_frozen.json`、`responses/`、`predictions_frozen.json` 分别保留逐 mention 调用、调用冻结、逐事件拼接和预测冻结；`summary.json` / `exact_set_analysis.json` / `focus_vs_joint_prompt.json` 为原2D评分、保守去重和同集预算对照；`header_audit.json` / `coordinate_audit.json` 记录像素与坐标审计。

```bash
source /ai/data/TriModal-Referring/.local-sync/reconstructed_pilot100/env.sh
cd /ai/data/MultiAgent
PYTHONPATH=. python tools/run_labelled_2d_focus.py infer
PYTHONPATH=. python tools/run_projected2d_routed_rgb.py evaluate --output /ai/data/MultiAgent/.local-sync/projected2d_mention_focus8_labelled_20261109
PYTHONPATH=. python tools/run_projected2d_routed_rgb.py exact --output /ai/data/MultiAgent/.local-sync/projected2d_mention_focus8_labelled_20261109
PYTHONPATH=. python tools/compare_projected2d_focus.py /ai/data/MultiAgent/.local-sync/projected2d_routed_composite_v2_20261109 /ai/data/MultiAgent/.local-sync/projected2d_mention_focus8_labelled_20261109
```

本轮实现、配对推理和评价已完成。下一步需要先固定可见对象、同类实例和时刻对应的独立检查，再将完整坐标/图片标识/时间协议冻结到未见 2D 样本；本批已评分样本不再用于调参后冒充泛化评价。
