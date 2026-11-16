# Projected-2D 复合图坐标契约复核

> **2026-11-09 原始基线核对更正（优先于下文解释）**：本轮 `FULL_RGB` 改了图像、gaze crop、提示、referent任务和坐标协议，不能作为原始 Experiment 2 基线。同32条原始8B预测经原evaluator重放，Point@100=0.2121、Joint@100=0.1970、Joint Exact@100=5/30；本轮FULL为0.0787/0.0630、Exact=0/30。864项原始计数全部复现，主要落差不能归结为样本或评分器。须先恢复原wrapper对照，再做区域辅助的单因素比较。详见 [基线核对](PROJECTED2D_BASELINE_AUDIT_20261109_zh.md)。

2026-11-09：32 条 projected-2D 配对首版要求模型输出源图 0..1 坐标，但结果中大量坐标落于复合图左上象限。以 `scene2_369` 的 P1 为例，模型称所找人物位于办公室左侧；原坐标 `[0.164,0.234]` 直接投到源图落在天花板，而按整张复合图坐标反算则落在人物附近。`scene4_room1_34` 和 `scene5_712` 显示不同响应也可能混用约定，不能统一对冻结输出乘二并当成正式结果。无 GT 的人工坐标可视化在 `.local-sync/projected2d_routed_rgb_20261109/coordinate_hypothesis_no_gt.jpg`。

首版 32 条四组已冻结并以原 2D evaluator 评分：Joint@100 全部 0，Point@100 仅 FULL 有 1 TP。存在输出坐标契约疑点，且原投影标签未独立标定；不能由此得出 gaze 效果。原结果在 `.local-sync/projected2d_routed_rgb_20261109/`，保留审计。

独立 v2 在相同 32 条原 P1/P2/P3 输入和四组上重新推理：明确要求模型输出整张复合图的 0..1000 坐标，脚本按实际标题高度、象限、窗口矩形与 letterbox 映射到该原帧 point_uv。落到标题、padding 或未提供的 frame ID 即拒绝；不用 3D ID。四组预算、完整 RGB、referent 解析与软时间提示不变。输入重用上一轮逐像素核验结果；坐标变换测试覆盖完整图、细节裁剪、标题和 padding。v2 也属于已接触结果后的探索性修复，不是独立留出。

独立运行目录：`.local-sync/projected2d_routed_composite_v2_20261109/`。v2 已完成四组模型推理和原 2D 评价，具体结果与失败分析见 [32 条 2D 结果](PROJECTED2D_V2_RESULTS_20261109_zh.md)。

```bash
source /ai/data/TriModal-Referring/.local-sync/reconstructed_pilot100/env.sh
cd /ai/data/MultiAgent
PYTHONPATH=. python -c 'from pathlib import Path; from MultiAgent.projected2d_routed_composite import infer; infer(Path("/ai/data/MultiAgent/.local-sync/projected2d_routed_composite_v2_20261109"))'
PYTHONPATH=. python tools/run_projected2d_routed_rgb.py evaluate --output /ai/data/MultiAgent/.local-sync/projected2d_routed_composite_v2_20261109
PYTHONPATH=. python tools/run_projected2d_routed_rgb.py exact --output /ai/data/MultiAgent/.local-sync/projected2d_routed_composite_v2_20261109
PYTHONPATH=. python tools/report_projected2d_routed_rgb.py /ai/data/MultiAgent/.local-sync/projected2d_routed_composite_v2_20261109
```
