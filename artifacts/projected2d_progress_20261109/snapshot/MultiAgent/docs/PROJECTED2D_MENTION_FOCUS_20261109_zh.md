# Projected-2D mention 聚焦验证

> **2026-11-09 原始基线核对更正（优先于下文解释）**：本轮 `FULL_RGB` 改了图像、gaze crop、提示、referent任务和坐标协议，不能作为原始 Experiment 2 基线。同32条原始8B预测经原evaluator重放，Point@100=0.2121、Joint@100=0.1970、Joint Exact@100=5/30；本轮FULL为0.0787/0.0630、Exact=0/30。864项原始计数全部复现，主要落差不能归结为样本或评分器。须先恢复原wrapper对照，再做区域辅助的单因素比较。详见 [基线核对](PROJECTED2D_BASELINE_AUDIT_20261109_zh.md)。

## 2026-11-09 首版完成，增加面板标签验证

32 条 2D 复合图坐标版中，GAZE_RGB 的 66 个输出点有 62 个选择 P1；FULL_RGB 为 56/61。原 manifest 中 P1/P2/P3 分别有 35/41/48 条 acceptable referent-panel 记录，定义依然依赖 gaze 与投影 anchor 的距离，不是独立时间真值。这只是 P1 过度集中迹象，不足以证明时序策略有害。

为检查早期 referent 与软时间工作如何复用，在已有 32 条中每个子场景选行号最小、解析成功且至少两个 mention 的事件，共 8 条 19 mention；选择按指令解析和地址，不按 GT 或模型分数，但发生在 32 条 2D 评分之后，因此是探索性开发子集。实际地址、指令与源输入见独立目录 `input_manifest.json`。

每个 mention 独立一次视觉请求，使用完整原指令、对应 mention 信息、软 phase 时间窗和全部原 2D panel；按面板时间距 phase 中心排序，最接近的排第一，其他面板全部保留作为上下文。提示明确顺序可非时间顺序，输出仍为实际 `frame_id` 和复合图坐标，脚本映射到源帧 `point_uv`。四组保持相同策略和图片布局；不把每个 mention 的推理成本伪称与原整句单次请求相同。使用与 32 条 v2 相同的原 2D evaluator 和显式父子共指辅助去重。

每个 mention 的完整输出保留在 `calls/`，拼接后的事件结果保留在 `responses/`；解析错误、无效坐标和调用失败保留。目录为 `.local-sync/projected2d_mention_focus8_20261109/`。首版模型推理和评价已完成：GAZE_RGB 的 Joint@100 为 6.45%，FULL/UNIFORM/MIRROR 分别为 0%/6.45%/0%，exact 全为 0/8。由于重排后图像并未可视化标识实际 P1/P2/P3，帧号关联存在歧义；相同八条和四组的标题内真实 panel ID 与时间核验版也已完成，见 [标识聚焦最终结果](PROJECTED2D_LABELLED_FOCUS_20261109_zh.md)。

```bash
source /ai/data/TriModal-Referring/.local-sync/reconstructed_pilot100/env.sh
cd /ai/data/MultiAgent
PYTHONPATH=. python -c 'from pathlib import Path; from MultiAgent.projected2d_focus import infer; infer(Path("/ai/data/MultiAgent/.local-sync/projected2d_mention_focus8_20261109"))'
PYTHONPATH=. python tools/run_projected2d_routed_rgb.py evaluate --output /ai/data/MultiAgent/.local-sync/projected2d_mention_focus8_20261109
PYTHONPATH=. python tools/run_projected2d_routed_rgb.py exact --output /ai/data/MultiAgent/.local-sync/projected2d_mention_focus8_20261109
PYTHONPATH=. python tools/report_projected2d_routed_rgb.py /ai/data/MultiAgent/.local-sync/projected2d_mention_focus8_20261109
```

该实验增加视觉调用，同时改变任务分解及图片顺序，不能把前后差异单独归因于时间策略或热图；四组对比可考察该聚焦协议下 gaze 区域作用。不可比较成独立评测结论。
