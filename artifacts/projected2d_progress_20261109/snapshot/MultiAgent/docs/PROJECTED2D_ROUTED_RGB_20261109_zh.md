# Projected-2D 的脚本引导 RGB 区域实验

> **2026-11-09 原始基线核对更正（优先于下文解释）**：本轮 `FULL_RGB` 改了图像、gaze crop、提示、referent任务和坐标协议，不能作为原始 Experiment 2 基线。同32条原始8B预测经原evaluator重放，Point@100=0.2121、Joint@100=0.1970、Joint Exact@100=5/30；本轮FULL为0.0787/0.0630、Exact=0/30。864项原始计数全部复现，主要落差不能归结为样本或评分器。须先恢复原wrapper对照，再做区域辅助的单因素比较。详见 [基线核对](PROJECTED2D_BASELINE_AUDIT_20261109_zh.md)。

## 2026-11-09 首版与坐标修复均已完成

首版四组推理和评分已完成，但发现模型可能混用复合图与源图坐标；这些分数不能用于判断gaze效果。相同样本和视觉输入的v2坐标契约修复、推理及评价已完成，见 [复合图坐标契约复核](PROJECTED2D_COMPOSITE_CONTRACT_20261109_zh.md)。

用户明确：本轮视觉辅助属于 2D setting。此前 `referent_rgb_expansion32_20261109_repaired` 实际将模型点映射到 3D anchor ID 并以集合评分，不是本实验，不能引用其“语言基线”或分数。

本次复用此前 32 条地址、指令、冻结的 referent mention 及原 clause 软时间提示。通过原 `exam2/build_2d_eval_manifest.py` 和重建事件行选取原协议 P1/P2/P3；其中一条只有两个实际面板。205/205 条 `scene,row_index,referent_name,panel_id,gt_x,gt_y,projection_valid,status` 与发布的 projected-2D 清单一致，95/95 源 JPEG 复现及真实 PTS 核验通过。原 API 输入行并未恢复，行和 temporal acceptable-panel 排名由已有遥测和原规则重建；这项一致性不证明像素几何或时钟已绝对标定。

四组均使用相同原面板和完整 RGB、模型、mention、输出与评价，分别是 `FULL_RGB`、固定区域 `UNIFORM_RGB`、脚本根据过去 2 秒 gaze 三阶段选择区域 `GAZE_RGB`、镜像空间注意分布的 `MIRRORED_GAZE_RGB`。细节窗口是同帧真实 RGB，模型不收到热图、gaze 分数、GT 或场景 candidate ID。固定区域组沿用相同支持门控，故不是彻底去除 gaze 的对照；原始录屏可能包含采集标记。

RGB 全图统一缩到最长边 512 后再裁细节，故细节不会增加源视频高分辨率信息。保留完整画面和两个或三个原面板，模型对各 mention 返回真实面板 ID 和完整源图归一化 `point_uv`；没有 3D anchor ID 预测/解码。时间提示是软阶段先验，不是词级语音对齐，不排除其他面板。

**380/380** 条条件输入可从源 RGB 与固定矩形逐像素重建，27/27 无时钟支持面板四组完全相同；95 个原面板中 68 有时钟支持，23/32 事件至少一个面板有支持；31/32 冻结语言解析成功，失败保留。

原 2D evaluator 按像素距离计算 Temporal、Point@K、Joint@K，未改匹配规则。它在 mention 文本不能与 GT 标签精确匹配时回退为按空间距离匹配，因此不是语义 referent 身份真值；acceptable panel 由 gaze 与投影 anchor 的距离排序，也不是独立人工时间真值。Exact-set 指原 evaluator 下某阈值事件 FP=FN=0，并报告可评价事件分母，不等同于语义对象 ID exact-set。

另固定辅助去重：仅同一明确 `parent_mention_id`、相同 coreference ID、单数、同帧且两点间不超过 50 个原图像素时合并；不同帧或仅共享 coreference 的记录不自动合并。未合并原输出是主分析，GT 不参与去重。

路径：`.local-sync/projected2d_routed_rgb_20261109/`。`panel_parity_audit.json`、`input_manifest.json`、`input_reconstruction_verification.json`、`exact_set_policy.json` 分别保存面板对照、输入、逐像素核验及固定的去重规则。`responses/` 是首版按条件冻结的模型结果，因坐标契约疑点仅供审计。

```bash
source /ai/data/TriModal-Referring/.local-sync/reconstructed_pilot100/env.sh
cd /ai/data/MultiAgent
PYTHONPATH=. python tools/run_projected2d_routed_rgb.py infer
PYTHONPATH=. python tools/run_projected2d_routed_rgb.py evaluate --output /ai/data/MultiAgent/.local-sync/projected2d_routed_rgb_20261109
PYTHONPATH=. python tools/run_projected2d_routed_rgb.py exact --output /ai/data/MultiAgent/.local-sync/projected2d_routed_rgb_20261109
PYTHONPATH=. python tools/report_projected2d_routed_rgb.py /ai/data/MultiAgent/.local-sync/projected2d_routed_rgb_20261109
```

样本此前已用于另一项 ID-set 诊断，因此是探索性研究，不是独立留出。旧的 3D ID-set 分数不得和本次 2D 指标比较。
