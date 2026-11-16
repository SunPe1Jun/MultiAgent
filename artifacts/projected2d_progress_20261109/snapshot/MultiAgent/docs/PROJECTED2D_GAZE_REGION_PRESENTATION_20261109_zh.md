# 2D 注视区域像素提示：主图对照

> 历史诊断。本角标实验不属于当前主线视觉处理；当前回到原 Experiment 2 paired full RGB + gaze crop 输入，详见 [原基线视觉迭代](PROJECTED2D_ORIGINAL_BASELINE_VISUAL_ITERATION_20261109_zh.md)。

用户要求将每个时刻的 gaze 信息转成模型能理解的主图像素提示，同时保留 referent 和时间判断。此处的“像素提示”是数据呈现方式，不意味着模型已经正确理解。它不是新增热图图片、目标框、场景 3D ID 或目标标签。完整原始 RGB 保留，逐 mention 输出原面板上的 2D 点。

## 需要纠正的输入假设

前一版 `MAIN_RGB_768` 提示称可以比较绿色 gaze 点，但它没有把清晰可识别的 gaze 标记作为输入协议绘制出来。manifest 中另有帧级 gaze 投影信息，但不能因此断言模型在 RGB 中看到了对应点。10 条交互的提示词对照中，只要求利用时间的 v3 和要求识别绿色点的 v4 均选 P1/P2/P3 = 18/3/3；v4 的 29 个 mention 中 27 个将 gaze 关系写为 unclear，2 个写为 supports。由此只能说模型在这批输入上基本没有报告可用的 gaze 关系，不能概括为所有录像都不存在注视标记。

本轮没有使用只把单帧 gaze 投影点放大的 v5。v5 仅生成了示例输入，未做模型调用，也未替换主流程。单点只能说明测得的位置；未经校准不能把任意大小的圆称为注视不确定区域。

## 当前像素表征

从此前已冻结的 `projected2d_routed_rgb` 搜索计划，读取每个 P 面板对应的最近一个历史时间箱的 gaze 路由窗口。只有该时段有至少 0.1 秒支持且时钟可用时才绘制。窗口是此前按 gaze 支持选出的固定视图分配区域，边长约为图像的 40%，不是目标检测框、精确注视区域、置信区间或对象标注。主图上的四个绿色短角标表示“可以重点检查这个区域”；区域之外仍可选，不能把标记、框中心直接作为输出。每个时刻独立提供一张完整 768×768 RGB 主图，没有四宫格、裁剪或热力图。

示例：[scene2/482 P1：原图加区域角标](../.local-sync/2d_gaze_region_presentation_20261109/inputs/scene2_482/P1/GAZE_REGION.png)，[对应未加标记的原图](../.local-sync/projected2d_main_rgb32_20261109/inputs/scene2_482/P1/MAIN_RGB_768.png)。角标会遮挡其周边少量像素，不能保证不遮挡小对象。60 张角标图片逐像素检查显示：区域内部未修改，每张图最多改动 1,896/589,824 像素，约 0.32%。这不是像素级 referent 遮挡率的评估。

## 10 交互配对结果

**补充核对：原始 8B 冻结预测在完全相同的 10 条交互上**，用同一清单、原 2D evaluator 重新评分，Temporal / Point@100 / Joint@100 F1 = 0.8077 / 0.2308 / 0.2308，Joint Exact@100 = 1/10；原始主图 768 开发协议则是 0.6531 / 0.0816 / 0.0000、Exact=0/10。两组物理预测数为 25 和 22。此处原始结果是预测回放，未重新运行原始 wrapper；原始版本提供了来自标注的 expected_count 并含 gaze 局部放大，本轮逐 mention 主图未提供该数量，两者不是独立 gaze 单因素对照。

真实 gaze 区域、镜像区域、无标记的 Joint@100 分别为 0.1154、0.1154、0.0769，均低于同 10 条原始发布预测的 0.2308。原因贡献不能仅凭此对照确定：图片、局部视觉细节、提示、输出任务单位和部件定位规则均不同。此前没有先建立原 wrapper 当前环境端到端复现，也没有证明当前逐 mention 主图恢复原水平；不能把角标实验当成原基线上的 gaze 增量验证。机器记录：[同10条基线比较](../.local-sync/gaze_region10_original_baseline_comparison_20261109/comparison.json)。

使用同一批已见开发样本、29 个 mention，三种条件的 prompt、时间提示、帧数和原 2D evaluator 相同；只改变 RGB 中的角标：

| 输入 | 选帧 P1/P2/P3 | Temporal F1 | Point@100 F1 | Joint@100 F1 | Joint Exact@100 |
|---|---|---:|---:|---:|---:|
| 无标记 | 17/3/5 | 0.6154 | 0.1538 | 0.0769 | 0/10 |
| 真实 gaze 区域 | 17/3/5 | 0.6154 | 0.1538 | 0.1154 | 0/10 |
| 左右镜像 gaze 区域 | 19/3/3 | 0.6154 | 0.1538 | 0.1154 | 0/10 |

三个条件各 10 次调用均完成并通过 vision forward 验证；实际 processor 的完整 input_ids、pixel_values 和 image_grid_thw 逐次独立重建一致。真实及镜像区域在全部 30 个源帧上不同。真实区域与无标记的 25 个可配对单点相比：0 个换帧，25 个坐标发生变化；镜像区域有 2 个换帧、20 个坐标变化。真实区域的 29 条 mention 解释中有 6 条出现 gaze、region、green 等相关词，镜像条件有 12 条；关键词不能证明理解或实际视觉观察正确。

真实区域与镜像区域的 Joint@100 相同，不能确认真实 gaze 带来独立收益。角标可影响模型输出，但当前没有证明它能让模型可靠地建立“当前 referent—对应注视区域—对应时刻”的关系。0/10 的 exact 使用原投影代理及其匹配规则，不能当作已验证的语义对象身份 exact。样本经反复开发，不是独立测试；不能根据这组数字推断泛化，也不能把不同协议的历史基线分数作为同一实验配对。

## 继续方向

在同一主图上研究更易理解且少遮挡的区域表达，例如标注视线历史对应的少量轮廓或边界，而非把模型必须解码的伪彩热图当作目标证据。区域应明确时间来源、有效支持和缺失回退，不将设定的显示半径解释为校准精度。先独立验证模型能否辨识区域及其中的对象，再在固定输入、真实/错位/无区域三条件下逐 referent 验证帧选择和原 2D 指标；必要时改进脚本里的区域分配。不能为了降低 P1 比例强制分帧。

实现与审计：[probe_2d_gaze_region_presentation.py](../tools/probe_2d_gaze_region_presentation.py)、[输入逐像素审计](../.local-sync/2d_gaze_region_presentation_20261109/input_image_audit.json)、[结果](../.local-sync/2d_gaze_region_presentation_20261109/summary.json)、[processor 重建](../.local-sync/2d_gaze_region_presentation_20261109/processor_reconstruction_verification.json)。31 项相关回归测试通过。模型进程已结束。

```bash
source /ai/data/TriModal-Referring/.local-sync/reconstructed_pilot100/env.sh
cd /ai/data/MultiAgent
PYTHONPATH=. python tools/probe_2d_gaze_region_presentation.py prepare --output /path/to/new-run
PYTHONPATH=. python tools/probe_2d_gaze_region_presentation.py infer --output /path/to/new-run
PYTHONPATH=. python tools/probe_2d_gaze_region_presentation.py report --output /path/to/new-run
PYTHONPATH=. OMP_NUM_THREADS=2 python tools/verify_referent_frame_inputs.py /path/to/new-run
```
