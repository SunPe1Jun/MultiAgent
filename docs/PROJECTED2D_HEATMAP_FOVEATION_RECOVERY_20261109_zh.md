# 2D 热图驱动 Foveation 首次恢复

> 该轮单主图诊断已被后续原 wrapper 复现实验取代。它使用另一种 referent 协议，不能回答当前主线是否涨点。当前 paired-image 方法与结果见 [原基线视觉迭代](PROJECTED2D_ORIGINAL_BASELINE_VISUAL_ITERATION_20261109_zh.md)。

用户明确的研究目标是：在原 2D setting 中，结合 gaze 热图与 foveation 改善视觉处理，再评价 2D 指代结果。此前区域角标只是另一种提示，既没有用热图驱动 foveation，也没有实现该方法，不能称为方法成功。已有时序/相机对齐、投影一致性与 GT 代理核查工作保留；这些不等价于 GT 绝对位置已验证或视觉方法已改善。

本次恢复一个可核查的原 P 面板 2D 探针：每个 768×768 RGB 主图，在其原时间点从 gaze 遥测生成过去两秒、三阶段的时长热图；脚本利用热图驱动已有的多焦点 foveation。模型收到一张完整经过处理的 RGB 帧，不收到单独的热图、区域角标或 3D 候选。全图仍保留，当前算法在热图支持较强的位置保留高保真，外围与降采样图层混合；不是源分辨率超分辨，无法恢复原图已经缩放损失的细节。时钟/姿态无支持或热图为空时逐帧回退原图。本探针延续逐 referent 任务和软时间提示，三个条件模型 prompt、帧和原 2D evaluator 一致，不使用标签构造热图或输出任务。

条件为原 RGB、真实 gaze 热图驱动 foveation、横向镜像同一热图驱动 foveation。相同 prompt/图像/生成设置的 RGB 冻结预测直接复用 10 条，真实和镜像各新调用 10 条。该设置使用原 2D 的 P 面板和评价器，但不等同于原发布基线：原发布流程另有 GT expected_count 和 gaze 局部放大；此处遵照用户认可的逐 mention 及单主图约束。不能将本结果称为原始 wrapper 的单因素实验或直接声称与其等价。

## 开发样本结果

10 个已见开发交互、29 个 mention，共 24 个定位点，30 条逻辑记录（20 次新模型调用，10 条严格核验后复用）：

| 条件 | Temporal F1 | Point@100 F1 | Joint@100 F1 | Joint Exact@100 | P1/P2/P3 |
|---|---:|---:|---:|---:|---|
| RGB | 0.6275 | 0.1569 | 0.0784 | 0/10 | 18/3/3 |
| 真实热图 + foveation | 0.6275 | 0.1176 | 0.0000 | 0/10 | 18/3/3 |
| 镜像热图 + foveation | 0.6275 | 0.1569 | 0.0784 | 0/10 | 18/3/3 |

真实条件相对于 RGB 24 对有效单点没有换帧，20 对坐标变化；Joint@100 命中数从 2 降到 0。当前没观察到收益，也没有恢复用户原始 2D 水平。样本已使用过且很小，不能据此断言热图与 foveation 方法无效。当前结果无法单独归因于时间、焦点宽度、背景降质或对象识别等任何一项。

30/30 帧有时钟及 gaze 支持，输出真实与镜像处理均不同；30 个三阶段热图、输入及预测哈希均通过核验；30/30 逻辑请求的 processor tensor、帧顺序及 vision forward 核验通过。真实处理平均修改约 88.26% 的图像像素，来自连续的焦点和外围混合，不等于 88% 区域被遮挡；焦点场值至少 0.8 的像素平均占 0.20%，显示该既有参数的高保真焦点较小。这是值得后续检查的参数，不是已经证明的失败原因。无 gaze 的精确回退和多焦点保持由两项既有 foveation 测试通过；相关对齐与主图单元测试 34 项通过。

示例：[原 RGB](../.local-sync/projected2d_main_rgb32_20261109/inputs/scene2_482/P1/MAIN_RGB_768.png)、[热图驱动 foveation](../.local-sync/2d_heatmap_foveation_recovery_20261109/inputs/scene2_482/P1/HEATMAP_FOVEATION.png)。每个 panel 的热图以 `gaze_history_heatmap.npz` 保存，仅用于脚本处理和核验。

下一步应先以固定任务合同针对现有焦点占比和外围细节做受控实验，同时核查逐 referent 时间证据和原源分辨率细节损失；随后再用固定的新交互验证。扩大样本前不能挑选本轮最优阈值后宣称有效，也不能用模型识别出热图/角标替代 2D referent 评价。原 2D GT 投影和时间对齐仍存在外部绝对校准、遮挡和动态对象等限制。

实现：[probe_2d_heatmap_foveation_recovery.py](../tools/probe_2d_heatmap_foveation_recovery.py)；[输入审计](../.local-sync/2d_heatmap_foveation_recovery_20261109/input_image_audit.json)；[评分](../.local-sync/2d_heatmap_foveation_recovery_20261109/summary.json)；[RGB 复用审计](../.local-sync/2d_heatmap_foveation_recovery_20261109/rgb_reuse_audit.json)；[processor 重建](../.local-sync/2d_heatmap_foveation_recovery_20261109/processor_reconstruction_verification.json)。
