# Projected-2D 投影、时钟与指代表达复核（2026-10-23）

当前结论：用户指出的问题有实际依据。已发现旧报告省略画外 GT、RGB 与相机遥测的时钟配对存在明显疑点、Visual 输入缺少对象类型及共指字段。本轮完成独立投影复核、无 GT 的图像运动时钟诊断、语言协议修复和两条真实本地 Qwen3-VL-8B 检查。尚未完成像素投影校准；旧四组指标不能证明 heatmap/foveation 有效或无效。

后续复核已恢复全部 60 张源 JPEG 的实际 PTS，并完成一轮基于真实 PTS 的固定 20 条时钟重拟合，见 [头动偏差与源帧 PTS 复核](HEAD_MOTION_ALIGNMENT_OPTIMIZATION_20261023_zh.md)。13 条获得 `VALIDATED_CANDIDATE`；7 条候选仍为 `CLOCK_UNCERTAIN`，但不再因此禁止新投影：这些事件使用原始 offset 作为 `ORIGINAL_CLOCK_FALLBACK_PTS_POSE`，将实际帧 PTS 与插值姿态接入新点生成。原抽帧请求时间与真实保存帧 PTS 的中位差约 21.1 ms。PTS 姿态尚未写入热图、GT manifest 或模型评价。

## 1. GT 为什么看起来始终在画面中

GT 来自各场景 anchor table 的世界坐标。原构建器已有相机变换和画外判定，并未把 GT 强行裁剪进图像。但旧可视化只绘制 `projection_valid=True` 的项目，没有在图注中列出被省略的目标，这种呈现不足以判断相机运动下的 GT 状态。报告现已逐帧列出全部目标及画外状态，旧图已归档。

独立用 SciPy 标准化四元数复算固定 20 条的 153 个目标—帧记录：

|投影状态|记录数|含义|
|---|---:|---|
|`in_frustum_visibility_unknown`|134|世界 anchor 投影进入视锥；实际遮挡未知|
|`outside_frame`|19|保留原始越界 UV，不在图中边缘伪造点|
|`behind_camera`|0|此子集中未出现；代码和测试覆盖背面情况|

全部 153 条的视锥判断与原 manifest 一致。复建 GT 的像素/状态也与原 manifest 相同。这证明复用了原协议，不能证明原时钟或渲染相机已经准确标定。

投影采用 `p_camera = inverse(R_camera_to_world) * (GT_world - C_world)`，+z 为相机前方。由记录的 vertical FOV 和图像高度求焦距，按宽高比映射到左上角为原点的 UV，检查正深度及 UV 边界。相机移动和转动都可能使 GT 出画，单元测试覆盖这些情况。

`gazePoint` 是独立的有限世界交点，`gazeVector` 是方向；它们用于注视线索和坐标一致性交叉检查，不能替代 GT 世界位置，也不能恢复整张图的深度。判断 GT 视锥位置只需 GT、相机位姿和内参。判断是否被遮挡、物体是否部分可见，还需要深度/mesh/物体范围；本轮没有这些条件，因此不能把 `in_frustum` 标为确认可见。

八个场景的当前 anchor table 与原仓库表逐项相同，详见 [anchor_table_parity.json](../runs/projected2d_attention_qwen8b_20261022_int8/diagnostics/geometry/anchor_table_parity.json)。这不等于动态物体的位置、所有 anchor 标注语义或屏幕录制相机均已验证。

[153 条投影审计](../runs/projected2d_attention_qwen8b_20261022_int8/diagnostics/geometry/frustum_audit.json)；[scene4_room2/145 逐帧画外状态图](../runs/projected2d_attention_qwen8b_20261022_int8/diagnostics/geometry/scene4_room2_145_frustum_audit.jpg)。

## 2. 时钟问题有多大

旧 offset 来源于原方法的音频/文件时间规则，并不是已校准的 RGB 曝光—遥测时钟。仅验证四元数方向和 gazePoint/cameraHitPoint 数值一致，无法证明当前 RGB 对应了正确的相机姿态。

新增 CPU 检查从真实视频按约 0.5 秒间隔抽取图像，在最长边 640 像素上用 SIFT、匹配筛选和 RANSAC 提取相对稳定的特征。根据记录的相机旋转和 FOV 预测图像运动，扫描时钟 offset。偶数特征对选参，奇数特征对保留作检查；下表在原规则和候选规则共同可评价的保留特征对上比较。固定 FOV 的结果只改变时钟因素。

时间定义为 `video_seconds = telemetry_seconds + offset_seconds`。

|事件|旧 offset（s）|候选 offset（s）|旧运动残差（px）|候选运动残差（px）|
|---|---:|---:|---:|---:|
|scene4_room1/170|3.492|1.950|186.85|32.68|
|scene4_room2/145|2.961|1.570|84.50|18.75|
|scene2/139|1.597|0.470|17.48|8.49|

残差为每对图像特征误差的中位数，再对共同保留特征对取中位数；单位是缩到最长边 640 后的像素，不是原图的 GT 定位误差。搜索和选参不读取 GT，也不使用 gaze。扩展至固定 20 条后，20 条均产出诊断：16 条候选残差下降、4 条上升；事件残差中位数从 40.10 降至 14.30 像素。上升样本为 scene5/583、scene1/366、scene4_room2/87、scene5/4，均保留。这说明不能统一改一个 offset，也不能自动采用所有拟合值。完整逐样本结果见 [motion_alignment_summary.json](../runs/projected2d_attention_qwen8b_20261022_int8/diagnostics/geometry/motion_alignment_summary.json)。

这些结果支持旧配对有明显问题，但只转动模型忽略平移/视差，静态特征假设也可能失败，剩余残差仍然明显。相邻特征对的保留检查也不能替代独立序列上的标定验证。不能把候选 offset 当成准确的相机曝光时间，更不能按 GT 调整候选直到十字“贴上去”。

另检查了用视频音轨与独立 WAV 作互相关的可行性：scene4_room2/145 的 MP4 音轨解码到 8 kHz 后全部为零，独立录音有信号，无法由此建立同步。详见 [音频同步可行性记录](../runs/projected2d_attention_qwen8b_20261022_int8/diagnostics/geometry/scene4_room2_145_audio_sync_feasibility.json)。工作区未找到采集端相机 C# 实现或完整渲染投影矩阵，当前只使用已记录的位姿与 FOV。

候选 offset 没有应用到模型输入、热图或 evaluator。修改 FOV 的额外搜索也仅存为诊断，没有混入四组比较。三条检查视频的实际 FPS 约 24，不能默认所有视频为 30 FPS。

[同一 RGB 下旧时钟与候选时钟的投影对照](../runs/projected2d_attention_qwen8b_20261022_int8/diagnostics/geometry/scene4_room2_145_clock_hypothesis.jpg)：上排旧时钟把人物 anchor 画在墙上；下排候选配对后更接近左侧人物，后两帧人物 anchor 正常出画。喷瓶等点仍有偏差，说明校准尚未完成。这里的 GT 只在无 GT 拟合结束后用于诊断作图。

## 3. 模型此前是否明确知道寻找什么

存在输入协议问题，已修复：

- 旧 Visual prompt 仅保留 mention 文本、span、kind、role、relation、count，丢失 `object_type / component / parent_text / anchor_text / coreference_id / parent_mention_id / coreference_note`。新版传递语言解析结果，明确任务中的目标类型、上下文和共指关系。
- 旧 `scene1/705` 解析遗漏 “the other person”，并把空间表达里的 “the side building” 当作部件父项抑制。新增高置信显式 actor 漏项检查，触发原有模型重试；嵌套表达抑制改为要求实际 component 证据。
- `it` 的合法显式前件曾被错误当作不在自身子串内的 parent 而拒绝。仅在原文前件匹配且 coreference_id 相同的条件下，将该链接保存为 `parent_mention_id`，不按 GT 猜前件。
- 增加 `fellow → person` 的词义归一，候选仍只从当前 scene context 读取。保留跨场景隔离测试。
- 一个过渡版本 v2 的真实模型返回照抄了示例 `[0.5,0.5]`，并漏掉其余 mention。该失败保存原样。当前 v3 改为列出本次全部真实 mention ID 的输出骨架，不提供默认数值点，并明确 gaze crop 到源帧坐标的转换。

提示词明确：定位目标实体，不能把热区当成答案；模糊周边不等于物体不存在；图像左侧为完整源帧，右侧为同帧 crop；所有点必须转换到源帧归一化坐标。原始完整指令和每个 mention 的语言字段都保留。输入不含 GT、anchor 答案列表或标注数量。

固定 20 条旧解析中有 4 条上游失败，1 条包含高置信 actor 漏项，9 条指令有动作代词。没有人工 mention 级 GT，本轮不报告 mention accuracy，也不将这个窄覆盖检查说成完整解析质量评价。

## 4. 已实际运行的本地模型检查

保留正在运行的推理服务，未停止、未重启。使用本地 `Qwen3-VL-8B-Instruct` checkpoint，INT8 权重、BF16 运算输入。两条检查采用旧实验完全相同的 RGB 图片，并验证文件 SHA-256；本轮只检查新 Referential 与 Visual 语言协议，未增加热图或修改几何，也未运行 Hand/Coordinator。

|样本|新版表达任务|Visual 返回|仍存在的问题|
|---|---|---|---|
|scene1/705|5 个 mention，含 person、truck、空间表达、it、building|5 个条目完整返回；it 与 truck 共享预测点|把黄色 forklift 当成 truck；共指内部一致不代表实体正确|
|scene4_room2/145|3 个 mention：person、spray bottle、rack|3 个条目完整返回；person 的 P1 点为 `[0.085,0.405]`|漏看 spray bottle；rack 点仍不精确|

这两条只说明任务字段送达、格式完整性改善，并显示剩余识别问题。它们不能证明 heatmap/foveation 提升，不能证明二维定位准确率提高，也不能泛化到完整开发集。

真实请求、prompt、原始响应、processor tensor hash 和视觉编码器输入消费校验均保存在 [v3 probe 目录](../runs/projected2d_attention_qwen8b_20261022_int8/diagnostics/referential/local_qwen8b_contract_v3_probe/)。[probe_review.json](../runs/projected2d_attention_qwen8b_20261022_int8/diagnostics/referential/local_qwen8b_contract_v3_probe/probe_review.json) 中两条 `model_consumption_verified` 均为 true。

- [scene1/705 原始 RGB 与新版预测（无 GT）](../runs/projected2d_attention_qwen8b_20261022_int8/diagnostics/referential/local_qwen8b_contract_v3_probe/scene1_705_predictions_no_gt.jpg)
- [scene4_room2/145 原始 RGB 与新版预测（无 GT）](../runs/projected2d_attention_qwen8b_20261022_int8/diagnostics/referential/local_qwen8b_contract_v3_probe/scene4_room2_145_predictions_no_gt.jpg)
- [scene4_room2/145 完整实际 Visual prompt](../runs/projected2d_attention_qwen8b_20261022_int8/diagnostics/referential/local_qwen8b_contract_v3_probe/samples/scene4_room2/145/visual_prompt.txt)
- [scene4_room2/145 完整响应及消费校验](../runs/projected2d_attention_qwen8b_20261022_int8/diagnostics/referential/local_qwen8b_contract_v3_probe/samples/scene4_room2/145/RGB.json)

旧 pilot20 的真实输入审计有 64 个成功 Visual 输入记录，全部通过视觉编码器消费校验；处理组相对 RGB 的 48 个像素 tensor hash 均不同。[原始输入审计](../runs/projected2d_attention_qwen8b_20261022_int8/pilot20/report/actual_model_inputs/provenance.json) 能反驳“处理图从未送入模型”这一具体可能，不能证明模型学会按研究预期使用它们。

检查沿用原二维协议的 3 个 evidence panel，只为冻结图像、隔离语言修复因素；不宣称它们足以覆盖多指代视频。没有把本轮检查混入旧四组指标。

## 5. 代码和结果边界

新增 `projection_diagnostics.py / camera_motion_alignment.py / referential_diagnostics.py`，修改 `mention_mapping.py / agents.py / projected2d_attention_study.py / projected2d_study_report.py` 并增加回归测试。原 evaluator、物理 referent 匹配规则和历史预测未改。新版实验使用独立 `_contract_v3` 目录和缓存 namespace，防止重放旧 Visual 输入。

已运行：独立 153 行 GT 视锥审计、8 场景表一致性、固定 20 条真实 RGB 运动诊断、两条本地 8B 协议检查、相关 66 项测试。所有旧失败与过渡版本失败均保留。

尚未完成/NOT_RUN：确认真实屏幕录制相机内参及完整像素校准；遮挡/物体范围验证；完成校准后的 smoke4 和固定 pilot20 四组重跑；BF16 权重版本；完整 Hand/Coordinator 系统重跑。旧四组指标加 `PROVISIONAL_GEOMETRY_NOT_CALIBRATED` 状态，不据此认定注意引导无效。

下一轮顺序：先依据不含 GT 的 RGB—相机运动检查确定时钟的可用性和不确定度，再核对内参/视差等剩余误差；确认给定表达的原始 RGB 定位协议后，冻结同一批解析、源帧和解码设置，重做 RGB、当前热图、历史热图、历史 foveation 四组。时钟和语言修复同时存在时，应分别留出对照，避免把修复收益误记为热图收益。

## 6. 可复查命令

```bash
cd /media/kunyu/16TB/spj/Multiagent
export PATH=/home/kunyu/miniconda3/envs/trimodal/bin:$PATH
export SPJ_PREPARED_DATA_ROOT=/media/kunyu/16TB/spj/V3dMD/prepared/data
python -m MultiAgent.projection_diagnostics runs/projected2d_attention_qwen8b_20261022_int8
python -m MultiAgent.camera_motion_alignment runs/projected2d_attention_qwen8b_20261022_int8 --all --reuse-existing
python -m MultiAgent.referential_diagnostics runs/projected2d_attention_qwen8b_20261022_int8 --render-probe
python -m unittest MultiAgent.test_projected2d_attention_study MultiAgent.test_projection_diagnostics MultiAgent.test_referential_retry MultiAgent.test_context_isolation MultiAgent.test_local_video_visual MultiAgent.test_gaze_evidence MultiAgent.test_qwen_backend
```

真实模型检查的命令为 `python -m MultiAgent.referential_diagnostics runs/projected2d_attention_qwen8b_20261022_int8 --probe-events scene1:705 scene4_room2:145`。它会加载本地 8B；上面的 CPU 命令不会启动推理服务。原始数据未改，已有服务保持运行。
