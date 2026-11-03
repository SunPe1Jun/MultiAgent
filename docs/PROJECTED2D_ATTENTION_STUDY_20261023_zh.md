# Projected-2D 视觉引导配对实验

> 2026-10-23 复核：下文四组结果标记为 **PROVISIONAL_GEOMETRY_NOT_CALIBRATED**。已发现 RGB/遥测时钟配对疑点及表达协议缺项，旧指标不能证明 heatmap/foveation 有效或无效。最新事实、修复和真实模型检查见 [投影、时钟与指代表达复核](PROJECTED2D_GEOMETRY_REVIEW_20261023_zh.md)。下文记录历史实验设置，不代表已经完成新版校准实验。

本轮沿用 `python -m MultiAgent.run_v3`，增加 `--projected2d-study` 诊断入口。
既有 anchor 预测与 Coordinator 主流程保留。新增诊断调用原 ReferentialAgent，冻结其 mention，
由 Visual 输出每个 mention 的 `status / predictions(frame_id, point_uv) / evidence`。
适配器把点输出转换成原二维 `referents` JSON，调用原 evaluator，未修改匹配规则。
此诊断没有调用 Hand 或 Coordinator，不将二维定位指标解释为完整 agent 系统的 anchor 选择效果。

## 复用和变更

复用原仓库 `exam2/build_2d_eval_manifest.py` 的 evidence panel 选帧、时钟 offset 和 GT 投影；
复用 `exam2/run_qwen3vl_2d_point_grounding.py` 的 `multi_image + paired_crop` 图像组织；
复用 `exam2/evaluate_2d_point_grounding.py` 的 Temporal、Point@50/100/150/200、Joint@50/100/150/200。
距离阈值以原始源帧像素计算，保持原协议。

四组共用指令、冻结 mention、三张原协议 evidence panel、裁剪框、图片尺寸、token 预算和解码设置。
三帧是复用原二维任务的 panel 定义，不是对多指代视频的充分采样结论，也不是先前 34–100 帧实验的延续配置。
RGB 已包含原方法的 gaze crop，因此热图增益是在已有 gaze crop 上测得。

|组|视觉处理|
|---|---|
|RGB|原始 RGB 内容，统一中间编码，再按原方法拼接全图与 gaze crop|
|CURRENT_HEATMAP|距参考时刻不超过 0.18 秒的最近有效 gaze，重投影到参考相机后叠加|
|HISTORY_HEATMAP|参考时刻前 2 秒的 gazePoint 重投影到参考相机，按有效持续时间累加|
|HISTORY_FOVEATION|使用与历史热图完全相同的数值图，平滑弱化周边，至少保留 0.55 原始 RGB 成分|

热图空间核为 2 度，显示映射 `1-exp(-2H)`，透明度上限 0.28；
实际透明度取决于持续时间权重，不能把上限当成实际强度。
Foveation 使用 2/4/8 倍降采样再上采样的多尺度混合；无有效注视时保留 RGB。
所有组采用相同 JPEG 编码阶段，避免额外压缩差异混入对照。

## 模型与服务

保留已有推理服务，未停止或重启。现有服务提供的模型与指定 Qwen3-VL-8B 不同，故实验使用本地指定 checkpoint。
为共享剩余显存，四组统一使用 bitsandbytes INT8 权重及 BF16 运算输入，模型 checkpoint 不变。
结果只与本轮同设置 RGB 比较，不与历史 BF16 或其他模型结果直接比较。
该量化设置是本轮资源约束，不应将其结果表述为完整精度原方法复现。

## 标注与 GT 边界

当前可复用的二维标注为原 manifest 中的事件 referent、3D anchor 投影和 acceptable panel。
未找到独立人工逐 mention 的二维像素点与时刻标注。新增 mention 来自指令解析，不冒充人工 GT。
原 wrapper 的 `expected_count` 来自标注；本轮去掉该输入，避免给定表达实验向模型暴露答案数量。
因此本轮是“给定表达、无标注数量”的原二维协议扩展，不宣称逐字复现旧 wrapper。
模型输入不含 anchor ID、GT 坐标、GT 名称列表或答案映射。

固定样本由原 manifest 按 scene 分层随机抽取，seed=20261022，要求媒体存在且有可评价投影；
这种可评价性筛选需要在报告中说明。GT 仅用于筛选可评价记录、构建评价 manifest 与推理后作图，
不用于选帧、构建注视图或预测。
适配器直接展开 mention 的全部点，不用 GT 合并重复指代。原 evaluator 按物理 referent 匹配，
重复 mention 可能带来额外预测惩罚，这是本轮兼容接口的限制。

## 几何与时间

数值一致性审计支持 `gazePoint` 是有限世界交点、`gazeVector` 是世界方向，
相机四元数为 xyzw camera-to-world；世界点先减相机位置再逆旋转，方向仅逆旋转。
相机采用记录的 vertical FOV、方形像素 pinhole 假设、左上角原点。
历史 gaze 对每张参考图重新投影，不累计不同相机下的旧像素坐标。
无可靠深度/mesh，不能验证遮挡或移动物体的跨时刻对应。
时钟 offset 复用原方法，坐标一致性不能证明时钟正确。
当前 gaze 可使用参考时刻之后容差内的最近观测；历史窗口止于参考时刻。整体实验按离线结果报告。
P1/P2/P3 用源 JPEG 路径、SHA-256 和请求 seek 时间追溯；原 extractor 未记录真实解码 PTS/frame number。

## 留痕位置

历史未校准实验目录：`runs/projected2d_attention_qwen8b_20261022_int8/`。

- `fixed_samples.json`、`config.json`：固定样本与配置。
- `inputs/<scene>/<row>/input_audit.json`：源帧、时间、裁剪框、四组图片路径与 SHA-256。
- `geometry_provenance.json`：参考相机位姿、FOV 内参、原 gaze 时间戳与逐点重投影、注意图重算校验。
- `coordinate_audit.json`：原始字段的世界系/相机系一致性审计。
- `smoke4|pilot20/samples/`：冻结 Referential、Visual prompt、原始响应、规范化结果、失败与耗时。
- `smoke4|pilot20/cache/`：实际请求、processor token/grid、视觉编码器输入 hash、完整 processor tensor。
- `smoke4|pilot20/report/actual_model_inputs/`：从真实 input_ids 解码的完整 system/user prompt。
- `smoke4|pilot20/report/`：原指标、逐样本变化、输入改变量、实际 processor 图像、预测点与 GT 对照。

初轮发现 RGB 缺少处理组的一次中间 JPEG 编码；该版结果已保存至 `archive/pre_codec_alignment/`，
不作为最终比较。`codec_alignment_correction.json` 记录修正原因。

## 运行

```bash
cd /media/kunyu/16TB/spj/Multiagent
export PATH=/home/kunyu/miniconda3/envs/trimodal/bin:$PATH
export SPJ_PREPARED_DATA_ROOT=/media/kunyu/16TB/spj/V3dMD/prepared/data
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
python -m MultiAgent.run_v3 --projected2d-study --split smoke4 --quantization int8
python -m MultiAgent.run_v3 --projected2d-study --split pilot20 --quantization int8
python -m MultiAgent.audit_gaze_coordinates \
  --manifest runs/projected2d_attention_qwen8b_20261022_int8/fixed_samples.json \
  --output runs/projected2d_attention_qwen8b_20261022_int8/coordinate_audit.json
python -m MultiAgent.audit_projected2d_inputs runs/projected2d_attention_qwen8b_20261022_int8
python -m MultiAgent.projected2d_study_report runs/projected2d_attention_qwen8b_20261022_int8 --split pilot20
```

实际 prompt、input_ids、像素 hash 和 vision forward 校验可证明处理图送入视觉编码器，
不能单凭这些校验宣称模型按预期使用了注意线索。定位是否改善以固定配对指标和失败分析为准。

## 已完成结果：smoke4 → 固定 pilot20

2026-10-23 已完成两个 split；pilot20 包含 smoke4 的前四条，不是额外 20 条。
同配置、同 prompt、同图片内容哈希的调用可以重放；RGB 编码修正后不匹配的记录均已补跑。
20 条各有四组结果记录，其中 16 条四组 Visual 调用成功，共 64 次成功 Visual 调用；
另外四条在冻结的 Referential 阶段失败，四组均记为 NOT_RUN/空预测，保留在 20 条原 evaluator 分母中。
本地 8B 进程已正常结束，保留的推理服务始终没有被停止或重启。

下表均为原 evaluator 的 micro F1；K 是原始源帧中的像素距离阈值。

|输入组|Temporal F1|Point@100|Joint@100|Joint@200|相对 RGB 的 Joint@100 变化|
|---|---:|---:|---:|---:|---:|
|RGB|0.4835|0.1538|0.1099|0.2198|—|
|当前热图|0.5217|0.1087|0.0870|0.2609|-0.0229|
|历史热图|0.5714|0.1538|0.0879|0.2418|-0.0220|
|历史 foveation|0.5000|0.1087|0.0652|0.2391|-0.0447|

历史热图的 Temporal F1 提高约 8.79 个百分点，但严格空间/时空定位未改善。
在较宽松的 200px 阈值下，三种处理的 Joint 均高于 RGB；不能因此宣称精确定位已改善。
这些是开发样本上的诊断结果，不构成全量数据集结论。

## 模型是否真正接收处理

64 次成功调用都通过 vision forward 像素消耗校验与输入文件 SHA-256 校验。
16 条完整配对的实际 input_ids 和 image_grid 全部组间相同，图像像素张量均与 RGB 不同。
每次输入三张 1554×768 的全图+crop JPEG，processor 恢复为三张 1568×768 图，
实际展开 3528 个 image token，四组一致。没有把总 prompt token 当成视觉 token。

同一 prompt 明确告知：热色/亮区与更清晰区域是弱注视线索，不是目标标签；
要求依据 RGB 和动作定位，不把模糊解释成物体不存在，不输入世界坐标数值。
解码后的真实 system/user 输入保存在 `pilot20/report/actual_model_inputs/`，不只是手写 prompt 模板。

|处理|16 条完整配对中定位输出改变|Joint@100 TP 增加/减少/不变的样本数|processor 图像相对 RGB 的 MAE 中位数（0–255）|
|---|---:|---|---:|
|当前热图|15/16|1 / 2 / 13|0.0675|
|历史热图|14/16|1 / 2 / 13|0.3621|
|历史 foveation|15/16|1 / 3 / 12|2.5173|

这能排除“处理图完全没有交给模型”，但不能证明模型在内部按预期解读了 gaze。
当前热图尤其微弱：60 张源 panel 的峰值透明度中位数 0.0558，alpha>0.01 的像素占比中位数 0.21%；
历史图分别为 0.2349、1.11%。没有在看过 GT 或结果后调强热图重跑本轮。
Foveation 保留了全局原图成分，改变可见，但 token 数没有减少；共享 GPU 下的时长差异不能证明加速。

## 影响解释的具体问题

1. **继承的二维投影代理有图像对齐疑点。** `original_annotation_parity.json` 确认 153 行原 GT 像素、有效性和 status 全部一致，未更改标签。后验对照图中，例如 scene4_room2/145 P1 的 person 标记落在墙面附近，而可见人物位于左侧；scene2/139 P3 的多个投影点接近顶部、并不贴合可见目标。坐标字段数值自洽，不代表这些点与视频实际时刻/像素准确对应。尚未查明是原时钟、anchor 位置还是投影参数引起，不能把所有低分归因于模型理解失败。
2. **Referential 有现有校验限制。** scene2/592（battery/laptop）、scene2/416（its label/that file）、scene4_room3/151（its front legs/the chair）、scene4_room4/163（its left fold/the curtain fabric）因 parent_text 不在 mention 字面子串中而失败。本轮冻结它以保持四组一致，没有用 GT 修补。视觉实验中需要看到这四条共享上游失败。
3. **原标签不是人工逐 mention 像素真值。** 新输出按 mention 组织，旧 evaluator 按物理 referent 匹配；直接展开会对重复对象预测产生额外惩罚，没有添加 GT 合并规则。
4. **基线本身已有 gaze crop。** 测得的是在原方法 gaze crop 之上再加入 heatmap/foveation 的增量，并不是完全无 gaze 的 RGB 对照。
5. **三张 panel 与 8B INT8 限定本轮范围。** 没有完成 BF16 全精度复现、更多帧对照、B1 全景新增对照或完整 Hand/Coordinator 融合；本轮没有声称这些已运行。原 wrapper 的 GT expected_count 未输入，因此是遵守原评估协议的 given-expression 扩展，而不是旧提示词逐字复现。

## 真实对照案例与阅读入口

图中 cyan 为模型预测，magenta 为原 projected-anchor GT，均是推理后绘制；不带 GT 的真实输入也独立保留。

|样本（row_index）|例子|Joint@100 TP：RGB / 当前 / 历史 / foveation|
|---|---|---|
|scene2/139|历史热图与 foveation 改善；RGB 与当前图误定位|0 / 0 / 1 / 1|
|scene4_room2/145|历史图时间匹配增加，但空间匹配丢失；同时有 inherited GT 对齐疑点|1 / 1 / 0 / 1|
|scene4_room1/170|当前热图与 foveation 丢失 RGB 的严格命中|1 / 0 / 1 / 0|

- [完整指标与预算](../runs/projected2d_attention_qwen8b_20261022_int8/pilot20/report/REPORT.md)
- [scene2/139 原图—处理图—预测—GT](../runs/projected2d_attention_qwen8b_20261022_int8/pilot20/report/scene2_139_input_prediction_gt.jpg)
- [scene4_room2/145 原图—处理图—预测—GT](../runs/projected2d_attention_qwen8b_20261022_int8/pilot20/report/scene4_room2_145_input_prediction_gt.jpg)
- [scene4_room1/170 原图—处理图—预测—GT](../runs/projected2d_attention_qwen8b_20261022_int8/pilot20/report/scene4_room1_170_input_prediction_gt.jpg)
- [scene2/139 真实 processor 输入恢复图](../runs/projected2d_attention_qwen8b_20261022_int8/pilot20/report/scene2_139_actual_processor_inputs.jpg)
- [逐样本原指标](../runs/projected2d_attention_qwen8b_20261022_int8/pilot20/report/per_event_original_metrics.csv)
- [逐样本配对变化](../runs/projected2d_attention_qwen8b_20261022_int8/pilot20/report/paired_changes.csv)

CPU 最终复核为 20 事件、60 panel 的数值注意图全部精确重建。
协议/几何/backend 单元测试最终为 24 项通过。原 evaluator 和其匹配规则未改动。
