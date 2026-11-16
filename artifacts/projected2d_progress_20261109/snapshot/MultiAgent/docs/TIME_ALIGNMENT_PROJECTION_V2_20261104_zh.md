> **2026-11-05 后续更新**：审计、实验图像准备和独立 PTS 诊断重投影已共用注意时钟选择；审计只保存该交互的标注 referent 投影，标签在拟合结束后加载。本文下方的“全部场景 anchor”和旧测试/产物路径是 11 月 4 日的历史状态。当前实现及验证见 [时钟入口统一与 referent 投影验证](TIME_ALIGNMENT_REFERENT_AUDIT_20261105_zh.md)。

时间对齐与投影 v2：本机修复与验证（2026-11-04）

已统一相机姿态与投影基础计算，并修复源帧时间绑定和注意图使用不可靠时钟的问题。代码在 `fix/time-projection-20261104` 分支的本地工作区；没有训练、模型权重加载或付费API调用。

**修复了什么**

- `camera_geometry.py`：时钟拟合、gaze插值和anchor投影共用位置线性插值、归一化xyzw四元数与SLERP。明确vertical FOV、方形像素、+z前向、左上UV；画外UV保留原值，视锥内也只标记遮挡未知。
- 非有限时间/位姿、零四元数、无效FOV、边界外查询和超过0.25秒的采样缺口不会生成可用参考姿态。单个相机样本只支持精确同一时刻。
- `video_timing.py`：每个解码帧必须有有限且严格递增的真实PTS；拒绝缺失、重复时间戳，避免过滤后帧号错位。帧号必须是合法非负整数且与PTS表一致；负索引、布尔索引和NaN PTS不再通过。
- `frame_pose_alignment.py`：候选还必须保留原时钟支持的至少80%有效验证对，避免通过丢掉困难帧得到较低残差；该比例是工程门槛，不是统计置信度。offset扫描复用只读误差缓存，训练/反向时间块扫描不重复计算相同offset。
- `gaze_evidence.py`：核宽改用当前参考相机FOV；观察窗口与参考姿态必须使用同一遥测查询时间。输出保存参考姿态、FOV、核宽和geometry_version。
- `projected2d_attention_study.py`：准备图片前统一核对源JPEG哈希、实际帧PTS、查询时刻与offset等式；同时支持精确JPEG重放和已核对的顺序解码帧两种绑定。缺少时钟支持时保留RGB并记录回退。

候选被拒绝时，原offset仍可用于诊断重投影；`POSE_AVAILABLE`只表示有几何姿态，不表示时钟已校准。注意图要求候选通过检查，或原时钟有RGB运动支持证据；原时钟也不确定时，不生成强gaze热点。时钟、FOV和注意图的改变均记录为 `camera_geometry_v2`，后续模型运行应使用新输出目录和缓存命名空间。

归档GT、原evaluator和已有预测没有修改。现有并行PTS重投影manifest仍是独立诊断标签，不能替代原主评测标注或证明绝对像素正确。

**测试与真实输入验证**

新增20项回归测试，覆盖缺失/重复PTS、变帧率及非零PTS起点、源图/时钟契约错配、非有限时间、采样缺口、变FOV、方向与有限点区别、共享投影一致性和验证覆盖率。合成的200ms RGB/遥测偏移可在5ms测试容差内恢复，未使用场景GT。

所选125项测试：124通过、1跳过、0失败/错误。跳过项需要当前缺失的原始二维manifest。Python语法及两个CLI帮助入口通过。测试过程阻止网络连接，模型后端使用已有mock。

真实CPU验证使用重建pilot100中的固定三个样本，跨scene1、scene4和scene5；每条取4张真实帧。输入不是原机冻结CSV的逐字节恢复，所有检查均以本机原始视频/遥测为准。拟合仅使用RGB背景特征与camera位姿/FOV，未用GT或gaze选offset；拟合结束后才投影全部场景anchor和gaze作诊断。

| 样本（row_index） | 原 offset / 候选（s） | 同一验证对残差中位数（px） | 处理 |
| --- | --- | --- | --- |
| scene1/162 | 2.052209 / 2.115 | 22.55 → 26.80（14 对） | 拒绝候选，保留原时钟诊断投影 |
| scene4_room2/33 | 2.219330 / 0.998 | 16.03 → 4.91（6 对） | 拒绝候选，保留原时钟诊断投影 |
| scene5/28 | 2.069154 / 1.523 | 15.26 → 3.91（7 对） | 接受候选，仍非绝对标定 |

表中是同一组验证特征对、最长边640像素下的背景旋转残差；不是源图GT定位误差，不是F1，也不是三条样本的全量性能。scene4_room2/33虽然残差下降，因训练/验证时间块不一致而拒绝；scene1/162还有高残差和回归。scene5/28通过的是背景运动一致性，曝光时间、真实渲染相机内外参、有限深度视差和遮挡仍未校准。

12/12张源图的JPEG哈希和帧号—PTS契约通过检查；8张能在合法采样区间内取得参考姿态，只有scene5的3张帧得到时钟支持的注意图，其余保留RGB。起始黑屏/遥测范围外的帧没有外推补姿态，也未从分母删除。

**本机产物与复查入口**

- 最终检查目录：`/ai/data/MultiAgent/.local-sync/alignment_projection_v2_20261104_final/`。
- `summary.json`、`engineering_results.json`：三条真实输入的完整状态与残差摘要。
- `pose_alignment/<scene>/<row>/clock_fit.json`：完整offset曲线、训练/验证覆盖、拒绝原因。
- `frame_pose_contract.json`、`video_frame_pose_timeline.csv`：实际PTS、遥测查询、插值两侧与回退来源。
- `events/<scene>/<row>/audit.json`：全部场景anchor投影、gaze原始投影及RGB回退；`Vxxx_projection.jpg`中青色十字为全部场景anchor，红色为支持时钟下的注意图。
- `regression.json`、`regression.txt`、`source_snapshot.json`：测试结果与当前代码SHA-256。

新增入口 `python -m MultiAgent.alignment_projection_audit` 不依赖丢失的原二维manifest/源帧，可直接读取事件manifest执行小型CPU诊断；默认只检查2条，每条4帧，必须指定新输出目录。它没有推理选项，不构造模型后端。

```bash
source /ai/data/TriModal-Referring/.local-sync/reconstructed_pilot100/env.sh
cd /ai/data/MultiAgent
python -m MultiAgent.alignment_projection_audit \
  --manifest .local-sync/alignment_v2_samples.json \
  --output .local-sync/alignment_projection_v2_new_check \
  --limit 3 --frames 4

python -m unittest MultiAgent.test_alignment_projection \
  MultiAgent.test_frame_pose_alignment MultiAgent.test_gaze_evidence \
  MultiAgent.test_projection_diagnostics MultiAgent.test_head_motion_diagnostics -v
```

已有输出目录会拒绝覆盖。旧 `frame_pose_alignment` 的默认输出目录也改为 `runs/pts_camera_clock_v2`，避免直接覆盖历史 `runs/pts_pose_alignment_20261023`。

后续应先人工核对新投影图、区分时钟误差与真实渲染相机/视差问题，再在固定输入和视觉预算下比较注意呈现。当前没有任何模型精度提升结论。
