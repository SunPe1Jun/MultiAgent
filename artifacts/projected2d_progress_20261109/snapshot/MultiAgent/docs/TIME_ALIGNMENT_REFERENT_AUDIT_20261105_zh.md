# 审计/实验时钟统一与交互 referent 投影（2026-11-05）

本次统一了 CPU 审计与 projected-2D 实验的注意时钟使用，并将审计输出从全场景 anchor 改为当前交互的标注 referent。未加载模型、未调用 API；原 evaluator、主评测标注及历史结果保留。

## 时钟与姿态如何选择

`camera_geometry.aligned_frame_geometry` 是共享入口，策略 ID 为 `validated_candidate_or_rgb_supported_original_v1`。

- 输入含 `attention_camera` 时明确使用它；旧契约只有 `reference_camera` 时兼容后者。选定姿态不可用时保留不可用状态，不悄悄换成另一时钟的姿态。
- 候选通过 RGB 背景运动验证时，注意图和本次诊断主投影使用候选姿态；候选未通过时使用原时钟姿态，注意图还需要原时钟有 RGB 运动支持。时钟不可靠时输出 RGB 回退；可用但未经标定的原姿态仍可用于诊断投影。
- 验证源图 SHA-256、真实 PTS、telemetry query 与 offset 等式；姿态不可用时也保存选定时钟的查询时间。拒绝候选/纯诊断姿态不能凭残留的 supported 标志开启注意图。
- `alignment_projection_audit.audit_context` 和 `projected2d_attention_study.make_condition_images` 都直接调用此入口。输出实际使用的 offset、时钟来源、姿态状态及查询时刻，另存原时钟作为对照。
- `reproject_gt_manifest` 的独立 PTS 诊断 manifest 也使用同一选定姿态，仅按已有 referent 行重投影。它不替换原主评测 manifest，不证明绝对像素正确。

## 每个交互只保留对应 referent 的投影

审计先用 `load_ground_truth=False` 加载推理上下文并完成时钟拟合，然后使用原 evaluator 的 GT loader 和场景别名映射，按 `scene_key / row_index` 读取本交互 referent。GT 指令非空时核对其与当前交互一致；不从模型预测或同类候选池猜 referent。

- `referent_projections` 仅含该交互的去重 referent，包含原时钟对照投影。
- 画外、背面、缺少姿态、未映射或缺失 anchor 的 referent 都保留状态；不把画外点裁剪到图像边缘，也不扩成全场景候选。
- 青色圆圈为原时钟对照，橙色十字为选定时钟主投影，红色为有时钟支持的 gaze。图下列出全部交互 referent 的投影状态。两套标记独立绘制，原时钟画外时仍可显示选定时钟的画内点。
- GT 只用于拟合结束后的审计展示；不写回推理上下文、模型输入或时钟拟合材料。审计明确记录 `ground_truth_role=post_fit_referent_projection_only`。

## 测试与真实视频核验

所选 15 个测试模块共 135 项：134 通过、1 跳过、0 失败/错误。跳过项仍需要缺失的历史二维 manifest；测试阻止网络连接，模型后端使用 mock。

新增回归检查覆盖：候选/原时钟/不确定时钟、选定姿态不可用、查询时间一致性、残留 supported 标志、跨入口热图一致、独立 PTS referent 投影一致、无关 anchor 不输出、GT 在拟合后读取、GT 行错配及原时钟画外时的标记绘制。

沿用三个固定重建输入，每事件4张真实帧。相同12张源帧在真实实验图片准备入口的时钟、查询时刻、SHA-256和历史热图数值全部一致；保存20条 referent—帧记录，逐条核对 referent 集合通过：

| scene / row_index | 仅保留的 referents | 时钟处理 | 有支持的注意帧 |
| --- | --- | --- | ---: |
| scene1/162 | helicopter2 | 候选拒绝，原时钟诊断，RGB 回退 | 0/4 |
| scene4_room2/33 | cloth、chair | 候选拒绝，原时钟诊断，RGB 回退 | 0/4 |
| scene5/28 | drone5、oil_tank2 | 候选通过；主投影与注意图使用同一候选姿态 | 3/4 |

人工查看本轮总览仍可见部分投影与可见物体像素有偏差；没有把入口一致性解释为绝对定位正确。

这些输入来自本机重建 pilot100，不是找回了原冻结 CSV。候选仍只是 RGB 运动一致性验证；相机曝光、内外参、视差、动态目标和遮挡没有完成绝对标定。

## 产物与复现

最新输出目录：`/ai/data/MultiAgent/.local-sync/alignment_projection_referents_20261105_final/`。

- `summary.json`：3条交互的 referent 集合、时钟及支持帧数。
- `events/<scene>/<row>/audit.json`：仅交互 referent 的投影、原时钟对照、实际选定时钟/姿态、GT来源哈希。
- `events/<scene>/<row>/Vxxx_projection.jpg`：带 referent 名称及画外/缺姿态状态的审计图。
- `referent_projection_contact_sheet.jpg`：三个交互 V002/V003 的投影对照总览。
- `referent_projection_validation.json`：12张源图哈希/时钟等式与20条 referent—帧记录的范围校验。
- `pose_alignment/`：不使用 GT/gaze 的时钟拟合与源帧契约。
- `experiment_entry_parity/parity_summary.json`：同一批12张真实帧在实验图像入口的时钟、姿态时间、图像哈希和历史热图数值一致性检查；不调用模型。
- `regression.json`、`regression.txt`、`source_snapshot.json`：当前测试记录与关键源码哈希。

同帧入口核对直接使用本轮真实帧准备 paired-crop 四组图片；不是恢复原 projected-2D manifest，也不生成新的模型性能指标。所有审计标记只出现在诊断图片，实验图片仅为 RGB/注意呈现。

```bash
source /ai/data/TriModal-Referring/.local-sync/reconstructed_pilot100/env.sh
cd /ai/data/MultiAgent
python -m MultiAgent.alignment_projection_audit \
  --manifest .local-sync/alignment_v2_samples.json \
  --output .local-sync/alignment_projection_referents_new_check \
  --limit 3 --frames 4

python -m unittest MultiAgent.test_alignment_projection \
  MultiAgent.test_alignment_projection_audit \
  MultiAgent.test_projected2d_attention_study -v
```

输出目录必须是新的，避免覆盖历史诊断。此次结论仅为时钟/姿态使用一致且投影范围符合交互 referent，不是最终指代精度提升。
