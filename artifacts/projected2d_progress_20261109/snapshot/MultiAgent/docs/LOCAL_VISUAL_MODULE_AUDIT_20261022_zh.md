# Local Visual Module Audit (2026-10-22)

本轮只运行 Referential/Visual，`Hand`、Coordinator、fusion 和最终 evaluator 均为 `NOT_RUN`。模型是本机 `/media/kunyu/16TB/spj/Qwen3-VL-8B-Instruct`，Transformers 5.9.0、PyTorch 2.9.1+cu128、bfloat16、SDPA，使用 GPU `NVIDIA RTX PRO 6000 Blackwell Workstation Edition`。原进程 `164313` 和 `16413` 均已停止。

## 输入与候选边界

100 条固定清单位于 `runs/local_dense_video_visual_8b_20261022_pilot100_v4/manifest.json`。每个事件从真实视频 PTS 均匀选取偶数帧，范围为34--100帧，中位数70；Qwen 的最大事件输入形成最多50个 temporal groups。Processor 审计显示采样关闭、无二次重采样、源帧哈希和时间戳均保存。

场景 anchor table 在每个事件单独加载，candidate ID 只来自当前 scene。候选审计覆盖242个 mention，没有跨场景或 evidence 越过 mention closed set 的记录。`there`/`here` 等 deictic mention 的候选集合为空，Visual 只能返回空间观察，不能把整个场景表当作候选池。

## 已验证的坐标字段

`gazePoint` 被按有限距离的世界坐标交点处理，`gazeVector` 被按世界方向处理；`cameraGazeDirection` 和 `cameraGazeOrigin` 被按相机坐标处理。`cameraRotation` 按 `(x,y,z,w)` 归一化，使用相机到世界旋转的共轭做世界到相机变换，FOV 按垂直 FOV、相机前向 `+z`、像素原点左上处理。

100 条事件共检查7918个 telemetry 样本。旋转后的 `cameraGazeDirection` 与世界 `gazeVector` 的事件中位数中位约 `1.34e-6` 度；`gazePoint` 与 `cameraHitPoint` 逆变换距离的事件中位数中位约 `1.14e-6` 世界单位。这只证明字段坐标内部一致，不证明视频与 telemetry 时钟、场景遮挡或 anchor 身份已经校准。

## 真实模型现象

v4 全量结果保存在 `runs/local_dense_video_visual_8b_20261022_pilot100_v4/`。100/100 事件完成，242 个 mention，100 次事件级 Visual 调用；模型候选正向证据只有2个 mention。199条 evidence 中197条引用 `G001`，143条观察也引用 `G001`。其中183个 mention 报告 `supported` 但分数和置信度为零，校验器已将其记录为无正向实例支持。这个历史 v4 运行沿用了旧 context adapter，曾读取 evaluator/GT 文件但没有把 GT 放入 prompt；因此不能把它称为严格的 GT-free 运行。

进一步检查 scene5_387 的真实图像发现，`G001` 对应视频开头的 Unity 黑屏，而模型声称该组能看到 drone、crate 和 truck。因此模型返回的 `frame_id` 不能当作精确观测时刻。代码现在保留 `model_cited_view_id`，正式 evidence 改用 `event_sequence` 或连续 `temporal_window` provenance，保存完整 source frame IDs 和 time range，并标记 `citation_exactness=unverified_model_view`。

分块对照 `runs/local_dense_video_visual_8b_20261022_v6_chunked_scene5/` 将100帧分为5个连续20帧窗口，实际覆盖全部帧并完成5次 Visual 调用。该结果可用于窗口级证据实验，不能解释为精确帧定位结果。

## 结论边界

- 已验证：场景候选隔离、deictic 空候选、完整多帧输入、源帧哈希/PTS 审计、gaze 字段坐标内部一致性、分块不丢帧。
- v6 Visual-only 入口已改为 `load_ground_truth=False`，并有 context isolation 测试；新的冻结对照不读取 evaluator/GT 文件。
- 已实现但未证明有效：逐 mention 调用、连续时间窗口、事件/窗口级 evidence provenance、无 GT 的 Visual-only context loading。
- 当前失败风险：8B MLLM 的精确时间引用不可靠；仅增加帧数没有解决同类实例对应；anchor 投影诊断仍显示几何位置不可信；没有运行 Coordinator 和最终指标。
- gaze 数值没有作为本轮 Visual 的直接输入；坐标审计与 Visual 证据分离。

## 复现命令

历史100条 v4 基线使用的固定清单：

```bash
SPJ_PREPARED_DATA_ROOT=/media/kunyu/16TB/spj/V3dMD/prepared/data \
PATH=/home/kunyu/miniconda3/envs/trimodal/bin:$PATH \
python -m MultiAgent.run_v3 --local-dense-video-visual \
  --manifest runs/visual_evidence_8b_20261021/pilot100_relaxed_clock/manifest.json \
  --output-dir runs/local_dense_video_visual_8b_20261022_pilot100_v4 \
  --cache-dir runs/local_dense_video_visual_8b_20261022_cache_v4_pilot100 \
  --qwen-model /media/kunyu/16TB/spj/Qwen3-VL-8B-Instruct \
  --visual-fps 4 --visual-min-frames 32 --visual-max-frames 100
```

冻结上游、连续窗口 Visual：

```bash
SPJ_PREPARED_DATA_ROOT=/media/kunyu/16TB/spj/V3dMD/prepared/data \
PATH=/home/kunyu/miniconda3/envs/trimodal/bin:$PATH \
python -m MultiAgent.run_v3 --local-dense-video-visual \
  --manifest runs/local_dense_video_visual_8b_20261022_chunked_scene5_manifest.json \
  --output-dir runs/local_dense_video_visual_8b_20261022_v6_chunked_scene5 \
  --cache-dir runs/local_dense_video_visual_8b_20261022_cache_v6_chunked_scene5 \
  --qwen-model /media/kunyu/16TB/spj/Qwen3-VL-8B-Instruct \
  --visual-call-granularity event --visual-window-groups 10 \
  --replay-from runs/local_dense_video_visual_8b_20261022_pilot100_v4
```

所有结果均为 offline full-interaction observation，不是在线结果，也没有加载 GT 参与预测或报告 mention accuracy。
