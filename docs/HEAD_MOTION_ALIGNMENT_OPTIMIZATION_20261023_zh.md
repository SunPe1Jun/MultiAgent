# 头动时投影偏差：诊断与优化顺序（2026-10-23）

当前结果支持头动会放大残余时间配对问题。本轮新增的是不使用 GT 的 CPU 诊断，没有修改模型输入、GT 世界坐标、旧评价 manifest 或 evaluator，没有运行新的四组 MLLM 比较。

## 已验证的具体问题

固定 20 条、60 张原始源 JPEG 全部重新解码，通过 SHA-256 字节一致性核对，恢复其实际容器 PTS。解码使用原来的 `-ss ... -frames:v 1 -q:v 2` 设置，并用 `-copyts` 和 `showinfo` 记录输入帧时刻；只有重放 JPEG 哈希等于保存源图、PTS 能匹配 ffprobe 帧表时才标记确认。

- 60/60 张源图确认帧 PTS；不把请求 seek 时间或最近 PTS 冒充实际源帧时间。
- PTS 比抽帧请求时间晚：中位数 21.0615 ms，最大 78 ms。
- 例：scene4_room2/145 P2 请求 5.139 s，实际保存图片对应 5.165944 s，差 26.944 ms。
- 相机遥测的事件内采样间隔中位数，再对事件取中位数为 111.18025 ms，约 9 Hz；不是假定全部相机姿态都有 30 Hz。
- 当前热图已用位置线性插值、四元数 SLERP；原二维 GT 构建器使用离散样本姿态。原 panel 本身偏向采样时刻，因此原时钟下最近姿态和插值的差很小。改变时钟后，同一源帧通常落在两个姿态之间，差异会放大。不能以“加了 SLERP”作为绝对校准已经完成的证明。

容器 PTS 仍不是渲染相机曝光/提交时刻。恢复源图 PTS 消除的是可确认的“请求时刻代替实际帧时刻”问题，不能消除全部渲染延迟。

## PTS 版时钟重拟合结果

已实际使用真实视频 PTS 重做固定 20 条样本的时钟诊断。图像特征从按 `frame_index / fps` 改为按 ffprobe PTS 组织，视频帧顺序由连续 FFmpeg 解码确认，特征对使用 0.20 秒锚点和 0.16/0.36 秒间隔。拟合只使用静态背景候选特征，不读取 GT、gaze 或 anchor。

训练与验证按 1.5 秒时间块划分，边界前后 0.1 秒作为 guard；两侧不共享源帧。候选必须同时满足：验证特征数量足够、验证残差不高、低损失 offset 区间不宽、正向/反向时间块拟合一致，并且不能比原时钟在共同验证对上明显变差。

|结果|数量|处理|
|---|---:|---|
|`VALIDATED_CANDIDATE`|13/20|生成 PTS 对应的插值相机姿态，可供下一步重投影使用|
|`CLOCK_UNCERTAIN`|7/20|拒绝候选 offset，使用原始 offset 加精确 PTS/姿态插值生成新点，单独标记回退|

7 条不确定样本为 `scene1/705`、`scene2/416`、`scene3/318`、`scene4_room2/87`、`scene4_room4/29`、`scene5/4`、`scene5/583`。主要原因是时间块拟合不一致、候选 profile 过宽、验证残差过大或候选比原时钟变差。它们的拒绝候选保存在 `diagnostic_camera`，实际 `reference_camera` 使用原始 offset 回退；源帧和姿态采样可用时照常输出新投影。原始时钟的 RGB 运动验证只作诊断，不控制是否生成新点。

例如：

- `scene4_room1/170`: PTS 版 offset `2.044 s`，验证残差 `9.07 px`；
- `scene4_room2/145`: PTS 版 offset `1.542 s`，验证残差 `12.43 px`；
- `scene2/139`: PTS 版 offset `0.482 s`，验证残差 `7.75 px`。

这里的像素是最长边缩放到 640 后的背景旋转残差，不是 GT 定位误差。候选 offset 仍表示 `video_pts = telemetry_time + offset`，不是曝光时刻校准。全量摘要见 [PTS pose alignment summary](../runs/pts_pose_alignment_20261023/summary.json)，每个事件的时钟拟合和拒绝原因见各事件目录的 `clock_fit.json`。

随后用 PTS 版姿态对原世界 anchor 做了独立 post-fit 投影审计。第一版实现把 `CLOCK_UNCERTAIN` 错误地当成“不生成新点”，因此 7 条不确定事件的 42 条记录全部被跳过；这不是由旧点是否准确决定的。已修正为：候选时钟通过时使用候选 offset；候选被拒绝时仍使用原始 offset，只把精确容器 PTS 和姿态插值接入新投影。重跑后这 42 条记录全部生成了新投影，来源标为 `ORIGINAL_CLOCK_FALLBACK_PTS_POSE`；其中 38 条进入视锥、4 条画外。只有没有可用 offset、源帧时间未核实或姿态区间不可用时才不生成新 UV。

回退到原始时钟由候选拒绝状态决定，不读取旧点或 GT 的精度；新点生成不再受原始时钟 RGB 运动诊断门槛控制。新旧点很接近时，旧点用洋红圆环/十字、新点用蓝色十字区分；新点画外时不裁剪到边缘，CSV 保留未裁剪 UV 和 `outside_frame` 状态。

例如 `scene4_room2/145` 的 PTS 对照图同时画出旧 magenta 和新 cyan 投影；有些目标会从画内变成画外，符合相机转动的可能情况，但仍需结合真实画面、动态物体和相机模型复核。[PTS 投影审计图](../runs/pts_pose_alignment_20261023/projection_audit/scene4_room2_145_pts_projection_audit.jpg) 和 [逐目标 CSV](../runs/pts_pose_alignment_20261023/projection_audit/per_target.csv) 已保存。这个审计只是在拟合完成后使用 GT 画图，GT 没有参与 offset 选择。

## 头动敏感度

在图像九个固定测试射线上，使用同一世界方向，改变对应相机时间后计算投影位移。不使用 anchor、GT、gaze、深度或对象类别。各帧记录 ±10/20/50 ms 时间扰动；每帧先取九条射线位移中位数，再取两个扰动方向中的较大值。

以下数字使用前轮候选时钟，仅为诊断假设，未应用到生产输入；单位为源图像素：

|诊断量|60 帧中位数|60 帧最大值|
|---|---:|---:|
|最近离散姿态相对插值姿态的旋转投影差|1.71|70.72|
|±10 ms 时间扰动敏感度|0.65|27.39|
|±20 ms 时间扰动敏感度|1.36|55.59|
|±50 ms 时间扰动敏感度|3.98|135.97|
|请求时刻改用已确认 PTS 的旋转投影差|1.64|83.84|

候选配对下，60 帧中 10 帧的姿态区间平均角速度超过 45 deg/s，最高 165.98 deg/s。45 只是本次描述统计的阈值，不是已验证的排帧规则。原时钟配对会对应到不同的遥测时刻，不能比较两个时钟下的角速度就判断哪个标定准确。

这些数字不是实际目标定位误差，也不是精度提升；旋转测试不包含平移、有限深度视差、目标运动、遮挡或真实内参误差。±20/50 ms 是敏感度测试假设，不是估出的时钟置信区间。

## 建议按以下顺序优化

### 1. 先统一源帧 PTS 与参考姿态

新路径应保存 `source_frame_id / frame_pts / requested_seek_time / telemetry_query_time / pose_source_times`，源图、GT 世界点重投影与 gaze 热图使用同一个参考帧 PTS 和同一份插值相机位姿。保持原 world GT 和原 evaluator 匹配规则；如要输出校准后二维 GT，则单独版本化 manifest，保留原投影标签和旧指标作对照。

不要简单把已恢复 PTS 代入旧拟合 offset 就称为完成校准。PTS 版拟合已经完成第一轮，但仍需在后续投影图和独立同步证据上复核。当前图像对使用约 0.20/0.36 s 间隔；应继续检查不同间隔、不同背景区域和不同运动强度下的稳定性。

### 2. 重拟合时钟，以运动信息约束，但控制过拟合

先每段交互一个固定 offset；选参不使用 GT 或 gaze-target 对应。用静态背景匹配的几何残差，记录内点率、空间分布、共同可观测范围和失败。头动区间有助于区分 offset，但严重模糊、遮挡、平移和动态对象占主导的帧不能无条件加大权重。

保留没有参与拟合的连续时间区间验证；相邻图像对可能共享帧，前轮偶奇划分的保留检查不等于完全独立验证。固定 offset 仍不能解释且序列长度、运动激励足够时，再比较带正则的线性时间漂移 `t_video = a * t_telemetry + b`。短序列缺乏依据时，不逐帧自由调整 offset，不按 GT 移点，不同时任意修改 FOV。

### 3. 处理约 9 Hz 姿态采样的限制

统一使用位置线性插值和四元数 SLERP；保存插值两侧采样时间与间隔，不跨大缺口或边界外外推。SLERP 只能约束两个已知姿态间的路径，无法恢复真实未记录的加速、急停或回头细节。快速运动中的角加速度和渲染相机位姿差异仍需单独检查；盲目平滑姿态可能引入额外时间滞后。

可尝试用可靠背景特征提供帧间旋转细化，但需要保持相邻遥测姿态端点约束、检查残差并处理失败。没有深度时不要让一个全局 homography 伪造包含明显平移/近景视差的精确世界几何。

### 4. 用不确定度控制热图和 foveation

先从不含 GT 的保留残差或独立同步证据估计可支持的时间不确定范围，再把该范围投影到当前帧。可在小角度近似下理解为 `像素偏差 ≈ 焦距 × 角速度 × 时间偏差`，但边缘、平移和有限深度时须用实际投影评估。

候选方案是沿时间不确定范围投影多个参考姿态，将同一 gaze 世界点的投影扩散成带状注意区域，并降低过分集中的峰值；不能只把核无限加宽或将不确定注视归一化成强热点。Foveation 使用同一注意图，保留更温和的周边原图成分。线索不可信时保留 RGB 并记录回退，模型仍需处理该样本。

这属于新的视觉处理因素，必须在完成时间修正后另做配对对照，不能与时钟修复的收益混算。目前未实现或评价该不确定度呈现。

### 5. 选帧兼顾稳定观察与视觉转移

同一事件/mention 允许观察范围内，用时间覆盖、背景匹配质量、角速度和手势可用性选择代表帧。稳定帧承担精确定位，转移前后帧保留目标切换及顺序信息。不能把所有快速转头帧删除，也不能退回只保留少量静态帧造成多指代语义缺失。

RGB、当前热图、历史热图和历史 foveation 必须共用相同源帧集合及预算。单独按头动速度分层报告原指标，失败与低置信样本保留；GT 只用于评价和推理后画图。

## 实现、运行与状态

新增 [head_motion_diagnostics.py](../MultiAgent/head_motion_diagnostics.py) 和 [frame_pose_alignment.py](../MultiAgent/frame_pose_alignment.py)，实际运行固定 20 条、60 张源图的 PTS 核验和时钟重拟合。新增时钟/姿态测试通过，当前相关测试合计 44 项通过。没有修改推理架构或停止服务。

- [逐帧量化结果 per_panel.csv](../runs/projected2d_attention_qwen8b_20261022_int8/diagnostics/head_motion/per_panel.csv)
- [完整证据、精确源 JPEG PTS 与敏感度](../runs/projected2d_attention_qwen8b_20261022_int8/diagnostics/head_motion/sensitivity_audit.json)
- [PTS 版固定 20 条汇总](../runs/pts_pose_alignment_20261023/summary.json)
- [逐帧姿态契约示例](../runs/pts_pose_alignment_20261023/scene4_room2/145/frame_pose_contract.json)
- [PTS 版 GT 投影审计汇总](../runs/pts_pose_alignment_20261023/projection_audit/summary.json)
- [PTS 版 GT 投影逐目标结果](../runs/pts_pose_alignment_20261023/projection_audit/per_target.csv)

```bash
cd /media/kunyu/16TB/spj/Multiagent
export PATH=/home/kunyu/miniconda3/envs/trimodal/bin:$PATH
export SPJ_PREPARED_DATA_ROOT=/media/kunyu/16TB/spj/V3dMD/prepared/data
python -m MultiAgent.head_motion_diagnostics \
  runs/projected2d_attention_qwen8b_20261022_int8 --verify-saved-frame-pts
python -m unittest MultiAgent.test_head_motion_diagnostics \
  MultiAgent.test_projection_diagnostics MultiAgent.test_gaze_evidence
```

数字诊断、PTS 版时钟重拟合和 post-fit 投影审计已完成。散点图未生成：当前环境没有 matplotlib，记录 `NOT_RENDERED_MATPLOTLIB_UNAVAILABLE`，无需安装即可复核 CSV 和 JSON。PTS 候选尚未接入热图/foveation，也尚未重建校准后的 GT manifest 或运行四组模型；下一步应先人工复核 13 条可靠样本的旧/新投影，确认剩余偏差是否来自相机模型、移动物体或纯时间问题，再决定是否将候选姿态接入视觉输入。
