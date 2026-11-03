# VR-TriRef 视觉全过程审阅报告

日期：2026-10-22  
工作区：`/media/kunyu/16TB/spj/Multiagent`  
模型：本地 `/media/kunyu/16TB/spj/Qwen3-VL-8B-Instruct`  
报告性质：基于已保存输入、请求、processor 张量、视觉编码器入口、原始响应和历史评估文件的只读审阅；生成本报告没有新增模型调用。

## 先给结论

当前的视觉处理不是“图片没有交给模型”。最新组织实验中，热图、foveation、RGB 和 B3 图集都确实生成；10 次本地 8B 调用的 70 个图片条目文件哈希全部吻合，processor 还原出的图片与实际输入逐像素一致，保存的 processor 张量哈希也和视觉编码器实际入口一致。

但目前也不能说“模型已经有效使用了这些视觉引导”。证据分成三层：

1. **图像层：已验证。** 热图和 foveation 改变了输入像素，并进入了本地 Qwen3-VL-8B 的视觉编码器。
2. **低层线索读取层：热图部分成立，foveation 尚未成立。** 单帧反事实测试中，模型识别了两张非零热图的热点象限，也识别了零热图没有热点；但它没有稳定给出精确热点坐标。两个非零 foveation 样本都没有被模型识别出清晰焦点。
3. **指代推理层：尚未证明。** 最新的五组组织实验只运行 Visual，没有运行 Hand、Coordinator 和原评估器；B1 全景两条样本都失败并回退到 B3 图集，模型没有实际看到 B1 全景。因此这些结果不能回答“哪种视觉组织方式提高了最终 Any-hit/Exact-set/F1”。

此前 100 条远程完整 agent 试验中，`H_local` 相对 raw RGB 的最终 Micro-F1 从 `0.500` 变为 `0.511`，Any-hit 保持 `0.670`，Exact-set 从 `0.220` 降到 `0.210`；这个变化很小，而且旧试验仍把原始 gaze 数值/文本线索交给 Coordinator，不能作为“视觉热图已经被模型理解”的证据。当前最稳妥的判断是：**热图已经被模型看见，尚未证明它被可靠地用于同类实例指代；foveation 目前没有被验证为有效引导。**

## 一、审阅范围与实验分层

项目原本的流程是：

```text
原始指令
  -> Referential 逐 mention 解析
  -> 当前 scene 的候选映射
  -> 事件/mention 时间计划
  -> RGB + gaze heatmap + Hand cue
  -> Visual 逐 mention 证据
  -> closed-set Coordinator
  -> 原 evaluator
```

本轮保留了原构想中的两种 gaze 热图组织：**A 是多个关键时刻各自的热图，与同一时刻 RGB 配对；B1 是按相机姿态把跨时刻 gaze/RGB 组织到一张长球面图中。** A 本轮实际采用 8 个质量筛选后的时刻，不是三帧；B1 已输出真实尝试图，但两条样本都未通过配准检查，所以模型实际收到的是明确标记的 B3 `spatial_atlas` 多视图回退。B3 拼图不冒称全景。

本报告把实际运行分成三类，避免把不同终点混在一起。

| 运行 | 样本 | 视觉输入 | 是否完整 agent | 用途 |
|---|---:|---|---|---|
| `pilot20_corrected` | 20 | 每组 3 个事件帧，R/H/F | 是，含 Coordinator 和 evaluator | 旧版系统级配对指标 |
| `pilot100_relaxed_clock` | 100 | 每组 3 个事件帧，R/H/F | 是，含 Coordinator 和 evaluator | 旧版较大规模系统级配对指标 |
| `local_dense_video_visual_8b_20261022_pilot100_v4` | 100 | 每事件 34--100 个真实 PTS 帧，中位数 70 | 否，停在 Visual | 检查密集视频、候选隔离和时序证据 |
| `local_dense_video_visual_8b_20261022_v6_*` | 1 个事件 / 2 个事件 | 74--100 帧；连续窗口或逐 mention | 否，停在 Visual | 检查 100 帧分块与 mention 隔离 |
| `local_attention_guidance_probe_20261022_pilot4` | 4 个事件 | 8 个真实 PTS 帧，R/H/F，两种提示词 | 否 | 检查提示词和线索可读性 |
| `local_visual_organization_guidance_probe_20261022_pilot2` | 2 个事件 | 8 个真实 PTS 帧，C0/A-H/B-H/A-F/B-F | 否，未含 Hand/Coordinator/evaluator | 检查两种视觉组织方式的真实输入 |
| `heatmap_pixel_sensitivity_20261022_scene1_163_V051` | 1 帧 | 热图翻转/零值/原图，foveation 翻转/零值/原图 | 否 | 检查模型是否读取像素线索 |

最新组织实验的固定样本是：

| scene/event | 指令 | Visual mention | 候选 |
|---|---|---|---|
| `scene1/scene1_163` | `Turn toward that helicopter.` | `that helicopter` | `helicopter1`, `helicopter2` |
| `scene2/scene2_35` | `Slide that chair toward there.` | `that chair` | `chair1`--`chair8` |

最新实验配置明确设置 `ground_truth_loaded=false`、`coordinator_run=false`，没有加入红圈或其他目标标注。它复用了已经冻结的 Referential、scene-local 候选和真实视频帧，但没有重复运行上游模型。

## 二、模型、后端和留痕

最新组织与敏感性测试使用本地 Transformers 后端，不经过之前的远程 OpenAI-compatible 服务：

- checkpoint：`/media/kunyu/16TB/spj/Qwen3-VL-8B-Instruct`。
- Transformers：`5.9.0`。
- PyTorch：`2.9.1+cu128`。
- dtype：`bfloat16`。
- device：`cuda:0`。
- attention：`sdpa`。
- GPU：`NVIDIA RTX PRO 6000 Blackwell Workstation Edition`。
- 模型配置哈希：`5cd452860dc1e9c29dd71cc3cef7f39b338b7a40793f7a260655c2d3568f3661`。
- 最新组织实验 cache namespace：`visual_organization_qwen8b_20261022_v1`。
- 旧远程 100 条实验的服务端 `/models` 留痕为 `Qwen3-VL-8B-Instruct`，服务端 root 也是本地 8B 权重路径。

最新 10 次组织调用的总耗时为约 `1037.97 s`，其中模型推理耗时约 `1021.94 s`。这只是已保存运行的耗时，包含本地模型和输入处理；不能由此推断 foveation 会加速推理。当前代码的 `budget.json` 也明确把“模糊导致推理加速”标为 false。

所有最新本地请求都保存了：

- 实际输入图片路径和图片 SHA-256；
- prompt 和 prompt SHA-256；
- processor 输入张量形状、dtype、字节数和 SHA-256；
- 视觉编码器 forward hook 的入口张量哈希；
- 原始响应、解析结果、调用耗时；
- source frame ID、PTS、时间范围和 view provenance。

密钥没有写入本地报告或请求日志。

## 三、候选映射与 agent 架构是否保留

### 3.1 最新 Visual-only 路径

最新组织实验不是独立的图像分类脚本。它从冻结的 local v4 `referential.json` 读取 mention，再调用 `_candidate_scope(ctx, mention)`：

```text
mention.candidate_ids
  -> 检查是否全部属于当前 scene anchor table
  -> 只把当前 mention 的候选传给 Visual
```

最新 100 事件的 `scene_anchor_scope.json` 审计结果为：

- `cross_scene_anchor_tables_loaded=false`；
- `anchor_table_basename_matches_scene=true`；
- `candidate_ids_are_scene_local=true`；
- 100 个事件中没有记录跨 scene anchor table 泄漏。

100 条 local v4 的候选范围统计为：

| 范围 | mention 数 |
|---|---:|
| `mention_closed_set` | 215 |
| `unresolved_spatial_observation` | 15 |
| `unresolved_object_observation` | 12 |

Visual 收到的 scene-local candidate table 只包含候选 ID、类别和 aliases，不包含 anchor 世界坐标；因此这一阶段是“在隔离的候选 ID 集合中看 RGB 判断”，不是用候选坐标先把 gaze 映射到对象。旧 Coordinator 有自己的候选空间上下文，但没有进入本轮组织实验。

像 `there`、`here` 这种空间指示表达，如果没有 lexical object candidate，当前代码返回空候选，并要求 Visual 只记录空间观察，不把整个场景 anchor table 当作候选集合。这一点在当前实现中已经改成显式的 `unresolved_spatial_observation`，不是把全场景对象灌给 Visual。

因此，“`there` 获得几十个同场景候选”不能作为当前 `_candidate_scope` 的行为描述。早期运行和旧文档中可能出现更宽的空间候选上下文，但不能用它们替代最新 `scene_anchor_scope.json` 的隔离审计。

### 3.2 仍然存在的架构边界

最新组织实验只冻结并复用了 Referential。它没有继续执行：

```text
Hand -> Visual evidence fusion -> Coordinator -> evaluator
```

所以这轮能验证的是“Visual 收到什么、是否报告 cue、是否产生候选证据”，不能验证完整 agent 的最终集合。local v4 也明确记录了 `fusion=NOT_RUN`。

另外，旧 local v4 使用过旧 context adapter，`ground_truth_load_unknown_events=100`。旧运行没有把 GT 作为 prompt 输入，也没有用 GT 做预测，但不能把那一批历史运行描述为严格的全流程 GT-free。最新组织实验通过 `load_contexts(..., load_ground_truth=False)` 重新加载上下文，GT 没有参与输入构造。

## 四、视频抽帧和时间组织

### 4.1 密集视频版本

为了回应“三帧不够”的问题，local v4 不是三帧输入：

- 每个事件实际使用 34--100 个源帧；
- 中位数 70 帧；
- 目标 Qwen temporal group rate 约 4 groups/s；
- 每个 Qwen temporal group 由两个连续源帧组成；
- processor sampling 关闭；
- 没有二次 processor 重采样；
- 每个事件最多 100 帧，视觉 token 估计约 3321--4056。

local v6 还实际把一个 100 帧事件分成 5 个连续 20 帧窗口，完成 5 次 Visual 调用；另一个 mention pilot 使用 74--100 帧的逐 mention 输入。这些运行的目的，是确认全序列和连续窗口能被送入模型，而不是把模型引用的 `G001` 当成精确观测时刻。

历史 remote pilot20/pilot100 是更早的三帧系统级对照，不能代表当前密集视频版本。

### 4.2 最新组织实验为什么又使用 8 帧

最新组织实验选择了 local v4 事件帧池中的 8 个合格 PTS 帧，目的是把热图是否可见、A/B 组织差异和真实模型输入留痕控制在可检查规模。它不是最终视频输入方案，也不是声称 8 帧足够。

两条样本的 frame pool 和发送帧为：

| 样本 | 完整 local 帧池 | 通过 pose/质量筛选 | 发送给五组 | 发送时间范围 |
|---|---:|---:|---:|---|
| `scene1/scene1_163` | 56 | 39 | 8 | 2.125389--6.887933 s |
| `scene2/scene2_35` | 58 | 41 | 8 | 2.122967--7.164744 s |

具体发送帧为：

- `scene1`：`V018,V023,V029,V034,V040,V045,V051,V056`，对应原视频 `frame_000046,054,063,070,079,087,105,119`。
- `scene2`：`V018,V024,V029,V035,V041,V047,V052,V058`，对应原视频 `frame_000051,069,085,103,121,139,154,172`。

选择策略不是 GT 选帧，也不是由目标 anchor 选择关键帧，而是“有相机 pose 且画面不是黑屏后，在合格帧中按时间分层均匀选择”。因此它还不能称为“关键时刻判别器”。它只是质量过滤后的固定时间覆盖采样。

### 4.3 实际视频时间不是固定 30 FPS

源视频实际记录为 1920×1920，PTS 被保留：

| 样本 | 视频时长 | ffprobe 平均帧率 | gaze 记录 | gaze 中位间隔 |
|---|---:|---:|---:|---:|
| `scene1/scene1_163` | 6.890067 s | 约 17.42 FPS | 49 | 0.165849 s，约 6.03 Hz |
| `scene2/scene2_35` | 7.205656 s | 约 24.01 FPS | 62 | 0.111082 s，约 9.00 Hz |

没有把“视频 30 FPS”硬编码成 `frame_index / 30`。模型请求中的 Qwen 时间标签来自相对 PTS，而不是简单帧号。

### 4.4 ASR 提示、动作阶段和实际观测时刻

最新两个冻结上游记录的 `asr_hint` 都是 `null`，时间计划是 `instruction_event_only`。因此本轮没有把音频词级时间当作 mention 的精确 `tau`。视频事件范围只作为软动作阶段/观察先验；gaze 历史和每张 RGB 的实际观察时刻则按已记录 telemetry timestamp 与视频 PTS 计算。实际送入模型的帧时间列在 `frames.csv`，不会把整个事件的软范围伪装成精确指代时刻。

两组离线查看了完整 recording 的可用范围，但具体 A/B 输入仍是质量过滤后的 8 帧。它不是在线结果，也不是逐帧观看全部源视频的 A/B 比较。

## 五、gaze 字段、坐标约定和时间保护

### 5.1 实际字段解释

当前读取器保留并区分：

- `gazePoint`：有限距离的世界坐标交点；全零值按无有效交点处理。
- `gazeVector`：世界坐标方向向量；只在没有有效 `gazePoint` 时作为无限距离方向使用。
- `gazeOrigin`：世界坐标视线起点，存在时保留。
- `cameraGazeDirection`：相机坐标方向，用于数值一致性审计，不直接作为 Visual 输入。
- `cameraPosition`、`cameraRotation`、`cameraFOV`：记录的相机位姿和标量 FOV。
- `isEyeOpen`、`handData` 和原始 timestamp：保留在 gaze record 中；数据中没有可靠 gaze confidence/quality 数值。

### 5.2 几何变换

代码和数值审计采用：

- quaternion 顺序 `(x,y,z,w)`；
- 记录的 quaternion 作为 camera-to-world rotation；
- world-to-camera 使用 quaternion 共轭/逆旋转；
- 相机局部 `+z` 为前向，`+y` 为上；
- `cameraFOV` 按垂直 FOV；
- 方形像素假设；
- 图像左上角为像素原点；
- 只做视锥投影，不把画幅外点 clamp 到边界。

对 local v4 的 100 个事件、7918 个 telemetry 样本做的数值一致性检查显示：

- world `gazeVector` 经逆旋转后与 `cameraGazeDirection` 的事件级中位误差约 `1.34e-6°`；
- `gazePoint` 逆变换到 camera 坐标后与 `cameraHitPoint` 的事件级中位距离约 `1.14e-6` 世界单位。

这证明字段之间在当前内部约定下自洽，不能证明：

- 记录的相机是中心眼、头部还是录制相机；
- 视频镜头和 telemetry 的外参已经标定；
- Unity 世界单位是物理米；
- 相机畸变、逐帧内参或非方形像素已被处理；
- gazePoint 真的可见、属于哪个 anchor，或对应哪个动态物体。

### 5.3 可见性和深度

当前每个投影样本会区分 `invalid_sample`、`behind_camera`、`outside_frame` 和 `projected_frustum_only`。在没有 depth 或可重放 mesh 时，`visibility` 保持 `visibility_unknown`。

数据中没有可用于整幅 RGB 反投影的可靠深度，也没有本轮可对应的场景 mesh 和逐时刻动态物体变换。因此没有伪造 B2 世界坐标重投影，也没有把 gazePoint 本身当作 anchor ID。

### 5.4 时钟保护的实际影响

最新两个组织样本仍使用已有 metadata 起点偏移，偏移状态是 `metadata_origin_estimate_not_pixel_calibrated`：

| 样本 | telemetry-to-video offset | 选中视频时间 | 对应 telemetry 时间 | 选定窗口外的 gaze 记录 |
|---|---:|---|---|---:|
| `scene1/scene1_163` | 2.052209 s | 2.125389--6.887933 | 0.073180--4.835724 | 19 |
| `scene2/scene2_35` | 2.084538 s | 2.122967--7.164744 | 0.038429--5.080206 | 16 |

这说明本轮虽然使用了离线完整交互记录，但“完整交互”不等于所有 gaze 记录都覆盖了发送给模型的帧尾，也不等于有精确 mention 时刻。报告没有把 gaze 末端自动补到视频末尾。

## 六、路线 A：局部关键时刻帧和逐帧热图/foveation

### 6.1 数值热图怎么生成

当前局部热图是每个参考帧独立计算的，不把不同相机视角的像素坐标直接相加：

1. 以参考帧 PTS 减去 clock offset 得到 telemetry 时间。
2. 取该参考帧之前最多 2 秒的有效 gaze 记录。
3. 对相机位姿做线性位置插值和 quaternion slerp，最大插值 gap 为 0.25 s。
4. 用参考帧相机 pose、FOV 和宽高比把世界 gazePoint 或 gazeVector 投影到参考帧。
5. 只对 `projected_frustum_only` 点累计 Gaussian kernel；画幅外、相机后方和无 pose 的样本不参与。
6. 保存未归一化 `H_unnormalized`、source index、原始时间、有效持续时间和投影状态。

配置为 `local_history_seconds=2.0`、`decay=uniform`、`kernel_sigma_degrees=2.0`、`display_scale=2.0`。这个热图表达的是“在给定窗口内的几何 gaze 证据”，不是目标概率，也不是用户意图标签。

以 `scene1/scene1_163/V051` 为例：

- 2 秒历史窗口中选择 12 个 gaze 样本；
- 12 个都完成视锥投影；
- 有效证据持续时间 `1.96143 s`；
- 未归一化热图最大值 `0.711437`，总和 `1031.5515`；
- 热图正值像素约占 `5.61%`；
- 2° kernel 换算到 672×672 输入约为 `9.166 px`。

### 6.2 H 和 F 的实际实现

局部 H/F 的关系是“同一份数值热图，两个呈现方式”：

- `H_local`：红色弱叠加，最大 alpha `0.28`。
- `F_local`：使用 2×、4×、8× 降采样后再上采样的图像金字塔；高热区域保留较多原图，低热区域仍保留 `0.55` 的清晰 RGB 成分。

`V051` 的 RGB 像素审计显示：

- `H_local` 相对 RGB 改变约 `5.61%` 像素，平均 RGB 差约 `0.118`；
- `F_local` 相对 RGB 改变约 `90.36%` 像素，平均 RGB 差约 `3.063`。

这说明 F 的图像扰动远大于 H，但“改变更多像素”不等于“提供更有用的视觉证据”。此外，配置里的 `foveation_max_blur_sigma_px=18` 在局部平面 renderer 中没有直接作为 Gaussian sigma 使用；实际采用的是 2×/4×/8×降采样金字塔和清晰度混合。B 的球面 foveation 另有角度核实现，不能与 A 的平面 foveation 视为完全同一 renderer。

### 6.3 最新 A-H 实际发送的内容

A-H 不是把红色 H 图直接覆盖在 RGB 上，而是按每个时刻发送：

```text
heatmap_guide(V018), R(V018), heatmap_guide(V023), R(V023), ...
```

因此每条样本发送 16 张静态图；每张热图与其 RGB 帧共享 `view_id`、source frame ID 和 PTS。热图是独立灰度图，RGB 保持未叠加。

`H_local.png` 这些红色叠加文件确实生成并保存，但**不是最新 A-H 模型请求中的输入**。这是本轮审阅中最需要区分的文件层事实。

### 6.4 A-F 实际发送的内容

A-F 发送同一组 8 个 source frame 的 foveated RGB 视频，使用 Qwen video processor，`video_grid_thw=[4,42,42]`，视觉 token 估计为 `1764`。它不是 16 张静态图，也没有额外给模型一张独立热图。

## 七、路线 B：B1 全景尝试、B2 缺失和 B3 回退

### 7.1 B1 的实际实现

B1 尝试使用相机内参和旋转，把 8 个视角的 RGB 射线映射到共同 equirectangular 球面；gaze 也用同一相机射线模型映射到球面。保存了：

- `RGB.png`；
- `heatmap.png` 和带时间权重的 `temporal_heatmap.png`；
- `heatmap_overlay.png`；
- `foveation.png`；
- `coverage.png`；
- `attention_and_provenance.npz`；
- `gaze_time_index.json`；
- source view、source frame、时间和球面区域映射。

球面热图使用角距离 Gaussian，处理经纬度缩放和 longitude seam；未知区域保留深灰，未做生成式补全。累计热图另存 early/middle/late 图层，但一张累计图本身不保存完整因果顺序。

### 7.2 B1 几何检查结果

两条样本都因为 rotation-only warp 的光度配准残差过大而回退：

| 样本 | B1 状态 | 矩形像素覆盖率 | 按纬度余弦修正的球面覆盖率 | 最大相机位移 | 相邻帧残差超过 5 px |
|---|---|---:|---:|---:|---:|
| `scene1/scene1_163` | `FALLBACK_SPATIAL_ATLAS` | 27.19% | 32.79% | 0.0808 world units | 4/7 |
| `scene2/scene2_35` | `FALLBACK_SPATIAL_ATLAS` | 27.74% | 35.11% | 0.0825 world units | 6/7 |

中位 residual 的具体范围为：

- scene1：`6.117, 9.055, 2.000, 2.000, 8.246, 11.314, 4.243 px`；
- scene2：`7.808, 6.516, 11.402, 9.220, 8.559, 11.402, 5.000 px`。

这些 residual 是基于局部 NCC 的工程诊断，不是静态世界配准的证明。B1 忽略平移，所以不能被称为几何准确的世界全景；本轮没有把它强行送给模型。

两条样本中 A 局部逐帧投影与 B1 球面投影最终纳入的有效 gaze `source_index` 集合相同，分别为 30/49 和 46/62；但有限距离 gazePoint 在 A 中投向每个参考帧相机，在 B1 中转成 gaze 观测相机的世界方向后忽略平移。B1 gaze 到最近 RGB source view 的最大时间差分别约 `0.364 s` 和 `0.368 s`，没有精确到同步曝光。故“source index 集合相同”不等于像素落点相同或跨视角几何已验证。

### 7.3 B2 为什么没有构造

只有 gazePoint 和 anchor 坐标不足以反投影整张 RGB。当前数据没有可靠 depth 或可对应的场景 mesh，因此 B2 明确记录为 `NOT_AVAILABLE`，没有伪造世界坐标 RGB。

### 7.4 B3 实际是什么

B1 失败后，实际送给模型的是 B3 `spatial_atlas`：

- B-H：`B3_heatmap_guide_atlas.jpg` + `B3_RGB_atlas.jpg`，两张 1280×704 图；
- B-F：一张 `B3_F_spatial_atlas.jpg`，1280×704；
- 图集由 8 个真实视角拼成 4 列×2 行 tile，每个 tile 带 `view_id` 和 PTS 文本；
- 外部 JSON 保留相机轨迹、姿态、source frame 和时间索引。

B3 是多视图图集，不是全景。模型看到的是 tile 图像和文字标签，**没有直接看到外部 JSON 中的世界位置、朝向或相机轨迹**。因此 B3 的空间组织主要依赖图像布局和 tile 标签，不应解释为模型完成了世界坐标空间推理。

## 八、模型实际收到了什么

最新组织实验的输入规模如下。这里的视觉 token 是 processor 的视觉网格估计，不是总 prompt token。

| 条件 | 每条样本图片/视频媒体数 | 视觉网格 | 视觉 token | 说明 |
|---|---:|---|---:|---|
| C0 | 8 帧视频 | `[4,42,42]` | 1764 | 原始 RGB control |
| A-H | 16 张静态图 | 16 个 `[1,42,42]` | 7056 | 热图、RGB 交替成对 |
| B-H | 2 张 atlas | 2 个 `[1,44,80]` | 1760 | B3 热图图集 + RGB 图集 |
| A-F | 8 帧视频 | `[4,42,42]` | 1764 | 8 帧 foveation |
| B-F | 1 张 atlas | `[1,44,80]` | 880 | B3 foveation 图集 |

两条样本合计：

- 10 次真实 local model call；
- 70 个模型输入图片条目；
- C0/A-F 每样本 8 个 source frame；
- A-H 每样本 8 个热图 + 8 个 RGB；
- B-H/B-F 实际使用 B3 atlas；
- B1 生成的全景图片没有进入这 10 次请求。

输入传递复核结果：

| 检查 | 结果 |
|---|---|
| 请求记录图片哈希与实际文件 | 70/70 一致 |
| processor 还原图片与实际文件逐像素 | 全部一致 |
| prompt SHA-256 与请求记录 | 全部一致 |
| 原始响应与预测文件保存内容 | 全部一致 |
| processor 张量与保存张量 | 全部一致 |
| 视觉编码器 forward 入口与 processor cast 后张量 | 全部一致 |

这已经排除了“图片生成了但没有发给模型”“processor 读了旧图片”“视觉编码器没有消费该输入”这几类问题。

## 九、提示词是否引导模型理解热图和 foveation

最新 Visual prompt 做了以下约束：

- 明确当前是 Visual agent，解决一个 mention；
- 候选 ID 只能来自当前 scene 和当前 mention；
- 不允许使用数值 gaze、world coordinate、camera pose、GT 或 candidate distance；
- 热图只作为弱 corroboration，不是目标标签或目标概率；
- 先逐 view 审计 cue，再使用 RGB、动作和关系解决 mention；
- 要求返回 `attention_trace`、`evidence`、`visual_observations`、`identity_status`；
- 要求每个 view 返回 cue type、中心、置信度和观察说明；
- A-H 明确说每个时刻第一张是灰度热图，第二张是对应未修改 RGB；
- B-H/B-F 明确说 B3 是 spatial atlas，不是 panorama；
- foveation 明确说模糊只是呈现方式，不代表遮挡或物体不存在。

提示词参考了用户提供的 *In the Eye of MLLM* 中“先让模型审阅视觉显著性线索，再进行任务推理”的思路，但没有使用红圈，也没有把数字 gaze 坐标直接给 Visual。

但是提示词还有一个具体缺陷：最新组织 `_prompt` 的 schema 写的是 `cue_center_uv: [u,v] in that view`，没有明确写 `[0,1]`；像素敏感性测试的 prompt 也只写 `center_uv`，没有明确归一化范围。后处理代码却按 `[0,1]` 期待中心，并把大于 1 的值按图片像素做事后归一化。这使得“坐标不符合格式”的部分不能完全归咎于模型，首先是 schema 没有把坐标系写死。

因此，当前提示词确实在告诉模型“要看热图/foveation”，但还没有形成足够严格的视觉接口：

- cue 是否被读取是模型自述和输出字段，不能视为内部注意力证明；
- cue 中心的坐标域没有在所有测试 prompt 中明确；
- B3 一个 atlas 图却要求按 8 个原始 view 返回 cue，视图和图片之间的对应关系对模型不够直接；
- 没有提供模型可用的对象级 gaze-to-anchor 关联，这一点是设计边界，不应由 prompt 伪造。

## 十、最新五组模型响应

最新组织实验没有 GT 评估，以下只报告模型是否报告 cue、候选证据和响应耗时。

### 10.1 `scene1/scene1_163`: helicopter

| 条件 | cue 报告 | cue 中心审计 | 正向候选 | identity |
|---|---:|---:|---|---|
| C0 | 0/4 | 无 | `helicopter1`, `helicopter2` | `resolved` |
| A-H | 8/8 | 平均中心误差约 0.1088；8 个中心均为 `[0.5,0.5]` | `helicopter1` | `resolved` |
| B-H | 0/8 | 无 | 空 | `none` |
| A-F | 4/4 | 平均中心误差约 0.1082；中心均为 `[0.5,0.5]` | `helicopter1` | `resolved` |
| B-F | 0/8 | 无 | 空 | `ambiguous` |

这里最值得注意的是 A-H 和 A-F 都报告了“cue 可见”，但几乎所有中心都退回到画面中心 `[0.5,0.5]`，而真实热图中心随时间变化。这更像是模型能够识别“有一个热图/引导”，但没有可靠地追踪每个时刻的精确空间位置。C0 在没有 cue 时反而报告了两个候选，说明候选集合和 RGB 本身已经能诱发对象判断；A-H 的单个候选不能单独证明是热图带来的提升。

### 10.2 `scene2/scene2_35`: chair

| 条件 | cue 报告 | cue 中心审计 | 正向候选 | identity |
|---|---:|---:|---|---|
| C0 | 0/4 | 无 | 空 | `not_visible` |
| A-H | 0/8 | 无 | 空 | `unclear` |
| B-H | 0/8 | 无 | 空 | `none` |
| A-F | 0/4 | 无 | 空 | `none` |
| B-F | 3/8 | 输出 `[280,700]`、`[580,850]`、`[800,900]`，不符合 `[0,1]` 解析 | `chair1`, `chair2`, `chair3` | `ambiguous` |

B-F 的三个坐标看起来是 atlas 或原始像素尺度，而不是当前后处理期待的每个 view 的归一化坐标；同时它把不同时间的多个 chair 都列为正向候选。这不能作为 foveation 定位成功，反而显示出“单张 atlas、8 个 view schema、候选 ID”之间的接口不够严谨。

### 10.3 这两条样本能否比较 A/B

不能做系统效果结论，原因有四个：

1. 样本数只有 2 个，且每条只跑一个 mention。
2. A-H 使用 7056 视觉 token，C0/A-F 约 1764，B-H 1760，B-F 880，视觉预算不相等。
3. A-H 是 16 张静态图，A-F 是 8 帧视频，B 组是压缩后的 atlas；媒体组织、模型 token 和提示词同时改变。
4. B1 没有进入模型，B 组实际比较的是 B3 spatial atlas。

这组实验能支持的是“实际输入管线和模型响应存在明显差异”，不能支持“局部路线优于全局路线”或“热图提升了最终指代”。

## 十一、独立像素敏感性：热图与 foveation 是否被看见

测试选取真实 `scene1/scene1_163/V051`，只改变视觉 cue，不提供目标标签、答案坐标或 GT：

- X1：真实热图旋转 180°；
- X2：全零热图；
- X3：真实热图；
- RGB 本身保持不变；
- foveation 也分别使用翻转、零值和原始热图构造。

### 11.1 热图

模型输出：

- X1 非零热点被检测，象限 `top_left`，与翻转热图象限一致；
- X2 无热点，被报告为 `none`；
- X3 非零热点被检测，象限 `bottom_right`，与原图一致；
- 热图象限判断为 `3/3`；
- 两张非零热图均被识别，零热图没有被误报为热点。

这是目前最直接的“热图像素进入模型且模型能读出粗粒度热点”证据。

但模型中心输出为 `[100,100]` 和 `[400,400]`。由于测试 prompt 没有明确写中心必须是 `[0,1]` 归一化坐标，报告中保存的 `0.174` 和 `0.297` 误差只能称为“按 672 像素事后解释的诊断值”，不能称为严格 schema 违例。它至少说明模型没有稳定给出可追溯的精确热图中心。

### 11.2 Foveation

模型输出：

- 翻转非零 foveation：未检测到清晰焦点；
- 零值 foveation：未检测到清晰焦点；
- 原始非零 foveation：未检测到清晰焦点；
- 对两个非零输入，检测数为 `0/2`；
- 原始 foveation 响应是顶层 JSON array，而 backend 契约要求 JSON object；原始响应没有丢失，汇总时只做了诊断性包装。

零值没有被检测为焦点是正确的，但不能抵消两个非零 foveation 都没有被识别。当前不能说模型读取了 foveation，更不能说它使用 foveation 改善了目标选择。

### 11.3 旧的 8 帧提示词可读性探针

`local_attention_guidance_probe_20261022_pilot4` 完成了 4 个事件、24 次调用、每次 8 帧。每个调用要求返回 4 个 temporal group 的 cue trace，共 96 条 trace row；保存结果中 cue 报告为 0/96，候选正向证据也为 0。

这与单帧热图 `3/3` 象限正确并不矛盾：

- 8 帧视频中的早期帧包含启动黑屏或无有效 gaze；
- stepwise prompt 要求模型逐 view 报告 cue；
- 热图在视频帧中可能很小，foveation 是弱的全图变化；
- 没有对视频 cue 先做独立的模型可读性校准。

它反而说明“在输入中存在热图像素”与“模型在真实视频 Visual 任务中稳定抽取 cue”之间还有明显鸿沟。

## 十二、历史完整 agent 结果

以下结果来自较早的 remote 8B 完整流程，包含 Referential、候选、时间计划、原始 RGB Hand、Visual、Coordinator 和原 evaluator。它不是最新 B1/B3 组织实验。

### 12.1 20 条 corrected pilot

保存的 `summary.json` 和 evaluator 文件给出的 Exact-set 是 `0.200`。旧的 `docs/MULTIAGENT_CURRENT_PROGRESS_zh.md` 曾写成 `0.250`，属于文档过期/不一致；本报告以逐组 summary 和 evaluator 为准。

| 组 | Any-hit | Exact-set | Micro-P | Micro-R | Micro-F1 |
|---|---:|---:|---:|---:|---:|
| R | 0.750 | 0.200 | 0.677 | 0.438 | 0.532 |
| H_inst | 0.750 | 0.200 | 0.724 | 0.438 | 0.545 |
| H_local | 0.800 | 0.200 | 0.759 | 0.458 | 0.571 |
| F_inst | 0.750 | 0.200 | 0.808 | 0.438 | 0.568 |
| F_local | 0.650 | 0.200 | 0.731 | 0.396 | 0.514 |

该 pilot 的时钟保护拒绝 13/20 个样本的 gaze/video 对齐，只有 7/20 至少生成有效投影；拒绝样本保留 raw RGB 并进入分母。因而 H_local 的 +0.05 Any-hit 和 +0.0398 Micro-F1 是小样本、低有效 gaze 覆盖下的观察，Exact-set 没有变化。

### 12.2 100 条 relaxed-clock pilot

| 组 | Any-hit | Exact-set | Micro-P | Micro-R | Micro-F1 |
|---|---:|---:|---:|---:|---:|
| R | 0.670 | 0.220 | 0.608 | 0.424 | 0.500 |
| H_inst | 0.670 | 0.210 | 0.607 | 0.429 | 0.502 |
| H_local | 0.670 | 0.210 | 0.615 | 0.437 | 0.511 |
| F_inst | 0.650 | 0.220 | 0.651 | 0.408 | 0.501 |
| F_local | 0.630 | 0.210 | 0.625 | 0.399 | 0.487 |

相对 R：

- H_inst：Any-hit `0`，Exact-set `-0.010`，F1 `+0.002`；
- H_local：Any-hit `0`，Exact-set `-0.010`，F1 `+0.011`；
- F_inst：Any-hit `-0.020`，Exact-set `0`，F1 `+0.001`；
- F_local：Any-hit `-0.040`，Exact-set `-0.010`，F1 `-0.013`。

这不是稳定提升。100 条中 98 条完成冻结上游，2 条上游 extraction 失败；五组仍有相同的上游失败记录。99/100 事件窗口被视频长度裁剪，时钟可用性和事件尾部覆盖仍是变量。

### 12.3 错误类型是否支持“模型理解了 gaze”

100 条的 evaluation-only failure analysis 统计如下：

| 组 | 有漏选 | 有额外对象 | 同类实例混淆 | Coordinator 空选择回退 |
|---|---:|---:|---:|---:|
| R | 78 | 44 | 42 | 24 |
| H_inst | 79 | 44 | 42 | 25 |
| H_local | 79 | 45 | 42 | 25 |
| F_inst | 78 | 38 | 36 | 30 |
| F_local | 79 | 43 | 40 | 28 |

热图没有消除同类实例混淆，foveation 也没有稳定减少漏选。H_local 的 F1 小幅上升主要来自集合统计平衡，不代表 mention 级定位已经正确。项目没有编造 mention accuracy。

### 12.4 Coordinator 的影响

旧 100 条实验中，Visual 输入变化后 Coordinator 最终集合确实改变：

| 组相对 R | Coordinator 改变集合 | Any-hit 改善/恶化 | Exact-set 改善/恶化 |
|---|---:|---:|---:|
| H_inst | 51 | 0 / 0 | 0 / 1 |
| H_local | 50 | 1 / 1 | 1 / 2 |
| F_inst | 43 | 2 / 4 | 2 / 2 |
| F_local | 47 | 2 / 6 | 1 / 2 |

但旧 Coordinator 同时接收原始 gaze 数值/文本 cue。这个结果测量的是“Visual 呈现变化 + 原有 Coordinator 数值 cue + 原融合规则”的系统行为，不是单独的热图因果效果。

## 十三、为什么现在会出现“没有提升”

从留痕看，问题不是单一环节，而是几个变量叠加。

### 13.1 热图是可见线索，不是目标身份

当前正确地没有把 gazePoint 直接变成 anchor ID。但这样也意味着热图只告诉模型“某些像素曾被注视”，并不告诉它“这一定是 helicopter2”或“这一定是 chair6”。当多个同类实例同时可见时，Visual 仍然需要靠 RGB、动作和时序完成身份判断。

### 13.2 A-H 的图像组织改变了视觉预算

A-H 发送 16 张图、7056 visual tokens，C0/A-F 只有 1764，B-F 只有 880。A-H 既增加了 cue 图片数量，也改变了媒体类型和 token 预算，所以它的输出变化不能单独归因于热图。

### 13.3 B 路线不是模型看到的 B1 全景

B1 两条样本都因 residual 失败，模型实际看到 B3 tile atlas。B3 的世界姿态索引只保存在外部 JSON，模型没有直接读取；图像 tile 和时间标签之间的映射也没有达到稳定的模型级空间理解。

### 13.4 foveation 对模型来说过于弱且全局改变较大

局部 foveation 不是一个清晰的独立热图，而是全图清晰度连续变化。模型需要从纹理差异中反推出 focus，且仍保留 55% 清晰图像；敏感性实验中两个非零 foveation 都没有被报告为 focus。它更可能改变了对象可辨识度，而不是提供一个可读的 cue。

### 13.5 真实模型提示词要求和视觉媒体之间存在错配

- A-H 的 16 张图按热图/RGB 交替，但 Visual 输出只拿到 view ID 文本映射；没有模型级的图像 token 与 view 的强绑定。
- B-H 要求对 8 个 tile 返回 trace，但实际输入是 2 张大 atlas；模型需要自行切 tile。
- B-F 只收到 1 张大 atlas，却被要求按 8 个 view 报告 cue；scene2 的像素尺度坐标正是这种错配的表现。
- cue_center_uv 的坐标域在 prompt 中没有明写为 `[0,1]`，后处理却假设了这一点。

### 13.6 “关键时刻”尚未真正实现

最新 8 帧是质量过滤后均匀覆盖，不是模型根据时间提示、gaze 稳定性、画面质量和手势可用性进行关键时刻选择。dense v4 有更多帧，但 `G001` 引用集中且曾指向 Unity 黑屏，说明 MLLM 的 frame citation 不能直接当作精确时刻。

### 13.7 本地 MLLM 是否影响 agent 处理

本地 8B 的作用是让输入和视觉编码器可留痕，它没有改变 agent 的抽象架构；最新组织实验只是主动停在 Visual。不能据此断言“远程服务影响了 agent”。旧远程和新本地还没有在完全相同的图片、prompt、候选、采样和 cache 条件下做严格 paired local-vs-remote 对照。

可以确认的是：本地模型实际读到了输入；不能确认的是：8B 的视觉表示能力足以稳定完成多指代、同类实例身份和长期时间证据融合。

## 十四、当前实现状态矩阵

| 需求/环节 | 状态 | 证据或限制 |
|---|---|---|
| 保留原始候选/evaluator schema | 已保留 | 旧完整实验继续使用原 evaluator |
| scene-local anchor 隔离 | 最新路径已验证 | 100 个 scope 审计无跨 scene table |
| `there` 等空间表达不扩成全场景候选 | 已实现 | 返回 spatial observation，候选为空 |
| 真实 PTS、source frame、时间范围留痕 | 已验证 | frame manifest、request、provenance |
| 34--100 帧密集视频 | 已实际运行 | local v4 100 条，Visual-only |
| A 单主帧/A 主帧加前后两帧参数化比较 | 未运行 | 最新 A-H 使用 8 个关键时间，每时刻一张 heatmap + RGB |
| 连续窗口分块 | 已实际运行 | v6 scene5 5 个连续窗口 |
| A-H 独立热图 + 原 RGB | 已实际运行 | 2 样本、16 图/样本 |
| A-F foveation | 已实际运行 | 2 样本、8 帧视频/样本 |
| B1 方向全景 CPU 尝试 | 已实际运行 | 两条均 residual 失败 |
| B2 深度重投影 | 未构造 | 没有可靠 depth/mesh |
| B3 spatial atlas | 已实际运行 | 两条均作为 B 实际输入 |
| B1 全景进入模型 | 未运行 | 因失败显式回退 |
| gaze 数值直接进入最新 Visual | 未进入 | 最新 prompt 禁止直接用数值 |
| Hand 进入最新组织实验 | 未运行 | 该实验停在 Visual |
| Coordinator 进入最新组织实验 | 未运行 | `coordinator_run=false` |
| 最新 A/B 的 Any-hit/Exact/F1 | 未运行 | 无 evaluator，不能编造 |
| 热图进入视觉编码器 | 已验证 | 70 图片哈希、processor、forward hook 全部吻合 |
| 热图低层热点可读 | 部分验证 | 2/2 非零检测，3/3 象限正确 |
| 热图精确中心可读 | 未验证成功 | 中心输出疑似默认/像素格式，schema 不严格 |
| foveation 焦点可读 | 未验证成功 | 非零样本 0/2 检测 |
| 视觉线索改善最终指代 | 未证明 | 最新无 Coordinator；旧系统结果不隔离数值 gaze |
| 同源帧、等 token 预算公平对照 | 尚未完成 | 当前五组 token 和媒体形式不同 |

local v4 的 100 条密集视频运行完成了 100 个事件、242 个 mention 和 100 次事件级 Visual 调用，但正向候选证据只有 2 个 mention；197 条 evidence 中 197 条引用 `G001`，143 条视觉观察也引用 `G001`。旧运行的特定样本中，`G001` 对应视频开头 Unity 黑屏，而模型响应声称在该 view 看见目标。因此历史 v4 的 frame citation 不能被解释为准确的关键时刻定位。后续 v6 将正式 provenance 改成连续窗口/source frame 集合并保留模型引用 ID，但 v6 样本规模有限，仍未运行融合或最终评价。

## 十五、交付文件和复核入口

完整报告就是当前文件：

- [`docs/VISUAL_END_TO_END_REPORT_20261022_zh.md`](VISUAL_END_TO_END_REPORT_20261022_zh.md)

本次生成的结构化审计和可视化入口：

- [`runs/visual_process_report_20261022/audit.json`](../runs/visual_process_report_20261022/audit.json)：所有样本、调用、B1 状态、响应和校验汇总。
- [`runs/visual_process_report_20261022/visual_gallery.html`](../runs/visual_process_report_20261022/visual_gallery.html)：可选择样本和条件查看实际模型输入图片、prompt 和原始响应。
- [`runs/visual_process_report_20261022/input_inventory.csv`](../runs/visual_process_report_20261022/input_inventory.csv)：每个请求实际使用的图片和哈希。
- [`runs/visual_process_report_20261022/frames.csv`](../runs/visual_process_report_20261022/frames.csv)：source frame、PTS、gaze 投影和 H/F 像素变化。
- [`runs/visual_process_report_20261022/call_inventory.csv`](../runs/visual_process_report_20261022/call_inventory.csv)：调用数、视觉 token、耗时、响应候选。
- [`runs/visual_process_report_20261022/source_hashes.json`](../runs/visual_process_report_20261022/source_hashes.json)：报告使用的源代码和结果文件哈希。
- `runs/visual_process_report_20261022/prompts/`：实际发给本地模型的 prompt。
- `runs/visual_process_report_20261022/raw_responses/`：实际模型原始响应。

原始视觉输入和几何文件仍保留在：

- [`runs/local_visual_organization_guidance_probe_20261022_pilot2/`](../runs/local_visual_organization_guidance_probe_20261022_pilot2/)
- [`runs/heatmap_pixel_sensitivity_20261022_scene1_163_V051/`](../runs/heatmap_pixel_sensitivity_20261022_scene1_163_V051/)
- [`runs/local_dense_video_visual_8b_20261022_pilot100_v4/`](../runs/local_dense_video_visual_8b_20261022_pilot100_v4/)
- [`runs/visual_evidence_8b_20261021/pilot100_relaxed_clock/`](../runs/visual_evidence_8b_20261021/pilot100_relaxed_clock/)

关键代码入口：

- [`MultiAgent/visual_organization_audit.py`](../MultiAgent/visual_organization_audit.py)
- [`MultiAgent/gaze_evidence.py`](../MultiAgent/gaze_evidence.py)
- [`MultiAgent/spatial_evidence.py`](../MultiAgent/spatial_evidence.py)
- [`MultiAgent/visual_input_trace.py`](../MultiAgent/visual_input_trace.py)
- [`MultiAgent/local_video_visual.py`](../MultiAgent/local_video_visual.py)
- [`MultiAgent/vision_reasoning.py`](../MultiAgent/vision_reasoning.py)
- [`MultiAgent/run_v3.py`](../MultiAgent/run_v3.py)

本报告审计包由以下无推理命令生成：

```bash
cd /media/kunyu/16TB/spj/Multiagent
PYTHONPATH=. /home/kunyu/miniconda3/envs/trimodal/bin/python \
  tools/build_visual_process_report.py --verify-tensors
```

## 十六、建议如何判定下一轮实验是否可信

下一轮在恢复完整 agent 之前，至少应先满足这些条件：

1. 把 `cue_center_uv` 明确写成 `[0,1]` 归一化坐标，显式标记图像坐标系；后处理拒绝像素值猜测，不再事后转换。
2. 对 A-H、A-F 使用同一批较密 source frame，并控制总视觉 token；否则 A-H 的图片数量优势会混入热图效果。
3. 让每个 view 的 cue 图和 RGB 通过独立、同一 view 的媒体项绑定；B3 atlas 需要让 tile ID、tile 坐标和时间在输入中可直接对应，不能要求模型从一张大图自行恢复 8 个 view 的 schema。
4. 先做不含目标类别语义的 cue 可读性测试：翻转、平移、零值和不同强度，分别检验热图热点和 foveation 清晰区。该测试只验证视觉读取，不拿它当指代精度。
5. 对 foveation 固定可解释的模糊核、清晰区域比例和最小局部对比度，并单独确认模型能读出 focus；当前 0/2 非零焦点检测不足以进入系统级结论。
6. 恢复 74--100 帧的 Visual 输入后，再接回冻结的 Hand 和 Coordinator，保持 candidate mapping、时间范围、融合规则、图片预算和调用粒度完全相同。
7. 最终 100 条运行同时报告 Visual evidence 变化和 Coordinator 最终集合变化，继续保留漏选、额外对象、同类实例混淆及所有回退，不报告没有 mention GT 支持的 mention accuracy。

在这些条件满足前，当前数据支持的研究判断应写成：**热图输入链路已打通，并能被 8B 模型粗略读取；foveation 输入链路已打通但视觉可读性未建立；两种证据组织方式对最终指代性能的优劣尚未完成公平验证。**
