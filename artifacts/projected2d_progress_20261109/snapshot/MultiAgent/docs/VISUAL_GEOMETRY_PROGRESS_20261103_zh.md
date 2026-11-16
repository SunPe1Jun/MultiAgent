# MultiAgent 视觉与几何实验进度（2026-10-22 至 2026-11-03）

本文根据当前服务器副本中的代码、配置、`runs/` 产物和已有报告整理。原始 Codex 会话已丢失；这里只记录能从代码和保存结果复核的事实。没有新增模型调用，没有重跑训练或完整推理。

## 1. 证据基础与边界

- 仓库：`D:\spj\Multiagent`（服务器 Linux 路径在产物中写作 `/media/kunyu/16TB/spj/Multiagent`）。
- 当前分支：`main`，恢复时 HEAD 为 `812a7944e4dd3390e68cd557297bbbb0e72ba629`，与 GitHub `origin/main` 一致。
- 本轮工作区有 18 个跟踪文件真实修改（958 行新增、87 行删除），另有大量新模块、测试、配置、文档和 `runs/` 产物未提交。
- `runs/` 约 13 GB，包含视频帧、缓存、processor 张量和逐样本结果；`.gitignore` 已排除视频、图片、CSV、NPZ、模型权重和 `runs/**/cache/`。本文只引用小型 summary 和文档，完整产物留在服务器。
- 下方“已验证”只指能由保存产物复核的链路事实；“已验证有效”只指最终指代指标得到稳定改善。两者不能混写。

## 2. 当前实际流程

### 2.1 默认完整 agent

```text
原始指令
  -> ReferentialAgent：逐 mention 解析（文本、span、角色、关系、数量、共指）
  -> 场景内候选映射（scene-local anchor table）
  -> 事件/mention 时间计划
  -> RGB + gaze heatmap + Hand RGB cue + 候选投影提示
  -> VisualAgent：逐 mention 证据
  -> closed-set Coordinator / 融合
  -> 原 evaluator（Any-hit、Exact-set、micro P/R/F1）
```

入口：`python -m MultiAgent.run_v3`（`run_v3.py` → `pipeline.py`）。`HandAgent` 继承 `VisualAgent`，Hand 只作为 Visual 的 RGB 条件线索；没有独立的 gesture 分类器。

### 2.2 新增的诊断/实验分支

这些分支复用 Referential 或原始 evaluator，但**不都执行完整的 Hand → Visual → Coordinator → evaluator**：

- `--local-dense-video-visual`：本地 Qwen3-VL，按真实 PTS 输入 34–100 帧事件视频；停在 Visual，`fusion=NOT_RUN`。
- `--visual-study temporal`：冻结上游，只改变 Visual 收到的 RGB/heatmap/foveation 呈现；旧远程 8B pilot20/pilot100 使用。
- `--visual-study organization`：2 个样本 × 5 种视觉组织（C0、A-H、B-H、A-F、B-F）；只到 Visual。
- `--projected2d-study`：复用原 projected-2D 三 panel 协议，比较 RGB/current heatmap/history heatmap/history foveation；调用原二维 evaluator，但不含 Hand/Coordinator。
- `--align-frame-poses`：PTS/相机姿态诊断，不调用模型。
- `--projected2d-study` 的 `referential_diagnostics` / `projection_diagnostics` / `camera_motion_alignment` / `head_motion_diagnostics`：CPU/本地检查，不进入主流程。

## 3. 相比仓库原有版本的主要代码/流程变动

| 文件/模块 | 变动 | 对实验的含义 |
| --- | --- | --- |
| `baseline.py` | 用 `SPJ_WORKSPACE_ROOT`/`SPJ_TRIMODAL_ROOT`/`SPJ_MULTIAGENT_ROOT`/`SPJ_V3DMD_ROOT`/`SPJ_PREPARED_DATA_ROOT` 解析路径；兼容历史 `/workspace/usr3` 与 `/ai/data/` | 代码可在当前工作区运行；旧 wrapper 仍含硬编码路径 |
| `telemetry.py` | 严格读取多种遥测 JSON；支持缺少数组外壳的修复 | `scene5/408` 等发布文件可读；不改原始数据 |
| `context.py` | `load_ground_truth=False` 支持 Visual-only；GT 不进入上下文 | 新 v6/组织实验可严格避免 GT 读取 |
| `mention_mapping.py` | v10：`fellow`/`woman` 等词义、literal component parent 修复、显式 actor 漏项检查、代词父项/共指处理、`there/here` 不再扩成全场景候选 | 减少漏 mention 和错误 parent；`there` 返回空候选/空间观察 |
| `agents.py` | Referential 最多 4 次尝试；parent_text 错误单独放宽；literal repair + actor coverage 校验 | 处理 `its side panel`、`the crate beside the truck` 等边界 |
| `qwen_backend.py` | 新增本地视频请求（真实 PTS、`do_sample_frames=False`、VideoMetadata ticks）、远程 image 请求、cache/call audit、processor/vision forward trace | 支持 34–100 帧本地 Visual；输入可审计 |
| `vision_reasoning.py` | 新增 `frozen_rgb_gaze_variants` 协议；在冻结 RGB 条件中不暴露候选面板/数值 gaze | 旧 heatmap/foveation 配对实验的 prompt 隔离 |
| `run_v3.py` | 新增 `--base-url`、`--local-dense-video-visual`、`--visual-study`、`--projected2d-study`、`--align-frame-poses`、`--finalize-study`、`--cpu-only` 等入口 | 同一 entrypoint 承载多种诊断；默认完整流程不变 |
| 新增 `gaze_evidence.py` | 世界 gaze 投影、时间窗、Gaussian kernel、heatmap/foveation 渲染 | heatmap/foveation 的实际实现 |
| 新增 `spatial_evidence.py` | B1 方向全景、B2 缺失说明、B3 spatial atlas | B 路线实现；B1 失败时回退 B3 |
| 新增 `visual_study.py`、`local_video_visual.py`、`local_attention_probe.py`、`attention_cue_validation.py`、`heatmap_sensitivity.py`、`visual_organization_audit.py`、`visual_input_trace.py`、`projected2d_attention_study.py`、`projected2d_study_report.py` | 视觉研究、密集视频、cue probe、投影配对、报告和留痕 | 见第 4、5 节 |
| 新增 `frame_pose_alignment.py`、`head_motion_diagnostics.py`、`camera_motion_alignment.py`、`projection_diagnostics.py`、`referential_diagnostics.py`、`reproject_pts_contract.py`、`audit_gaze_coordinates.py`、`audit_projected2d_inputs.py` | PTS 时钟、姿态插值、GT-free 运动诊断、视锥审计、语言协议诊断 | 见第 6 节 |
| 新增测试 | `test_*` 覆盖 context isolation、frame pose、gaze、head motion、local video、projected2d、projection、spatial、visual study、referential retry 等 | 服务器记录 24/44/66 项分阶段通过；本次未重跑全部依赖环境的测试 |

未改动：原 anchor 表、原 projected-2D GT、原 evaluator 的匹配规则、历史预测文件。`run_v3` 默认完整流程仍保留。

## 4. 已实现 / 已跑通 / 已验证有效

### 4.1 状态矩阵

| 能力 | 已实现 | 已跑通 | 已验证有效 | 说明 |
| --- | --- | --- | --- | --- |
| 默认完整 agent（Referential→Visual→Coordinator→evaluator） | 是 | 旧 20/100 条 pilot | 仅小样本，不是最终性能 | 新视觉模块未接回 Coordinator |
| 本地密集视频 Visual（34–100 帧） | 是 | 8B 100 条；30B 64/100 未完成 | 否 | 输入链路可审计；模型时间引用不可靠 |
| 逐 mention Visual / 连续窗口 | 是 | v6 scene5 1 事件 5 窗口；mention pilot2 | 否 | 样本小，无最终指标 |
| gaze heatmap | 是 | 8B pilot20/pilot100、组织 probe、敏感度 | 像素可见；粗象限可读；最终指代未证明 | 精确定位和身份未验证 |
| foveation | 是 | 8B pilot20、组织 probe、敏感度 | 否 | 非零样本焦点检测 0/3；无加速证据 |
| Hand/gesture cue | 是（Hand RGB/遥测） | 旧完整 pilot | 否 | 无独立 gesture 分类器；新视觉研究 `hand_input_to_visual=false` |
| PTS/相机姿态对齐 | 是 | 20 事件/60 帧 | 诊断有效；未接入 heatmap/GT/模型 | 13 validated、7 uncertain fallback |
| B1 全景 | 是 | 2 样本尝试 | 否（两条均 residual 失败） | 实际回退 B3 atlas，未进模型 |
| B2 深度重投影 | 否 | 否 | 否 | 无可靠 depth/mesh |
| B3 spatial atlas | 是 | 2 样本 | 链路已进模型；效果未证明 | 是 tile atlas，不是 panorama |
| Projected-2D 配对 | 是 | 8B INT8 pilot20 | 否（PROVISIONAL_GEOMETRY_NOT_CALIBRATED） | 30B 仅准备 manifest，未运行 |
| 原二维 evaluator | 保留 | pilot20 已调用 | 指标可复算 | 输入/GT 时钟未校准，不能据此判定 cue 无效 |

**结论：当前没有任何视觉线索被证明能稳定提升最终指代性能。** 已证明的是“图片确实进入模型/视觉编码器”和“部分粗粒度 cue 可被读出”；未证明的是“heatmap/foveation 改善了同类实例指代”。

### 4.2 旧完整 agent 结果（历史三帧实验）

来源：`runs/visual_evidence_8b_20261021/pilot20_corrected/`、`pilot100_relaxed_clock/`。这些是远程 8B 完整流程，含 Referential、Hand、Visual、Coordinator 和原 evaluator；不是最新密集视频/heatmap 组织实验。

pilot20（20 条，五组）：

| 组 | Any-hit | Exact-set | Micro-P | Micro-R | Micro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| R | 0.750 | 0.200 | 0.677 | 0.438 | 0.532 |
| H_inst | 0.750 | 0.200 | 0.724 | 0.438 | 0.545 |
| H_local | 0.800 | 0.200 | 0.759 | 0.458 | 0.571 |
| F_inst | 0.750 | 0.200 | 0.808 | 0.438 | 0.568 |
| F_local | 0.650 | 0.200 | 0.731 | 0.396 | 0.514 |

时钟保护拒绝 13/20 例的 gaze/video 对齐；只有 7/20 例至少一帧有效投影热图；其余保留 RGB 并进入分母。H_local 的 +0.05 Any-hit / +0.0398 Micro-F1 是小样本、低有效 gaze 覆盖下的观察，Exact-set 无变化。

pilot100（100 条，五组）：

| 组 | Any-hit | Exact-set | Micro-P | Micro-R | Micro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| R | 0.670 | 0.220 | 0.608 | 0.424 | 0.500 |
| H_inst | 0.670 | 0.210 | 0.607 | 0.429 | 0.502 |
| H_local | 0.670 | 0.210 | 0.615 | 0.437 | 0.511 |
| F_inst | 0.650 | 0.220 | 0.651 | 0.408 | 0.501 |
| F_local | 0.630 | 0.210 | 0.625 | 0.399 | 0.487 |

相对 R：H_local Any-hit 0、Exact-set -0.010、Micro-F1 +0.011；F_local Any-hit -0.040、Exact-set -0.010、Micro-F1 -0.013。98/100 条完成冻结上游，2 条上游 extraction 失败；99/100 事件窗口被视频长度裁剪。旧 Coordinator 同时接收原始 gaze 数值/文本 cue，因此这不是“纯 heatmap 因果效果”。

### 4.3 最新密集视频与视觉组织实验

| 运行 | 样本 | 模型 | 完成度 | 关键结果 | 路径 |
| --- | ---: | --- | --- | --- | --- |
| `local_dense_video_visual_8b_20261022_pilot100_v4` | 100 事件 / 242 mention | 本地 8B | 完成 Visual-only | 100 次 Visual 调用；仅 2 个 mention 有正向候选证据；197/199 evidence 引用 `G001`；183 个 `supported` 但无正向证据；`fusion=NOT_RUN` | `runs/local_dense_video_visual_8b_20261022_pilot100_v4/` |
| `..._v6_chunked_scene5` | 1 事件 / 5 窗口 | 本地 8B | 完成 Visual-only | 5 次调用；4 个 mention；3 个正向；引用 `W001` | `runs/local_dense_video_visual_8b_20261022_v6_chunked_scene5/` |
| `..._v6_mention_pilot2` | 2 事件 | 本地 8B | 完成，1 次 Visual 失败 | 7 次调用；7 个 mention；0 正向 | `runs/local_dense_video_visual_8b_20261022_v6_mention_pilot2/` |
| `..._prompt_pilot` | 4 事件 | 本地 8B | 完成，1 个语义失败 | 12 个 mention；1 个正向；`scene5_387` 把整句当对象 mention | `runs/local_dense_video_visual_8b_20261022_prompt_pilot/` |
| `local_attention_guidance_probe_20261022_pilot4` | 4 事件 × 6 条件 | 本地 8B | 完成 probe | 24 次调用；96 条 trace row 中 cue 报告 0/96；正向证据 0 | `runs/local_attention_guidance_probe_20261022_pilot4/` |
| `local_visual_organization_guidance_probe_20261022_pilot2` | 2 事件 × 5 条件 | 本地 8B | 完成 probe | 10 次调用、70 个图片条目；文件/像素/prompt/响应校验全部吻合；A-H/A-F 报告 cue 但中心多为 `[0.5,0.5]`；B-H 0/8、B-F 3/8 且坐标为像素尺度；无 evaluator | `runs/local_visual_organization_guidance_probe_20261022_pilot2/` |
| `heatmap_pixel_sensitivity_20261022_scene1_163_V051` | 1 帧 V051 | 本地 8B | 完成反事实 | 热图象限 3/3 正确；foveation 非零焦点 0/3；中心输出为像素尺度，schema 未限定 | `runs/heatmap_pixel_sensitivity_20261022_scene1_163_V051/` |
| `local_attention_cue_validation_20261022_pilot4` | 4 样本计划 | 本地 8B | **中断** | 仅 `scene1/scene1_163:R` 完成；`H_local` 中断；红圈控制调用 0 | `runs/local_attention_cue_validation_20261022_pilot4/INTERRUPTED.json` |

v4 的 `G001` 在多条样本中对应视频开头 Unity 黑屏，而模型声称看见 drone/crate/truck；因此模型返回的 `frame_id` 不能当作精确观测时刻。v6 改为连续窗口/source frame provenance，但样本量很小，仍未接回融合或最终评价。

### 4.4 Projected-2D 配对实验

`runs/projected2d_attention_qwen8b_20261022_int8/pilot20/`：20 个事件 × 4 组（RGB、CURRENT_HEATMAP、HISTORY_HEATMAP、HISTORY_FOVEATION），本地 8B INT8，三 panel paired crop，原二维 evaluator。16/20 个事件四组均成功，共 64 次成功 Visual 调用；4 个事件在冻结 Referential 阶段失败，列入分母但记 NOT_RUN。

| 输入组 | Temporal F1 | Point@100 | Joint@100 | Joint@200 |
| --- | ---: | ---: | ---: | ---: |
| RGB | 0.4835 | 0.1538 | 0.1099 | 0.2198 |
| CURRENT_HEATMAP | 0.5217 | 0.1087 | 0.0870 | 0.2609 |
| HISTORY_HEATMAP | 0.5714 | 0.1538 | 0.0879 | 0.2418 |
| HISTORY_FOVEATION | 0.5000 | 0.1087 | 0.0652 | 0.2391 |

状态：`PROVISIONAL_GEOMETRY_NOT_CALIBRATED`。旧 RGB/遥测时钟配对存在明显疑点；原 GT 是 anchor 代表点投影，不是人工逐 mention 像素标注；RGB 基线本身已含 gaze crop。历史热图 Temporal F1 提高约 8.79 个百分点，但严格空间/时空定位未改善；不能据此宣称 heatmap 有效或无效。

`runs/projected2d_attention_qwen30b_20261023_pilot100_contract_v3/`：已准备 100 个样本、609 行 manifest、fixed samples 和 original annotation parity（609 行，投影 GT 字段与原归档一致），但**没有 samples 输出目录或模型预测**，即只准备、未运行 30B。

### 4.5 30B 密集视频尝试

| 运行 | 计划 | 完成 | 失败 | 模型调用 | 状态 |
| --- | ---: | ---: | ---: | ---: | --- |
| `local_dense_video_visual_30b_20261023_pilot100` | 100 | 7 | 3 | 20 | 早期尝试，已停止；被下一项取代 |
| `local_dense_video_visual_30b_20261023_pts_pose_pilot100` | 100 | 64 | 2 | 136 | **未完成**；`progress.json` 记录 elapsed 10,020.9s、estimated remaining 5,636.8s |
| `local_dense_video_visual_30b_20261023_pts_pose_smoke` | 1 | 1 | 0 | 1 | 完成 smoke |
| `local_dense_video_visual_30b_20261023_component_repair_smoke` | 1 | 1 | 0 | 1 | 完成 smoke |

30B 密集视频使用 `local_dense_video_visual_v7_pts_pose_fallback_literal_referential_repair`，PTS 姿态、`fusion=not_run`、`hand_input_to_visual=false`、`gaze_numeric_input_to_visual=false`。64/100 不是代表性结果，不能用作全量结论。

## 5. gaze / 手势 / 时序 / 视觉预处理 / foveation / heatmap / 2D grounding 状态

### 5.1 gaze 与坐标

- `gazePoint` 按有限世界交点、`gazeVector` 按世界方向、`cameraGazeDirection` 按相机方向处理；相机四元数为 xyzw camera-to-world，world→camera 用共轭/逆旋转；vertical FOV、方形像素、左上原点。
- 100 个事件、7,918 个 telemetry 样本的数值一致性：`gazeVector` 旋转后与 `cameraGazeDirection` 的事件中位误差约 `1.34e-6°`；`gazePoint` 逆变换到 camera 与 `cameraHitPoint` 的中位距离约 `1.14e-6` 世界单位。
- 这只证明字段内部一致，不证明相机是中心眼/头部/录制相机、视频与 telemetry 外参已标定、Unity 单位是米，或 gazePoint 真的可见/属于哪个 anchor。
- 无可靠 depth 或可重放 mesh 时，`visibility=visibility_unknown`；没有伪造遮挡或对象身份。

### 5.2 手势

- 代码中的 Hand 是 `HandAgent(VisualAgent)`：把记录的 Hand RGB/遥测作为 Visual 的条件线索；没有独立的手势分类、骨架或动作识别模块。
- 旧完整 pilot 的 Hand 条件在流程内；最新 v4/30B/组织实验设置 `hand_input_to_visual=false`，没有把 Hand 送给 Visual。不能把旧 Hand 结果归因到新视觉模块。

### 5.3 时序对齐

- `frame_pose_alignment.py` 用真实容器 PTS、位置线性插值和四元数 SLERP 拟合 RGB/telemetry offset；拟合不使用 GT/gaze。
- 固定 20 条、60 张源图：60/60 恢复 PTS；PTS 比请求时间晚的中位数 21.06 ms、最大 78 ms；13/20 得到 `VALIDATED_CANDIDATE`，7/20 为 `CLOCK_UNCERTAIN`，使用原始 offset 的 `ORIGINAL_CLOCK_FALLBACK_PTS_POSE`。
- PTS 版 GT 投影审计：153 行全部生成新投影，127 行进入视锥；42 行来自 7 个 uncertain 事件，其中 38 行在视锥、4 行画外。
- 这些候选 offset 尚未接入 heatmap、GT manifest 或模型评价；`summary.json` 明确 `heatmap_foveation_changed=false`、`original_evaluator_or_manifest_changed=false`、`model_inference=NOT_RUN`。
- 头动敏感度：60 帧中位/最大 ±50 ms 位移 3.98/135.97 px；10/60 帧角速度 >45 deg/s（最高 165.98 deg/s）。这些是诊断假设，不是精度提升。

### 5.4 视觉预处理与 B1/B2/B3

- 密集视频：每事件 34–100 个源帧，中位 70；Qwen temporal group rate 约 4 groups/s；每个 group 两个连续源帧；processor sampling 关闭；每事件最多 100 帧，视觉 token 估计约 3321–4056。
- v6 连续窗口：100 帧拆成 5 个连续 20 帧窗口，完成 5 次 Visual 调用；保留 source frame 集合和窗口 provenance。
- B1：用相机内参/旋转把 8 视角 RGB/gaze 映射到 equirectangular 球面；两条样本均因 rotation-only warp 光度残差过大回退（覆盖率 27.19%/27.74%，相邻帧 residual 超 5 px 的为 4/7、6/7）。B1 图片没有进入模型。
- B2：无可靠 depth/mesh，标记 `NOT_AVAILABLE`，没有伪造世界坐标 RGB。
- B3：8 个真实视角拼成 4×2 tile atlas（1280×704），带 view ID/PTS 文本；B-H 两张（heatmap+RGB），B-F 一张。模型实际看到的是 B3，不是 B1 全景。B3 的外部 JSON 含世界位置/姿态/时间索引，但模型没有直接读取这些 JSON。

### 5.5 heatmap 与 foveation

- 局部 heatmap：参考帧前最多 2 秒有效 gaze，相机位姿插值（位置线性 + SLERP，gap ≤0.25s），投影到参考帧，只累计 `projected_frustum_only`，Gaussian kernel 2°；`local_history_seconds=2.0`、`decay=uniform`、`display_scale=2.0`。
- H_local：红色弱叠加，最大 alpha 0.28；F_local：2/4/8 倍降采样再上采样金字塔，至少保留 0.55 清晰 RGB；`foveation_max_blur_sigma_px=18` 没有直接作为平面 Gaussian sigma。
- 单帧敏感度：热图翻转/零值/原图 3/3 象限正确；foveation 翻转/零值/原图非零焦点 0/3；模型中心输出 `[100,100]`/`[400,400]`，prompt 未明确 `[0,1]` 归一化，不能称为严格 schema 违例，但也没有稳定精确中心。
- 8 帧视频 cue probe：96 条 trace row 中 cue 报告 0/96；说明“像素中存在 heatmap”与“模型在视频 Visual 任务中稳定抽取 cue”之间仍有明显差距。
- 组织实验中 A-H 发送 16 张图（8 heatmap + 8 RGB），A-F 发送 8 帧 foveation 视频，B-H 2 张 atlas，B-F 1 张 atlas；视觉 token 分别约 7056、1764、1760、880，预算不等，不能做公平的效果归因。

### 5.6 2D grounding

- `projected2d_attention_study.py` 复用原 `exam2/build_2d_eval_manifest.py` 选帧、offset、GT 投影，以及 `exam2/run_qwen3vl_2d_point_grounding.py` 的 `multi_image + paired_crop` 和 `exam2/evaluate_2d_point_grounding.py` 的 T/Point/Joint 指标。
- 8B INT8 pilot20 完成，但状态为 `PROVISIONAL_GEOMETRY_NOT_CALIBRATED`；投影 GT 是 anchor 代表点，不是人工像素标注；时钟配对未校准；prompt 去掉了 annotation-derived `expected_count`，因此是 given-expression 扩展，不是旧 wrapper 逐字复现。
- 30B contract_v3 只准备 manifest，未运行模型；没有任何 30B 2D 指标。

## 6. 已知问题与未完成事项

1. **时钟/外参未校准**：旧 metadata offset 与真实 PTS 有中位 21 ms 差异，头动下 ±50 ms 可引起中位 3.98 px、最大 135.97 px 位移；7/20 事件仍为 `CLOCK_UNCERTAIN`。PTS 候选尚未接入模型输入或 GT。
2. **GT 是投影代理**：projected-2D GT 来自 anchor 世界点投影，不是人工逐 mention 像素点；部分投影点与可见物体明显偏移；153 行与原归档字段一致，但“与视频像素对齐”未验证。
3. **Referential 上游失败**：固定 20 条中 4 条因 `parent_text` 不在 mention 字面子串而失败（`scene2/592`、`scene2/416`、`scene4_room3/151`、`scene4_room4/163`）；旧 pilot 冻结失败并保留分母，新代码有重试/修复但未重跑全部旧样本。
4. **模型时间引用不可靠**：v4 的 `G001` 大量指向视频开头黑屏，模型却报告看见目标；不能把 `frame_id` 当精确时刻。
5. **视觉预算不等**：A-H 7056 tokens、C0/A-F 1764、B-H 1760、B-F 880；媒体形式也不同，不能把 A/B 差异归因于单一 cue。
6. **B3 与 schema 不匹配**：B3 是一张大 atlas，却要求按 8 个 view 返回 cue；`cue_center_uv` 未明确 `[0,1]`，后处理却按归一化坐标解释；模型中心多退回 `[0.5,0.5]` 或像素尺度。
7. **foveation 可读性未建立**：非零样本焦点检测 0/3；没有推理加速证据；不能把“图像被改变”当作“模型使用了 foveation”。
8. **新视觉模块未接回 Coordinator/evaluator**：v4、v6、组织实验、cue probe 都停在 Visual；没有最终 Any-hit/Exact/F1；不能把“流程跑通”写成“性能提升”。
9. **30B 实验未完成**：密集视频 64/100；projected-2D 仅准备 100 样本/609 行 manifest；cue validation 在第一个条件后中断。
10. **旧 v4 GT 上下文读取不确定**：`ground_truth_load_unknown_events=100`；虽然 GT 未用于预测，但不能称为严格 GT-free。v6 已改为 `load_ground_truth=False` 并有 context isolation 测试，但样本小。
11. **没有 mention 级人工 GT**：不报告 mention accuracy；原 evaluator 按物理 referent 匹配，重复 mention 可能带来额外惩罚。
12. **依赖 Trimodal-Referring**：数据、anchor 表、projected-2D evaluator 和 GT 来自 `../Trimodal-Referring`；不要把一个项目的结果写入另一个仓库。

## 7. 关键入口与复现命令

环境（当前服务器路径示例；按实际位置替换）：

```bash
export SPJ_WORKSPACE_ROOT=/media/kunyu/16TB/spj
export SPJ_TRIMODAL_ROOT=$SPJ_WORKSPACE_ROOT/Trimodal-Referring
export SPJ_MULTIAGENT_ROOT=$SPJ_WORKSPACE_ROOT/Multiagent
export SPJ_V3DMD_ROOT=$SPJ_WORKSPACE_ROOT/V3dMD
export SPJ_PREPARED_DATA_ROOT=$SPJ_V3DMD_ROOT/prepared/data
source /home/kunyu/miniconda3/etc/profile.d/conda.sh
conda activate trimodal
cd "$SPJ_MULTIAGENT_ROOT"
```

默认完整 agent：

```bash
python -m MultiAgent.run_v3 \
  --manifest "$SPJ_MULTIAGENT_ROOT/runs/pilot20/manifest.json" \
  --output-dir "$SPJ_MULTIAGENT_ROOT/runs/reproduce_v3" \
  --cache-dir "$SPJ_MULTIAGENT_ROOT/runs/reproduce_v3/cache/qwen" \
  --experiment V3
```

密集视频 Visual-only：

```bash
python -m MultiAgent.run_v3 --local-dense-video-visual \
  --manifest runs/visual_evidence_8b_20261021/pilot100_relaxed_clock/manifest.json \
  --output-dir runs/local_dense_video_visual_8b_20261022_pilot100_v4 \
  --cache-dir runs/local_dense_video_visual_8b_20261022_cache_v4_pilot100 \
  --qwen-model "$SPJ_WORKSPACE_ROOT/Qwen3-VL-8B-Instruct" \
  --visual-fps 4 --visual-min-frames 32 --visual-max-frames 100
```

Projected-2D 配对：

```bash
python -m MultiAgent.run_v3 --projected2d-study --split smoke4 --quantization int8
python -m MultiAgent.run_v3 --projected2d-study --split pilot20 --quantization int8
```

PTS/姿态与投影诊断：

```bash
python -m MultiAgent.head_motion_diagnostics \
  runs/projected2d_attention_qwen8b_20261022_int8 --verify-saved-frame-pts
python -m MultiAgent.projection_diagnostics runs/projected2d_attention_qwen8b_20261022_int8
python -m MultiAgent.camera_motion_alignment runs/projected2d_attention_qwen8b_20261022_int8 --all --reuse-existing
```

测试（部分需要本地数据/模型；服务器记录分阶段 24/44/66 项通过，本次未重跑全部）：

```bash
python -m unittest MultiAgent.test_context_isolation MultiAgent.test_frame_pose_alignment \
  MultiAgent.test_gaze_evidence MultiAgent.test_head_motion_diagnostics \
  MultiAgent.test_local_video_visual MultiAgent.test_projected2d_attention_study \
  MultiAgent.test_projection_diagnostics MultiAgent.test_spatial_evidence \
  MultiAgent.test_visual_study MultiAgent.test_referential_retry
```

## 8. 下一次恢复工作的阅读顺序

1. 本文：当前状态、运行表、限制。
2. [`VISUAL_END_TO_END_REPORT_20261022_zh.md`](VISUAL_END_TO_END_REPORT_20261022_zh.md)：输入、B1/B2/B3、10 次组织实验的完整审阅。
3. [`PROJECTED2D_GEOMETRY_REVIEW_20261023_zh.md`](PROJECTED2D_GEOMETRY_REVIEW_20261023_zh.md)：投影、时钟、语言协议复核。
4. [`HEAD_MOTION_ALIGNMENT_OPTIMIZATION_20261023_zh.md`](HEAD_MOTION_ALIGNMENT_OPTIMIZATION_20261023_zh.md)：PTS 重拟合和优化顺序。
5. [`PROJECTED2D_ATTENTION_STUDY_20261023_zh.md`](PROJECTED2D_ATTENTION_STUDY_20261023_zh.md)：8B INT8 pilot20 的配对结果和限制。
6. `runs/local_dense_video_visual_30b_20261023_pts_pose_pilot100/progress.json`：30B 未完成运行的断点。
7. `runs/projected2d_attention_qwen30b_20261023_pilot100_contract_v3/config.json`、`manifest_build_summary.json`：30B 2D 的准备状态。

## 9. 建议的下一步

1. **先决定是否继续 30B 密集视频**：从 `progress.json` 的 64/100 续跑即可，不需要改代码；但当前结果仍是 Visual-only，不能直接给最终指标。
2. **先解决时钟/GT 再跑 30B 2D**：`projected2d_attention_qwen30b...contract_v3` 的 609 行 manifest 已准备好，但在 PTS 姿态接入 GT manifest、确认 `evidence_acceptable` 之前，不建议启动付费推理。
3. **做公平的 cue 对照**：同一源帧、同一视觉 token 预算、同一媒体形式；A-H 的 16 图与 A-F 的 8 帧视频不能直接比较。
4. **修复 cue schema**：明确 `cue_center_uv` 为 `[0,1]`；B3 atlas 必须让 tile ID/坐标/时间可直接对应，不能要求模型从大图自行恢复 8 个 view。
5. **单独校准 foveation**：固定模糊核、清晰区比例和最小局部对比度，先证明模型能读出 focus，再进入系统级实验。
6. **接回 Hand/Coordinator 后才报告最终指标**：新视觉模块目前停在 Visual；任何 Any-hit/Exact/F1 结论都必须走原 evaluator 并保留失败分母。
7. **保留所有失败/回退**：时钟拒绝、B1 residual 失败、B3 回退、Referential 失败、Visual 解析失败都要保留，不能只统计成功子集。

## 10. 本次整理提交的内容

- 更新 `README.md`、`docs/MULTIAGENT_CURRENT_PROGRESS_zh.md` 的状态说明。
- 新增本文。
- 提交新代码模块、测试、`configs/`、`tools/build_visual_process_report.py` 和 5 份新文档。
- 提交小型 run summary/audit/manifest 文件；不提交视频、图片、CSV、NPZ、模型权重、`runs/**/cache/` 和大型 per-sample 产物。
- 完整 13 GB `runs/` 保留在服务器；本文记录路径和规模。
