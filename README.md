# VR-TriRef MultiAgent 实验归档

本仓库归档 VR-TriRef 多模态指代实验代码、可复现实验入口、关键结果和旧版 TriModal-Referring 实验流程，并保留选定实验的冻结响应与评分文字记录。仓库不包含视频、模型权重、逐帧图片缓存或私钥；外部数据需按 [环境与数据说明](docs/ENVIRONMENT_AND_DATA_zh.md) 单独准备。

## 当前状态（2026-11-16）

**版本保存**：本次更新位于独立分支 `archive/multiagent-progress-20261116`，原 `main`/`edbf5e8` 保留。全部实验变体与 [11 月 9 日冻结快照](artifacts/projected2d_progress_20261109/README.md) 一并保存；版本选择、恢复方法和测试限制见 [版本交接](docs/REPOSITORY_SNAPSHOT_20261116_zh.md)。

**Scene 1 全量断点**：800 条 API 输入和 event JSON 已重建，并与旧 20 条核对一致；全量面板、热图和推理尚未启动。此前 Scene 1 的 4 条误跑只是归档子集，不能代表全量实验。详见 [准备状态](artifacts/scene1_heat_crop_20261110/scene1_800_preparation_status.json)。

## 2026-11-09 已完成的实验

**进度已存档**：[2026-11-09实验档案](docs/EXPERIMENT_ARCHIVE_20261109_zh.md)；本地源码、冻结响应与评分快照见 [.local-sync/archives/20261109_projected2d_visual_progress](.local-sync/archives/20261109_projected2d_visual_progress/README.md)。恢复上下文时先读实验档案和快照中的`RESUME_STATE.json`。

**回到原2D基线迭代**：32条原wrapper已端到端重跑（Point/Joint@100=0.1926/0.1630）；同批旧发布预测为0.2121/0.1970。保留原全图＋RGB局部细节、用历史gaze热图调整源分辨率crop，得到 **0.2519/0.2222，Exact=6/30**。当前上涨是已见32条开发样本结果，不能外推到全4000条。

**控制结果**：固定22%当前gaze窗口与镜像热图的Joint均0.2074，真实热图只多1个TP；热图＋foveation两个强度均0.1926，当前没有组合额外收益。9条历史热图为空，按原crop精确回退，实际改变输入23条。固定尺度确认组的误标已纠正重跑，输入/源码漂移和全部失败条件均披露。

本轮各组统一使用原expected_count、物理referent输出、P面板与原评价器。先前逐mention单主图、区域角标等保留为历史诊断，不混入当前成绩。实现、复现命令和边界见[原基线视觉迭代](docs/PROJECTED2D_ORIGINAL_BASELINE_VISUAL_ITERATION_20261109_zh.md)，权威机器记录见[最终核验报告](.local-sync/2d_original_visual_results32_20261109/FINAL_VERIFIED_REPORT.json)。

## 历史状态（2026-11-06）

**2026-11-06 全交互指代对照**：恢复整段全画面/完整候选。24交互首轮真实热图比无图Micro-F1高2.86点但配对区间含零，Exact无变化，空间镜像/时间反转表现接近；8交互自适应接口回归里真实热图低于无图。尚无可靠收益证据，均为旧样本。详见[实验报告](docs/FULL_INTERACTION_REFERRING_20261106_zh.md)。

**2026-11-06 实验目标纠正**：最新GT条件A/B配对和局部放大只保留为辅助诊断；主目标恢复为全帧/交互时序中由模型选择指代对象，使用既有referent GT事后评价最终集合。恢复后的完整对照尚未运行，见 [当前进度](docs/MULTIAGENT_CURRENT_PROGRESS_zh.md)。

**2026-11-06 热图可靠性检验**：468次配对/编号控制，局部RGB与图形注意同裁剪改善读图；真实弱代理出现小样本信号，但区间含零、候选类别存疑且改善集中少数旧交互，尚无可靠指代收益。见 [可靠性实验](docs/ATTENTION_RELIABILITY_20261106_zh.md)。

**2026-11-06 对象注意迭代**：增加全帧物体检测/分割、实例视觉编号、mask注视关联和原生视频轨迹记忆；254次本地模型对照尚未显示可靠热图收益。见 [自动调研](docs/OBJECT_ATTENTION_AUTORESEARCH_20261106_zh.md) 与 [实验记录](docs/OBJECT_ATTENTION_EXPERIMENT_20261106_zh.md)。

**2026-11-05 时钟入口统一**：CPU 审计、projected-2D 图像准备和独立 PTS 诊断重投影共用同一时钟/姿态选择。审计仅投影当前交互的标注 referents，GT 在时钟拟合完成后才读取。验证见 [时钟统一与 referent 投影记录](docs/TIME_ALIGNMENT_REFERENT_AUDIT_20261105_zh.md)。

**2026-11-04 时间/投影修复**：统一相机插值和投影、严格核对源帧PTS/图像哈希、检查时钟候选验证覆盖，并在时钟缺少支持时保留RGB。三条真实视频CPU诊断和测试见 [修复与验证记录](docs/TIME_ALIGNMENT_PROJECTION_V2_20261104_zh.md)。新增 `python -m MultiAgent.alignment_projection_audit` 入口，不调用模型。

完整进度、运行表和限制见 [视觉与几何实验进度（2026-10-22 至 2026-11-03）](docs/VISUAL_GEOMETRY_PROGRESS_20261103_zh.md)。

- **已实现**：默认完整 agent；本地密集视频 Visual；heatmap/foveation；PTS/相机姿态诊断；projected-2D 配对；B1/B3 空间证据；context isolation 和多项单元测试。
- **已跑通**：旧 20/100 条完整 agent pilot；8B 100 条密集视频 Visual-only；8B INT8 projected-2D pilot20；PTS 20 事件诊断；多个 cue probe。
- **未证明有效**：尚无视觉线索稳定提高指代效果的证据。最新 projected-2D 配对已由原2D evaluator评分；完整 Hand/Coordinator 系统的新增收益尚未验证。
- **未完成**：30B 密集视频 64/100；30B projected-2D 仅准备 manifest；cue validation 在第一个条件后中断。

不要把“流程跑通”写成“性能提升”。旧 H_local 的 +0.011 Micro-F1 是小样本、低有效 gaze 覆盖下的观察，且 Coordinator 同时接收数值 gaze，不能作为纯 heatmap 因果效果。

## 当前主流程

```text
原始指令
  -> Referential 逐 mention 解析
  -> 场景内候选映射
  -> 事件/mention 时间帧
  -> RGB + gaze heatmap + Hand cue
  -> Visual 逐 mention 证据
  -> closed-set coordinator
  -> 原评估器
```

当前代码位于 [`MultiAgent/`](MultiAgent/)。默认入口：

```bash
export SPJ_WORKSPACE_ROOT=/media/kunyu/16TB/spj
export SPJ_TRIMODAL_ROOT=$SPJ_WORKSPACE_ROOT/Trimodal-Referring
export SPJ_MULTIAGENT_ROOT=$SPJ_WORKSPACE_ROOT/Multiagent
export SPJ_V3DMD_ROOT=$SPJ_WORKSPACE_ROOT/V3dMD
export SPJ_PREPARED_DATA_ROOT=$SPJ_V3DMD_ROOT/prepared/data
source /home/kunyu/miniconda3/etc/profile.d/conda.sh
conda activate trimodal
cd "$SPJ_MULTIAGENT_ROOT"

python -m MultiAgent.run_v3 \
  --manifest "$SPJ_MULTIAGENT_ROOT/runs/pilot20/manifest.json" \
  --output-dir "$SPJ_MULTIAGENT_ROOT/runs/reproduce_v3" \
  --cache-dir "$SPJ_MULTIAGENT_ROOT/runs/reproduce_v3/cache/qwen" \
  --experiment V3
```

先用 `--limit 1` 检查环境。`--mention-isolation` 让每个 mention 使用自己的候选表；`--mention-focus` 是局部时间窗口实验开关，默认关闭；`--regions` 启用 GroundingDINO/SAM；`--no-gaze` 做 gaze 消融；`--base-url` 使用远程 OpenAI-compatible 服务；不传 `--base-url` 时使用本地 Qwen 权重。

## 诊断/实验入口

```bash
# 本地密集视频 Visual-only（停在 Visual，不含 Hand/Coordinator）
python -m MultiAgent.run_v3 --local-dense-video-visual \
  --manifest runs/visual_evidence_8b_20261021/pilot100_relaxed_clock/manifest.json \
  --output-dir runs/local_dense_video_visual_8b_20261022_pilot100_v4 \
  --cache-dir runs/local_dense_video_visual_8b_20261022_cache_v4_pilot100 \
  --qwen-model "$SPJ_WORKSPACE_ROOT/Qwen3-VL-8B-Instruct" \
  --visual-fps 4 --visual-min-frames 32 --visual-max-frames 100

# Projected-2D 配对：RGB / current heatmap / history heatmap / history foveation
python -m MultiAgent.run_v3 --projected2d-study --split pilot20 --quantization int8

# PTS 时钟/姿态与投影诊断（CPU，不调用模型）
python -m MultiAgent.head_motion_diagnostics \
  runs/projected2d_attention_qwen8b_20261022_int8 --verify-saved-frame-pts
python -m MultiAgent.projection_diagnostics runs/projected2d_attention_qwen8b_20261022_int8
```

## 文档入口

- [当前 MultiAgent 流程与进度](docs/MULTIAGENT_CURRENT_PROGRESS_zh.md)
- [视觉与几何实验进度（2026-11-03）](docs/VISUAL_GEOMETRY_PROGRESS_20261103_zh.md)
- [2026-10-21 gaze heatmap/foveation pilot 报告](runs/visual_evidence_8b_20261021/pilot20_corrected/AUDIT.md)
- [2026-10-22 视觉全过程审阅报告](docs/VISUAL_END_TO_END_REPORT_20261022_zh.md)
- [Projected-2D 热图与 foveation 配对诊断](docs/PROJECTED2D_ATTENTION_STUDY_20261023_zh.md)
- [2026-10-23 投影、时钟与指代表达有效性复核](docs/PROJECTED2D_GEOMETRY_REVIEW_20261023_zh.md)
- [头动偏差与源帧 PTS 复核及优化顺序](docs/HEAD_MOTION_ALIGNMENT_OPTIMIZATION_20261023_zh.md)
- [PTS 版时钟与帧姿态对应结果](runs/pts_pose_alignment_20261023/summary.json)
- [PTS 版 GT 投影审计](runs/pts_pose_alignment_20261023/projection_audit/summary.json)
- [复现、环境和数据准备](docs/ENVIRONMENT_AND_DATA_zh.md)
- [Exact-set 与修正模块记录](docs/EXACT_SET_REPAIR_PLAN_zh.md)
- [旧版 TriModal-Referring 全流程](docs/TRIMODAL_REFERRING_WORKFLOW_zh.md)
- [镜像和工作区备份恢复](docs/CONTAINER_BACKUP_zh.md)

## 已归档结果

关键 JSON、manifest、smoke 报告和 GT 审计位于 [`artifacts/`](artifacts/)，包括：

- unified V3 20 条样本：Any-hit 85%、Exact-set 35%、Micro-F1 61.90%；
- 跨 scene smoke 16 条：Any-hit 87.5%、Exact-set 50%、Micro-F1 67.53%；
- mention focus 四条冻结输入对照：Exact-set 未提升，故保持为可选实验；
- scene4 smoke、GT 审计和中断记录。

这些是研究过程中的可审计记录，不应被解释为完整数据集最终性能。

## 依赖与边界

- 数据、anchor 表、projected-2D evaluator 和 GT 来自同级 [`Trimodal-Referring/`](../Trimodal-Referring/) 仓库。
- `runs/` 是实验工作区，约 13 GB；`.gitignore` 排除视频、图片、CSV、NPZ、模型权重和缓存。仓库只保留小型 summary/audit/manifest。
- 旧 shell wrapper 仍含 `/workspace/usr3` 默认路径；代码通过 `SPJ_*` 环境变量适配当前工作区。

连续关注区域与指令筛选：[报告](docs/ATTENTION_REGIONS_AND_SCREENING_20261106_zh.md)。
