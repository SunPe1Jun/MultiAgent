# VR-TriRef MultiAgent 实验归档

本仓库归档 VR-TriRef 多模态指代实验代码、可复现实验入口、关键结果和旧版 TriModal-Referring 实验流程。仓库不包含视频、模型权重、逐帧缓存、原始运行产物或私钥；这些内容需按 [环境与数据说明](docs/ENVIRONMENT_AND_DATA_zh.md) 单独准备。

## 当前状态（2026-11-03）

完整进度、运行表和限制见 [视觉与几何实验进度（2026-10-22 至 2026-11-03）](docs/VISUAL_GEOMETRY_PROGRESS_20261103_zh.md)。

- **已实现**：默认完整 agent；本地密集视频 Visual；heatmap/foveation；PTS/相机姿态诊断；projected-2D 配对；B1/B3 空间证据；context isolation 和多项单元测试。
- **已跑通**：旧 20/100 条完整 agent pilot；8B 100 条密集视频 Visual-only；8B INT8 projected-2D pilot20；PTS 20 事件诊断；多个 cue probe。
- **未证明有效**：没有任何视觉线索被证明能稳定提升最终 Any-hit/Exact/F1。最新视觉研究停在 Visual，未接回 Hand/Coordinator/evaluator。
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
