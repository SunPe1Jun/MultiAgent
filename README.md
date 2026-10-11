# VR-TriRef MultiAgent 实验归档

本仓库归档当前 VR-TriRef 多模态指代实验代码、可复现实验入口、关键结果和旧版 TriModal-Referring 实验流程。仓库不包含视频、模型权重、逐帧缓存或私钥；这些内容需要按 [环境与数据说明](docs/ENVIRONMENT_AND_DATA_zh.md) 单独准备。

## 当前主流程

`原始指令 → Referential 逐 mention 解析 → 场景内候选映射 → 事件/mention 时间帧 → RGB + gaze heatmap + Hand cue → Visual 逐 mention 证据 → closed-set coordinator → 原评估器`

当前代码位于 [`MultiAgent/`](MultiAgent/)。默认主入口：

```bash
source /workspace/usr3/miniconda3/etc/profile.d/conda.sh
conda activate trimodal
cd /workspace/usr3
python -m MultiAgent.run_v3 \
  --manifest /workspace/usr3/MultiAgent/runs/pilot20/manifest.json \
  --output-dir /workspace/usr3/MultiAgent/runs/reproduce_v3 \
  --cache-dir /workspace/usr3/MultiAgent/runs/reproduce_v3/cache/qwen \
  --experiment V3
```

先用 `--limit 1` 检查环境。`--mention-isolation` 让每个 mention 使用自己的候选表；`--mention-focus` 是局部时间窗口实验开关，默认关闭；`--regions` 启用 GroundingDINO/SAM；`--no-gaze` 做 gaze 消融。

## 文档入口

- [当前 MultiAgent 流程与进度](docs/MULTIAGENT_CURRENT_PROGRESS_zh.md)
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
