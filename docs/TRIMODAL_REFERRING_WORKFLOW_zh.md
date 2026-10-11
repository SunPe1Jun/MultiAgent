# TriModal-Referring 旧实验流程整理

## 总体任务

旧仓库围绕 VR-TriRef 的三类互补诊断实验展开。三者输出空间、监督信号和指标不同，数值不能直接横向比较。

| 实验 | 目标 | 输出 | 规模 |
|---|---|---|---:|
| Experiment 1 | 闭集 referent identification | 场景 anchor ID 集合 | 4,000 events |
| Experiment 2 | 时间证据与图像平面定位 | evidence panel ID 与 2D point | 4,000 events |
| Experiment 3 | 无候选 ID 的 3D 方向点假设 | Unity world-coordinate point | 4,000 events |

## 共同准备流程

1. 准备视频、multimodal JSON、metadata 和事件标注。
2. 可选运行 gaze overlay。
3. 生成或修正 `event_annotations.csv`，检查 `t_start/t_peak/t_end` 与 GT。
4. 抽取 event keyframe。
5. 构建 event manifest。
6. 构建 Qwen 输入 CSV，汇总 instruction、gaze、hand 和 event JSON 空间信息。
7. 准备本地 Qwen3-VL / InternVL 权重和 CUDA 环境。
8. 运行对应实验脚本，检查 raw output、overlay 和 evaluator。
9. 生成样本级结果、分场景统计和完成度审计。

详细旧文档仍在原目录的 `TriModal-Referring/docs/`，本仓库提供浓缩版流程，避免把大数据和模型提交到 GitHub。

## Experiment 1：closed-set anchor

核心思想是给模型场景候选 anchor，要求从指令与多模态证据中选择一个或多个已有 ID。典型入口：

```bash
bash scripts/grounding/run_qwen3vl_30b_mention_first_all_scenes_full.sh
```

评估包括 Hit-All/Any-hit、mapped-only hit、exact set、set precision/recall/F1。主要 Qwen3-VL-30B 结果：Hit-All 0.7798、exact 0.3287、micro set F1 0.5399。

## Experiment 2：projected 2D point

该实验把事件和候选/投影面板用于图像平面定位，分离时间证据选择与 2D grounding。典型入口：

```bash
bash exam2/run_qwen3vl_30b_2d_full.sh
bash exam2/run_export_selected_panels.sh
```

评估 Temporal F1、Point@K F1 和 Joint@K F1。它是 projected-2D 诊断，不等价于原生 3D 定位。

## Experiment 3：candidate-free 3D directional

该实验隐藏候选 anchor ID，输出一个或多个 Unity 世界坐标点，再用最近 anchor、margin 和场景归一化误差评估。典型入口：

```bash
bash exam3/run_qwen3vl_30b_3d_directional.sh
```

主要 Qwen3-VL-30B 结果：anchor-set F1 0.4326、Margin-F1@1.0 0.2503、Margin-F1@2.0 0.4105。它是 candidate-free measured point-hypothesis diagnostic，不是无约束 3D 重建、3D box grounding 或 3D IoU。

## 模型与消融

Qwen3-VL-8B 和 InternVL3-38B 均完成三个实验的全量运行。旧流程还包含 Qwen/InternVL 对照、2D point、3D directional、gaze/hand/提示词及严格 hand ablation。严格 hand ablation 的技术审计通过，但仍记录残留手部像素和场景遮挡，不能宣称像素级因果去除。

## 复现边界

旧仓库脚本默认使用服务器绝对路径；恢复后需修改路径或保持 `/workspace/usr3` 布局。旧实验的大型 raw outputs、视频、frame、模型和日志不在本仓库。完整旧版入口可参阅原项目的 `README.md`、`SCRIPT_INDEX.md`、`docs/FULL_WORKFLOW.md`、`full_experiment_flow/` 和 `paper_experiment_evidence/`。
