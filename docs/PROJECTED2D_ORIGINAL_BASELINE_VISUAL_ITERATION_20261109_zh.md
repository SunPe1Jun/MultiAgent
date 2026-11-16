# 2026-11-09：原2D基线恢复与视觉迭代（最终核验版）

存档入口：[2026-11-09实验进度档案](EXPERIMENT_ARCHIVE_20261109_zh.md)，包含当前断点、历史尝试、源码/响应快照及后续恢复步骤。

本轮目标是回到用户原 Experiment 2 的实际水平，再通过视觉处理涨点。已在相同32条开发交互上完成原wrapper端到端重跑和多轮配对迭代。最佳历史热图crop版本 Point@100=0.2519、Joint@100=0.2222、Exact=6/30，高于同批旧发布预测的0.2121/0.1970、5/30。此为已见开发数据上的实测提升，未证明在完整4000条或独立新数据上稳定提高；热图与foveation的组合仍未得到额外收益。

本文及 [FINAL_VERIFIED_REPORT.json](../.local-sync/2d_original_visual_results32_20261109/FINAL_VERIFIED_REPORT.json) 是本轮有效结论。先前 completion_audit、FIXED_GAZE22_summary 和进度说明中的中间结论以这里为准。

## 原始设置与公平比较

实际调用 TriModal-Referring 的原生 `run_qwen3vl_2d_point_grounding.py`：Qwen3-VL-8B BF16，768输出token，原P1/P2/P3面板、`paired_crop`、768全图＋768局部RGB、原system/user prompt、原部件/区域语义、`paired_canvas_map` 和原2D evaluator。当前环境32条重跑为Point@100=0.1926、Joint@100=0.1630，未逐条复现旧预测；旧环境到当前环境的剩余差别尚未查明。旧冻结预测在相同manifest上的评分单独列出，不用旧预测替代当前端到端运行。

与原发布设置一样，各组都提供GT-derived `expected_count`，但图像处理不使用目标名称、GT点或referent投影。统一该条件是为了与原基线公平配对，不是在新增方法里额外提供GT。此前认可的逐mention时间/共指流程保留为独立代码；本轮涨点不算作已在那个单主图协议上成立。本轮最佳输入仍是原基线的整帧＋局部双视图，没有四宫格。

## 方法

保留全局原RGB及其细节，利用每个P面板之前2秒的三阶段gaze-duration热图。在最近时间箱中计算90%支持质量的范围、加入当前gaze位置和边界余量，将窗口边长限制在源图的22%–45%；从1920×1920原始帧裁局部再放大至768，而非先把全图缩至512再裁。热图留在脚本内部，模型判断真实RGB。

只有时钟/姿态和短时gaze支持成立时才使用历史热图，否则逐帧精确回退原wrapper已有的35% gaze crop。Foveation试验在左侧整帧上按同一热图保留清晰孔径、混合外围分辨率，同时保留独立的原RGB局部。两组foveation强度、镜像控制和固定尺度控制都用相同任务/评价合同；必要的crop位置元数据随图更新。

## 完整32条结果

| 条件 | Point@100 F1 | Joint@100 F1 | Joint TP | Joint Exact@100 |
|---|---:|---:|---:|---:|
| 旧发布预测（同32条回放） | 0.2121 | 0.1970 | 13 | 5/30 |
| 原wrapper当前环境重跑 | 0.1926 | 0.1630 | 11 | 4/30 |
| 真实历史热图引导局部crop | 0.2519 | 0.2222 | 15 | 6/30 |
| 固定当前gaze 22% crop（纠正后） | 0.2370 | 0.2074 | 14 | 5/30 |
| 镜像历史热图crop | 0.2519 | 0.2074 | 14 | 6/30 |
| 热图crop＋foveation，原RGB权重下限80% | 0.2222 | 0.1926 | 13 | 6/30 |
| 热图crop＋foveation，原RGB权重下限95% | 0.2222 | 0.1926 | 13 | 6/30 |
| 固定当前gaze 18% crop | 0.2074 | 0.1778 | 12 | 3/30 |
| 固定当前gaze 27% crop | 0.2370 | 0.1778 | 12 | 5/30 |

所有32条均保留，30条有可评价投影。最佳版本与当前原wrapper都输出69个点，评价GT为66个；Joint TP由11增至15，因此分数提高并非只靠减少输出。相对于同批旧预测13个TP，最佳版本多2个TP。相对当前wrapper的Joint增量+0.0593，按交互配对bootstrap 95%描述区间约[0,0.1280]；相对旧预测增量+0.0253，区间[-0.0469,0.1000]。区间宽且经过开发选择，不能称为统计显著的泛化增益。

开发10条上，真实热图crop的Joint@100=0.2593；其余22条为0.1975。这22条与开发10条不重叠，但以前也做过实验，不是独立留出。

## 处理覆盖及实际改变输入的交互

32条中9条的历史热图完全为空，27个面板精确回退原35% crop。这9条的输入图片、prompt和模型原始输出与当前原wrapper完全一致。全32条分数包含正确的回退策略；另外对真正改变输入的23条作配对：真实热图crop Joint@100=0.2286（12TP），当前原wrapper=0.1524（8TP），旧预测=0.1748（9TP）。对应旧预测差值的描述区间仍跨0。

其余22条中真正改变输入的13条上，真实热图crop为5TP，当前wrapper为4TP，旧预测为3TP；不能把所有22条都称作有效热图处理样本，也不能把回退情况隐藏在整体均值里。每条是否变化和回退原始响应是否相同都保存在最终JSON。

## 对照纠错与收益归因

早期目录 `2d_original_visual_fixed22_confirm22_20261109/HEAT_CROP` 被误标为固定22%控制，实际准备时调用的仍是历史热图选区。该22次模型调用不用于固定尺度结论，按无效控制计入本轮资源消耗。后续 `original2d_locked_crop_control.py` 明确绑定固定22%，逐帧验证bbox后重跑22条，才得到表中 `FIXED22_CORRECTED`。纠正后整体Joint仍0.2074，但Exact是5/30，而非旧误标控制的6/30。

真实热图crop相对纠正后的固定22%与镜像控制都只多1个Joint TP，差异集中在开发10条；确认22条没有增加TP。镜像控制与真实热图在实际处理23条上的差值区间包含0。当前证据支持“保留局部细节、优化局部尺度能够在原设置上恢复并提高分数”，不足以确认“历史热图时序/空间分布带来独立收益”。

固定18%、22%、27%各跑32条；30%只跑开发10条（Joint=0.1852），35%准备出与原wrapper完全相同的图像和prompt后不重复推理。18%及27%的整体Joint均为0.1778，低于22%。两个foveation版本整体Joint都为0.1926，低于不加foveation的热图crop。1120单张高清全图及其foveation只在开发10条验证，Joint均0.0377；没有以“更清楚”代替最终定位成绩。

## 核验与可复现记录

本轮共318次模型调用：32次原生wrapper；286次视觉干预/控制，包括22次误标控制。286次均从冻结图片和prompt重新构建完整input_ids、pixel_values、image_grid，精确匹配实际vision-forward记录。原生32次记录原图哈希、prompt、模型原始输出与代码锁，不追溯声称未做过的vision hook。35项相关回归测试及2项foveation不变量通过。

部分早期准备/CLI辅助脚本在输出冻结后又被编辑，导致这些run的完整源码锁与当前源码不再一致。最终报告逐run明示此漂移，重新核对实际输入/输出和processor消费；没有修改原锁来掩盖不一致。原wrapper、模型配置、原evaluator和主要推理代码的哈希保持一致。旧过程JSON与误标条件保留，最终报告明确列为已被替代，避免继续引用。

关键入口：[run_original2d_heat_crop.py](../tools/run_original2d_heat_crop.py)、[迭代实现](../tools/iterate_original2d_visual.py)、[固定尺度控制](../tools/original2d_locked_crop_control.py)、[输入核验](../tools/verify_original2d_recorded_inputs.py)、[最终评分与审计](../tools/finalize_original2d_recovery.py)。

复用已冻结的32条源数据建立新run：

```bash
source /ai/data/TriModal-Referring/.local-sync/reconstructed_pilot100/env.sh
cd /ai/data/MultiAgent
PYTHONPATH=. python tools/run_original2d_heat_crop.py prepare --scope all --output .local-sync/new_heat_crop_run
PYTHONPATH=. python tools/run_original2d_heat_crop.py infer --output .local-sync/new_heat_crop_run
PYTHONPATH=. python tools/run_original2d_heat_crop.py evaluate --output .local-sync/new_heat_crop_run
PYTHONPATH=. python tools/verify_original2d_visual_iteration.py .local-sync/new_heat_crop_run/HEAT_CROP
```

下一轮以已冻结的热图crop配置作为当前最佳候选，选新交互同时跑原wrapper、候选与尺度控制；原来的baseline不能再被删除。本次仅恢复并超越同批32条的历史分数；组合foveation的额外收益、全量4000条增益，以及逐mention单主图设置上的收益均未实现。
