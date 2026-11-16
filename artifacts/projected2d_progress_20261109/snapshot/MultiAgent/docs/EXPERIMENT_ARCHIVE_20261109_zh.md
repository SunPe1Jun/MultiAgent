# 2026-11-09 实验进度存档：Projected-2D 视觉优化

归档状态：已结束本轮模型运行，保留当前最佳配置与全部失败对照；本次归档只整理、复制和校验已有记录，不调用模型。

## 研究目标与当前断点

用户目标是在原 Experiment 2 的2D基线水平上，利用gaze热图和foveation改善视觉指代结果。已有时序/相机对齐、源帧PTS和GT投影核查保留；它们是基础工作，不等于完成方法优化或验证绝对GT位置。

当前最佳候选为 **`HEAT_CROP`**：保留原基线完整RGB与局部RGB放大双视图，用历史gaze热图调整局部窗口，并从原分辨率帧裁出细节。相同32条已见开发样本上，Point@100=**0.2519**、Joint@100=**0.2222**、Joint Exact@100=**6/30**。热图加foveation的两个配置均不如只用热图crop；因此“组合方法有独立收益”仍是未完成目标。

当前入口文档为本页；详细方法与纠错见 [原2D基线恢复与视觉迭代](PROJECTED2D_ORIGINAL_BASELINE_VISUAL_ITERATION_20261109_zh.md)。唯一有效的最终分数来源是 [FINAL_VERIFIED_REPORT.json](../.local-sync/2d_original_visual_results32_20261109/FINAL_VERIFIED_REPORT.json)。

## 协议与样本

| 项目 | 本轮实际设置 |
|---|---|
| 模型 | 本地Qwen3-VL-8B-Instruct，BF16，无量化，贪心解码 |
| 输出预算 | 768 tokens |
| 输入 | 原Experiment 2的95个P面板；每个时刻为全图＋RGB局部的原paired形式 |
| 全图/局部 | 原始源帧1920×1920；全图及局部各显示768，paired画布1554×768 |
| prompt与任务 | 恢复原system/user prompt、物理referent输出；各组一致保留原GT-derived expected_count |
| 坐标 | 原`paired_canvas_map`映射回源面板 |
| 评价 | 原Experiment 2 evaluator；@100以原源图像素为单位 |
| 样本 | 32交互，其中30条有可评价投影、66个可评价referent；开发10条＋其余22条 |
| 数据性质 | 两批都曾被使用；其余22条不是独立未见留出 |
| 图像处理 | 不使用GT名称/位置生成注意区域；GT数量仅按原基线协议统一提供 |

本轮恢复原发布任务是为公平比较原基线；此前用户认可的逐mention、共指及软时间处理仍保留在独立实现中。本页的分数不表示已经在逐mention单主图设置中成立，也不把RGB→3D anchor-ID集合实验当作2D评价。

## 最终结果

下表为同32条、同原评价协议；Exact分母30，另外2条仍保留在整体交互清单中。

| 条件 | Point@100 F1 | Joint@100 F1 | Joint TP | Joint Exact@100 |
|---|---:|---:|---:|---:|
| 旧发布冻结预测回放 | 0.2121 | 0.1970 | 13 | 5/30 |
| 原wrapper在当前环境端到端重跑 | 0.1926 | 0.1630 | 11 | 4/30 |
| **真实历史热图引导局部crop** | **0.2519** | **0.2222** | **15** | **6/30** |
| 固定当前gaze 22% crop，纠正后控制 | 0.2370 | 0.2074 | 14 | 5/30 |
| 镜像历史热图crop | 0.2519 | 0.2074 | 14 | 6/30 |
| 热图crop＋foveation，外围原RGB权重下限80% | 0.2222 | 0.1926 | 13 | 6/30 |
| 热图crop＋foveation，外围原RGB权重下限95% | 0.2222 | 0.1926 | 13 | 6/30 |
| 固定当前gaze 18% crop | 0.2074 | 0.1778 | 12 | 3/30 |
| 固定当前gaze 27% crop | 0.2370 | 0.1778 | 12 | 5/30 |

最佳版本相对当前重跑Joint提高5.93个百分点，相对旧发布同批预测提高2.53个百分点。相对旧预测的配对bootstrap区间跨0，开发选择也影响解释；这是已见样本上的数值提升，不是全4000条或新数据上的已验证增益。当前重跑也未逐条完全复现旧发布预测，环境差别仍未排清。

另有只跑开发10条的失败/辅助条件：单张1120高清RGB、单张高清RGB＋foveation的Joint均0.0377；仅宽孔径foveation为0.1852；固定30% gaze crop为0.1852。35%固定crop生成后与原wrapper的图片和prompt完全一致，未重复调用模型。

## 热图、foveation与回退策略

1. 读取每帧之前2秒gaze历史，生成三个时间箱的投影时长热图；保留时钟/姿态支持检查。
2. 使用最近时间箱的90%质量包络、当前gaze位置及1.35倍余量确定局部crop，边长限于源图22%–45%。
3. 从原始源帧裁局部后放大至768；完整RGB视野保留。热图用于脚本分配视觉细节，不要求模型读取另一张伪彩热图。
4. 无支持时精确回退原基线35% gaze crop，既不伪造热图，也不排除该交互。
5. foveation对整帧施加热图驱动清晰度场，并保留独立RGB局部。已测的80%/95%外围原RGB混合权重均未增加收益。

32条中有9条、27个面板的历史热图为空，输入图片、prompt及原始模型响应与当前原wrapper完全一致。真正改变输入的23条上，真实热图crop Joint=0.2286（12TP）、同事件native wrapper=0.1524（8TP）、同事件旧预测=0.1748（9TP）。

开发10条上真实热图crop Joint=0.2593，其余22条为0.1975。后者只有13条真正改变输入；这13条中热图crop比native多1个TP，固定22%和镜像控制也取得相同TP数。因此尚未分离出历史热图时序分布相对局部放大尺度的独立贡献。

## 历史尝试与已纠正问题

| 阶段 | 状态与用途 |
|---|---|
| 时序、相机、PTS、投影审计 | 保留为基础设施；绝对像素校准、遮挡与动态对象仍有限制 |
| RGB点→3D anchor-ID集合诊断 | 设置与当前2D不同，不能引用其F1/Exact作为当前2D结果 |
| 四宫格与缩图后裁细节 | 改变原基线输入/细节和任务合同，造成退化；保留历史记录 |
| 逐mention单主图与P1审计 | 发现语境/时间覆盖损失、顺序敏感等；仅为另一协议下诊断 |
| 绿色点、区域角标 | 研究输入表达的辅助尝试，不属于热图＋foveation方法收益证据 |
| 单主图heatmap-foveation恢复 | 跑通但未涨点，不能代表原wrapper配对结果 |
| 本轮原wrapper视觉迭代 | 当前有效主线，按上表与最终机器报告解读 |

固定22%确认组曾误调用`HEAT_CROP`。该22次调用的控制标签已撤回，原文件保留；随后用显式固定bbox重跑22条。必须引用`FIXED22_CORRECTED`，不能引用旧目录名推断处理方式。

部分准备/CLI脚本在模型输出冻结后又被修改，源码锁存在漂移。最终报告逐run披露当前源码与旧锁差异，实际图片、prompt、响应和processor tensor已独立复核；快照保存的是归档时源码，不声称恢复了被覆盖的历史脚本版本。中间报告和错误的按行序拆分bootstrap不再作为结论入口。

## 工程与记录位置

| 内容 | 本地入口 |
|---|---|
| 当前总进度 | [MULTIAGENT_CURRENT_PROGRESS_zh.md](MULTIAGENT_CURRENT_PROGRESS_zh.md) |
| 方法与完整限制 | [详细实验报告](PROJECTED2D_ORIGINAL_BASELINE_VISUAL_ITERATION_20261109_zh.md) |
| 最终机器结果 | [FINAL_VERIFIED_REPORT.json](../.local-sync/2d_original_visual_results32_20261109/FINAL_VERIFIED_REPORT.json) |
| 原基线运行 | `.local-sync/2d_original_wrapper_recovery_20261109/` |
| 最佳条件开发10条 | `.local-sync/2d_original_visual_fastdev_20261109/HEAT_CROP/` |
| 最佳条件其余22条 | `.local-sync/2d_original_visual_confirm22_20261109/HEAT_CROP/` |
| 纠正后固定控制 | `.local-sync/2d_original_real_fixed22_confirm22_20261109/FIXED_CURRENT_GAZE/` |
| 热图缓存 | `.local-sync/2d_original_gaze_cache32_20261109/`（部分条目引用早期缓存） |
| 最佳方法复现入口 | [run_original2d_heat_crop.py](../tools/run_original2d_heat_crop.py) |
| 配对评分及输入核查 | [finalize_original2d_recovery.py](../tools/finalize_original2d_recovery.py)、[verify_original2d_recorded_inputs.py](../tools/verify_original2d_recorded_inputs.py) |
| 本地归档快照 | [.local-sync/archives/20261109_projected2d_visual_progress/](../.local-sync/archives/20261109_projected2d_visual_progress/README.md) |

本轮共318次模型调用：32次原wrapper，286次干预/控制（包括22次误标控制）。286次干预请求的实际视觉消费及完整processor tensor重建通过；原生32次保留原始输出与输入哈希，不追溯声称做过vision hook。已有35项回归测试与2项foveation不变量通过。本次归档校验318条冻结响应，新增模型调用为0。

快照包含源码/文档、各run的清单/配置/冻结响应/trace JSON、最终评分、样本清单、环境信息和SHA-256索引。原始图像、NPZ热图、视频与模型权重保留原位置；快照记录图片与热图哈希，不重复拷贝这些大文件，因此不是可离线迁移全部数据的独立运行包。

## 复现与继续工作

```bash
source /ai/data/TriModal-Referring/.local-sync/reconstructed_pilot100/env.sh
cd /ai/data/MultiAgent

# 复核本次存档；不调用模型。
python tools/archive_projected2d_progress.py verify \
  --output .local-sync/archives/20261109_projected2d_visual_progress --external

# 在现有32条源数据上建立新的最佳配置复现run；输出目录必须是新的。
PYTHONPATH=. python tools/run_original2d_heat_crop.py prepare \
  --scope all --output .local-sync/new_heat_crop_run
PYTHONPATH=. python tools/run_original2d_heat_crop.py infer \
  --output .local-sync/new_heat_crop_run
PYTHONPATH=. python tools/run_original2d_heat_crop.py evaluate \
  --output .local-sync/new_heat_crop_run
PYTHONPATH=. python tools/verify_original2d_visual_iteration.py \
  .local-sync/new_heat_crop_run/HEAT_CROP
```

后续恢复上下文时先读本页及快照中的`RESUME_STATE.json`。下一轮应冻结新交互，同时跑原wrapper、热图crop候选、固定尺度控制；保留全图与源分辨率细节，并继续逐因素检验foveation清晰孔径、外围损失和时序对应。原GT/evaluator保持配对一致，预测冻结后才评价。原始2D水平上的独立泛化增益、热图＋foveation组合收益、逐mention单主图设置上的优化仍未完成。
