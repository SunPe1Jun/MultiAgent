# 2026-11-16 MultiAgent 版本保存与实验交接

本次提交保存截至 2026-11-10 中断时的实现与实验记录，不新增模型调用。上传分支为 `archive/multiagent-progress-20261116`；原 `main` 的 `edbf5e8` 保留，不合并、不改写既有历史。

## 版本选择

| 版本 | 位置与用途 |
|---|---|
| 更新前仓库 | `edbf5e8`，可建立独立 worktree 重试旧流程 |
| 当前代码 | 本分支的 `MultiAgent/` 和 `tools/`；包含时钟/投影修复、热图、foveation、逐 mention、单主图、配对 crop 及各种控制版本 |
| 11 月 9 日冻结快照 | [artifacts/projected2d_progress_20261109](../artifacts/projected2d_progress_20261109/README.md)；保留当时源码、配置、冻结请求/响应、评分和 SHA-256 清单 |
| 11 月 10 日 Scene 1 四条误跑 | [subset4](../artifacts/scene1_heat_crop_20261110/subset4/HEAT_CROP/summary.json)；仅保留为小样本重跑记录，不作为全场景结果 |
| Scene 1 全量准备断点 | [800 条准备状态](../artifacts/scene1_heat_crop_20261110/scene1_800_preparation_status.json)；记录重建输入的来源、哈希和完成范围 |

旧版独立运行目录示例（在仓库根目录执行）：

```bash
git worktree add --detach ../MultiAgent-before-20261116 edbf5e8
```

11 月 9 日源码在 `artifacts/projected2d_progress_20261109/snapshot/MultiAgent/` 下按原路径保存，其中同级 TriModal 的相关 evaluator/wrapper 快照也已保存。当前 `run_original2d_heat_crop.py` 增加了可选 `--scene`；`.orig` 文件保留筛选前入口。该 runner 仍只读取冻结的 32 条源清单，`--scene scene1` 只会筛出 4 条，**不能用于启动 Scene 1 的 800 条全量实验**。

## 已得到的实验结果

32 条已见交互中，热图仅用于脚本选择源分辨率局部放大区域，模型仍判断完整 RGB 与局部 RGB。原 wrapper、热图 crop、固定尺度及镜像控制的任务和评价口径见 [11 月 9 日实验档案](EXPERIMENT_ARCHIVE_20261109_zh.md)。可在 GitHub 直接访问的权威机器结果为 [FINAL_VERIFIED_REPORT.json](../artifacts/projected2d_progress_20261109/snapshot/MultiAgent/.local-sync/2d_original_visual_results32_20261109/FINAL_VERIFIED_REPORT.json)。

热图 crop 的 Point/Joint@100 为 0.2519/0.2222；原 wrapper 当次重跑为 0.1926/0.1630，同批旧发布预测为 0.2121/0.1970。这是已见样本结果，热图相对固定尺度的独立贡献与 foveation 组合收益均未确立。

11 月 10 日误跑的 Scene 1 四条子集，热图 crop 与复用的原 wrapper 预测均为 Point/Joint@100=0.4444，Joint Exact=2/3；无新增匹配。其结果与 800 条全场景任务分开保存。

## Scene 1 全量实验实际断点

用户要求的范围是 Scene 1 **800 个交互**，不是归档抽样。已核对本地 800 个视频和 800 份多模态 JSON；原 Experiment 2 投影清单覆盖 800 个交互、4,992 条 referent-panel 标注。

旧 `reconstructed_pilot100` 的 Scene 1 CSV 有 800 行，但仅 20 行有效，不能直接当成全量输入。随后从完整指令表和原始样本目录重建了 800 条 API 输入及 800 份 event JSON；指令、样本、视频与窗口字段在旧 20 条上全部一致，校验结果已随本分支保存。

本地新数据在 `.local-sync/2d_heat_crop_scene1_800_20261110/reconstructed_data/`。中断时仅完成输入重建；全量三帧 manifest、时钟/热图、两组视觉输入、模型推理及评价尚未完成。当前无 800 条实验分数，也无运行中的全量实验任务。

后续应扩展独立的全量入口，按 800 条输入构建统一面板，再准备原 wrapper 与热图 crop 配对条件。保持原 prompt、expected_count、768 token 预算、坐标映射和 evaluator 一致；明确记录热图支持和回退比例，冻结预测后评分。

## 校验与数据边界

在本机既有依赖环境执行：

```bash
source /ai/data/TriModal-Referring/.local-sync/reconstructed_pilot100/env.sh
PYTHONPATH=. python -m unittest discover -s MultiAgent -t . -p 'test*.py'
python tools/archive_projected2d_progress.py verify \
  --output artifacts/projected2d_progress_20261109
```

2026-11-16 测试运行 294 项：292 通过、1 跳过、1 报错。报错为未修改的 `MultiAgent.test_reasoning.ReasoningTests.test_original_sets_and_rooms_preserved`：本机 `TriModal-Referring/data` 缺少旧 `scene4_room2/scene4_57` 预测记录，无法建立索引；不宣称全套测试通过。冻结快照内 1,668 个文件哈希校验通过。

本次保留代码与实验变体，完整上传 11 月 9 日的小型文字快照及 11 月 10 日四条运行记录。视频、模型权重、源 RGB、NPZ 和 800 条重建数据仍在本地；快照保存引用与哈希，克隆仓库并不等于已经迁移全部实验数据。历史文件中的绝对路径和 protocol lock 保留原样；更换环境或源码后需新建实验目录、重新准备和冻结输入，不改写旧结果。
