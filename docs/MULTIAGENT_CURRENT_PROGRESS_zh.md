# MultiAgent 当前代码与实验进度

> 当前视觉模块完整审阅见 [2026-10-22 视觉全过程报告](VISUAL_END_TO_END_REPORT_20261022_zh.md)。下文 2026-10-21 gaze pilot 是历史三帧完整系统实验，不代表最新密集视频或热图/空间图集输入。其指标以保存的 `pilot20_corrected/summary.json` 为准；早期文档表格曾记录不一致数值。

## 2026-11-03 当前状态摘要

最新代码、运行和限制的完整记录见 [视觉与几何实验进度（2026-10-22 至 2026-11-03）](VISUAL_GEOMETRY_PROGRESS_20261103_zh.md)。

- **默认完整流程仍在**：Referential → scene-local 候选 → 时间帧 → RGB/gaze/Hand → Visual → Coordinator → 原 evaluator。
- **新视觉模块未接回最终评价**：密集视频、heatmap/foveation 配对、视觉组织 probe 都停在 Visual，不能报告最终 Any-hit/Exact/F1。
- **已跑通但未证明有效**：8B 100 条密集视频 Visual（仅 2 个 mention 有正向候选证据）、8B INT8 projected-2D pilot20（几何为 PROVISIONAL）、PTS 姿态诊断（13/20 validated）。
- **未完成**：30B 密集视频 64/100；30B projected-2D 仅准备 100 样本/609 行 manifest；cue validation 在第一个条件后中断。
- **已验证链路**：heatmap/foveation 图片确实进入视觉编码器；单帧热图粗象限 3/3；但视频 cue probe 的 cue 报告为 0/96，foveation 非零焦点检测 0/3。
- **不能声称**：任何视觉线索已稳定提升最终指代性能；旧 H_local 的 +0.011 Micro-F1 是小样本、低有效 gaze 覆盖，且 Coordinator 同时收到数值 gaze。

## 研究设想

实验大方向保持不变：从原始指令中提取所有 mention，再结合每条 mention 对应时间窗口内的 RGB、gaze、手势和场景候选信息，选择具体对象集合。系统输出原 anchor ID 集合，并沿用 TriModal-Referring 的评估函数。

## 当前实现

1. `agents.py` / `mention_mapping.py`：指令纯文本解析；保留原文 span、角色、关系、数量和共指；场景内候选映射；部件沿用父对象；显式实例 ID 单独处理；嵌套父对象去重。
2. `asr.py` / `preprocess.py`：ASR 只作为辅助时间；事件基础帧不会被 ASR 删除；offset 超过视频时长 30% 时回退全视频并记录原因。
3. `visualize.py` / `vision_reasoning.py`：真实 RGB、二维 gaze heatmap、Hand RGB cue、候选投影提示和逐 mention Visual；最终使用 closed-set coordinator。
4. `temporal_focus.py`：可选局部时间窗口实验。主窗口无有效证据时扩展到事件窗口，尚未替代默认策略。
5. `evaluate.py` / `run.py`：复用原场景 GT 和评估口径，同时计算 Any-hit、Exact-set、micro precision/recall/F1。

## 2026-10-21 Gaze 呈现试运行

沿用 `python -m MultiAgent.run_v3`，新增 `--visual-study temporal`，比较同一 20 条固定 pilot、每例相同 3 帧的五组：R、H_inst、H_local、F_inst、F_local。8B Referential、候选映射、direct event plan、Hand 原 RGB 观察及 Coordinator 提示/规则/图片均冻结；只改变 Visual 收到的 RGB 呈现，最终集合仍走原 evaluator。

| 组别 | Any-hit | Exact-set | Micro-P | Micro-R | Micro-F1 |
|---|---:|---:|---:|---:|---:|
| R | 0.750 | 0.200 | 0.677 | 0.438 | 0.532 |
| H_inst | 0.750 | 0.200 | 0.724 | 0.438 | 0.545 |
| H_local | 0.800 | 0.200 | 0.759 | 0.458 | 0.571 |
| F_inst | 0.750 | 0.200 | 0.808 | 0.438 | 0.568 |
| F_local | 0.650 | 0.200 | 0.731 | 0.396 | 0.514 |

H_local 在该 pilot 的 Any-hit 与 Micro-F1 最高，但五组 Exact-set 相同；时钟保护拒绝 13/20 例的 gaze/video 对齐，只有 7/20 例至少一帧生成有效投影热图，其余明确保留 RGB 并纳入评价。因此不把结果解释为稳定提升。Coordinator 在 100 个组别事件中有 53 个最终集合不同于 Visual 融合前集合；42 个 Coordinator 空选择沿用既有 pipeline 输出。错误、漏选、额外对象和同类实例候选混淆均在逐样本报告中。

运行、输入图、固定样本、模型原始请求/响应缓存、几何检查和逐样本预测位于 [`pilot20_corrected`](../runs/visual_evidence_8b_20261021/pilot20_corrected/RESULTS.md)；详细边界与命令见 [`AUDIT.md`](../runs/visual_evidence_8b_20261021/pilot20_corrected/AUDIT.md)。几何仅做 frustum 投影；没有同步深度、可重放 mesh 或动态对象变换，遮挡与历史点的对象身份都未验证。本轮不构建全景或空间图集，也不声称 foveation 加速推理。

## 关键结果

### unified V3 20 条样本

- Any-hit：17/20 = 85.00%
- Exact-set：7/20 = 35.00%
- Micro-F1：61.90%

### 跨场景 smoke 16 条样本

- scene1：Any-hit 50%、Exact-set 0%
- scene2：Any-hit 100%、Exact-set 0%
- scene3：Any-hit 100%、Exact-set 100%
- scene4 四房间合计：Any-hit 100%、Exact-set 62.5%
- scene5：Any-hit 50%、Exact-set 50%
- 合计：Any-hit 87.5%、Exact-set 50%、Micro-F1 67.53%

49 个 mention 中 46 个进入候选池；40 个 GT 对象全部在候选池中。因此 smoke 中的主要错误是后续实例辨别、漏选和额外对象，而不是 GT 对象完全无法映射。

## 最近的局部时间实验

`--mention-focus` 只用于验证时间隔离，不是当前默认配置。四条冻结输入对照：

| 指标 | 默认 | mention focus |
|---|---:|---:|
| Any-hit | 100% | 100% |
| Exact-set | 50% | 50% |
| Micro-F1 | 66.67% | 64% |

它修正了重复 chair 的串时，但没有改善具体实例编号，说明当前更需要核查投影与同类对象身份，而不是继续增加防御性规则。

## 当前优先级

- 保持核心流程简单：mention 解析、场景映射、时间对应、RGB 多模态判断、集合合并。
- 修正模块应是一次针对同类候选的多模态复核，不依赖 GT 规则，也不默认两个相同词组必须指向不同对象。
- 时间局部化保留为开关；如果不能提高 Exact-set，不替换默认流程。
