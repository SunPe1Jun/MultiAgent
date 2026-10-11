# MultiAgent 当前代码与实验进度

## 研究设想

实验大方向保持不变：从原始指令中提取所有 mention，再结合每条 mention 对应时间窗口内的 RGB、gaze、手势和场景候选信息，选择具体对象集合。系统输出原 anchor ID 集合，并沿用 TriModal-Referring 的评估函数。

## 当前实现

1. `agents.py` / `mention_mapping.py`：指令纯文本解析；保留原文 span、角色、关系、数量和共指；场景内候选映射；部件沿用父对象；显式实例 ID 单独处理；嵌套父对象去重。
2. `asr.py` / `preprocess.py`：ASR 只作为辅助时间；事件基础帧不会被 ASR 删除；offset 超过视频时长 30% 时回退全视频并记录原因。
3. `visualize.py` / `vision_reasoning.py`：真实 RGB、二维 gaze heatmap、Hand RGB cue、候选投影提示和逐 mention Visual；最终使用 closed-set coordinator。
4. `temporal_focus.py`：可选局部时间窗口实验。主窗口无有效证据时扩展到事件窗口，尚未替代默认策略。
5. `evaluate.py` / `run.py`：复用原场景 GT 和评估口径，同时计算 Any-hit、Exact-set、micro precision/recall/F1。

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
