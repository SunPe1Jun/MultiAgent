# 完整 RGB 主图：保留 referent 任务的呈现调整

> 后续自查（2026-11-09）：下文“首帧偏好”只描述统计现象。2D适配缩减了已有referent动作/时间证据，新增对照表明仅加比较措辞不能修复。60次诊断及具体代码损失见 [逐referent选帧自查](REFERENT_FRAME_SELECTION_AUDIT_20261109_zh.md)。本页旧冻结分数不变。

用户明确：逐 mention 的 referent 任务修改是合理部分，本轮保留；需要调整的是四宫格图像呈现。此前把协议差异都列为需要回退的说法不准确。原发布基线仅作为历史参考，当前主实验不恢复 GT expected_count。

旧四宫格左上显示整帧，其余三格重复全图或显示局部，模型输出整个拼图中的点，再由脚本反变换。新实现每个源时刻只提供一张完整 RGB，没有拼图、局部重复、额外热图或图像上的编号；模型消息用相邻文字标明 P1/P2/P3 和时间。原时序面板仍分别提供，让模型选择最适合当前 referent 的一帧。

“主图坐标”是模型输出的目标位置，不是热图表现形式。例如 `point_xy=[500,400]` 表示所选主图宽度的 50%、高度的 40%；脚本除以 1000 得到源帧归一化坐标，再交给原 Experiment 2 evaluator。完整图像保持比例，所以不用推断哪个图块、哪个 crop 或标题偏移。

保留：原 32 条样本、95 个时序面板及时间、冻结的 mention/共指字段、软时间提示、parser 失败处理、部件整体 parent 的既有任务要求、8B BF16、1536 输出 token 上限和原 2D evaluator。GT 只在预测冻结后用于评分与查看，不用于生成主图或提示。

新增 `MAIN_RGB_512` 与 `MAIN_RGB_768` 两个完整 RGB 条件。512 组与旧四宫格左上全图逐像素一致，用于检查保留原显示细节时的主图协议；768 组直接从源图缩放，增加可见细节。两组提示、帧号文字、mention、坐标和推理设置相同，只改变主图分辨率。相对旧拼图还改变了坐标契约和帧号文字关联、减少冗余视觉 token，不能把前后差异全部归因于单一布局。

本轮先测试主图 RGB 对照，不加入新的 gaze 区域标记或热图；原录屏内已有的可见标记保留。因此这不是去除采集端 gaze 的试验，也不是热图收益试验。gaze 的后续区域辅助应以同一主图和坐标空间为基础单独比较。

## 32 条主图对照结果

| 输入 | Temporal F1 | Point@100 F1 | Joint@100 F1 | Joint Exact@100 |
|---|---:|---:|---:|---:|
| 旧四宫格 FULL_RGB | 0.5354 | 0.0787 | 0.0630 | 0/30 |
| 完整主图 512 | 0.6000 | 0.1000 | 0.0333 | 0/30 |
| 完整主图 768 | 0.5833 | 0.1167 | 0.0500 | 0/30 |

768 主图相对 512 的 Joint@100 增加 1.67 个百分点，交互级配对 bootstrap 95% 区间 [-3.81, 7.48] 点，无法确认分辨率增益。两组主图均未恢复原发布基线水平；相对于旧四宫格同时改变了图片组织、帧号文本关联、坐标契约和视觉 token，不能判断任意单项导致的变化。更直观的输入形式不等于已经提高定位准确度。

原始 8B 同 32 条 Point@100/Joint@100 为 0.2121/0.1970，但原始 expected_count 与现在逐 mention 任务不同，只是历史参考，不代表 referent 任务本身错误。

单主图仍有很强首帧偏好：512 组 48/54 个点、768 组 47/54 个点选 P1；上次四宫格 FULL 为 56/61，原始发布预测为 13/66。主图在时间F1上有所改善，但并未消除帧选择问题。两组保守的显式共指、同帧、近距离去重均无可合并项，Exact 仍为0/30；这不证明不存在跨帧或隐式共指。原 evaluator 对不匹配标签可按空间距离回退，exact 是投影代理 exact，不是已验证的语义对象身份。

32条×2条件保留64记录，其中31条parser有效，实际62次视觉调用均通过 vision forward 输入核验；1条parser失败按原约定保留空结果。512组3条、768组2条模型把点附在ambiguous结果上，按冻结规则拒绝这些点，保留原始输出。190/190主图像素重建一致，95/95的512全图与旧四宫格左上视图相同；62/62次实际processor的图像、网格和完整input_ids重建一致，帧号文字顺序核验通过。30项针对测试通过。样本已被用于开发且评价代理未独立几何校准，不作泛化或gaze收益结论。

查看器：[32 条主图与预测点](../.local-sync/projected2d_main_rgb32_20261109/index.html)。结果与审计：[RESULTS.md](../.local-sync/projected2d_main_rgb32_20261109/RESULTS.md)、[completion_audit.json](../.local-sync/projected2d_main_rgb32_20261109/completion_audit.json)。查看器引用的190张图均存在，尚未做浏览器 JavaScript 运行测试。

实现：[projected2d_main_image.py](../MultiAgent/projected2d_main_image.py)。本地运行目录：`.local-sync/projected2d_main_rgb32_20261109/`。

```bash
source /ai/data/TriModal-Referring/.local-sync/reconstructed_pilot100/env.sh
cd /ai/data/MultiAgent
python -m MultiAgent.projected2d_main_image prepare
python -m MultiAgent.projected2d_main_image infer
python -m MultiAgent.projected2d_main_image evaluate
```

输入、代码、模型配置与评价清单在推理前锁定。旧实验的图像、响应和分数保持归档，新增入口不覆盖它们。
