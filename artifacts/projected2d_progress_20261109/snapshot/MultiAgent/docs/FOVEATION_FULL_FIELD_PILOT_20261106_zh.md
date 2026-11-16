# 完整画面 Foveation 小试（2026-11-06）

## 要回答的问题

在交互视频指代中，基于过去两秒 gaze 历史的空间变清晰处理，是否能帮助模型选出更正确的最终 referent 集合？测试不把 referent 位置用于构造焦点；也不裁剪候选对象。评价仍是原始 scene referent ID 集合。

## 处理方式

每帧先从冻结的三段注意时长图（早/中/近）求和，形成一个过去两秒的多焦点空间密度场。有效密度经平滑扩展成清晰区域，再用 2×、4×、8×降采样图层构造连续分辨率变化。完整 RGB 画幅一直保留，清晰区域直接使用原图；外围仍混有至少 55% 原 RGB 像素。清晰度强度跟有效注视时长共同缩放。没有时钟/相机支持或没有有效注视时，逐帧精确回退 RGB。

同一 2×2 全视野版式和同一任务提示比较四种条件：

- `RGB`：原 RGB，全热图面板为空。
- `FOVEATION`：RGB 左上面板按多焦点 gaze 历史调整清晰度，热图面板为空。
- `HEATMAP`：原 RGB，加原来的早/中/近热图。
- `FOVEATION_HEATMAP`：foveated RGB 加相同三张热图。

候选表、完整指令、12 张连续画面、相机信息、模型预算和后处理 decoder 均共享。没有画候选投影、对象框、目标裁剪或按答案选焦点。模型先在完整 RGB 里输出对象类别和像素位置，固定的全场景几何 decoder 再转成 scene ID，最后才按原 referent GT 评分。

## 先确认渲染确实生效

固定的 24 交互共 288 帧：142 帧有可用 gaze history 并做 foveation，146 帧因 clock/pose 支持不足逐帧原样回退。审计确认：

- 288/288 条件图片均为 1344×1400；模型预处理均为 11,088 个视觉 token。
- 146 个无支持帧的 `FOVEATION` 与 `RGB` 像素完全相同；`FOVEATION_HEATMAP` 与 `HEATMAP` 也完全相同。
- 142 个有效帧左上 RGB 按预期改变；`FOVEATION_HEATMAP` 的三个热图面板与 `HEATMAP` 逐像素完全一致。
- 平均清晰场占画幅约 0.23%；这说明首版是偏局部的多焦点清晰孔径，不是全图锐化。

## 两轮模型结果

**全局对象 ID 直接选择（24 个旧交互）不能评价视觉处理。** 四条件 96 个输出、63 次不同模型调用，视觉输入校验通过；92 个输出 schema 有效、4 个有 schema 错误。四组最终均为空集合，57 个 GT referents 全部漏掉，micro-F1 全为 0。它复现了 ID 接口地板效应，不能用来证明或否定 foveation。

**完整画面类别＋位置定位（8 个既有代表性交互）是当前可解释的小试。** 8 交互×4 条件共 32 个条件结果、29 次不同模型调用。全部 32 个视觉请求通过输入消费和 schema 校验；总共产生 46 条视觉定位行，之后由完全相同的全场景几何 decoder 评分。

| 条件 | TP / FP / FN | micro-F1 | micro-recall | Any-hit | Exact-set |
|---|---:|---:|---:|---:|---:|
| RGB | 4 / 1 / 16 | 0.320 | 0.200 | 4/8 | 0/8 |
| FOVEATION | 4 / 0 / 16 | 0.333 | 0.200 | 4/8 | 0/8 |
| HEATMAP | 4 / 1 / 16 | 0.320 | 0.200 | 4/8 | 0/8 |
| FOVEATION + HEATMAP | 4 / 1 / 16 | 0.320 | 0.200 | 4/8 | 0/8 |

RGB→foveation 的 micro-F1 差为 **+0.013**，交互 bootstrap 95% 区间 **[0.000, 0.030]**。这个总体微小变化来自一个错误的 `chair3` 预测在 FOVEATION 条件消失；没有新增正确 referent，正确率召回仍是 4/20，Exact-set 仍是 0/8。逐交互 mean-F1 和 exact-set 没有变化，8 个交互中也没有一个被判为 helpful 或 harmful。

热图→热图＋foveation 的 micro-F1 差为 **0.000**，bootstrap 区间 **[-0.041, 0.030]**。`scene2_411` 的错误 `chair3` 消失，同时 `scene1_261` 新增错误 `building1`，总体 TP/FP/FN 不变，逐交互 F1 也没有变化。

## 结论与边界

本轮证明了工程链路：可以把多时刻 gaze 转成不裁剪全画幅的多焦点清晰度场；无效支持会回退；热图和 foveation 可独立操纵；模型确实消费了视频帧。当前这组小样本没有显示 foveation 提高对象指代召回或完整集合选择。单独 foveation 的 +0.013 micro-F1 是错误预测减少带来的总体分数小变动，不能说成更会指代；与热图联合时未看到净收益。

限制也很直接：8 个是已反复使用过的旧交互，不是留出验证；只有 20 个 GT 对象且 16 个漏选；anchor-center 几何关联近似，受相机支持、遮挡和动态对象影响；foveation 的高保真孔径平均很小。直接 ID 接口另有明显地板效应。故结论是“该构造能运行，收益尚未出现”，不能外推为 foveation 无效，也不能声称它解决头动或时序问题。

下一轮应先改善完整任务中的视觉定位覆盖/scene-ID 解码诊断，并冻结真正未见交互；同时在同一批完整指代上预注册清晰孔径宽度和外围保真度，避免把微弱/过强模糊的任意参数与 foveation 概念混为一谈。主要指标仍以最终 referent 集为准，不以热图读取命中或框内定位替代。

## 产物

- 完整画面 foveation × 热图四组对照：`.local-sync/full_interaction_foveation_20261106/index.html`
- 原始 ID 入口地板诊断：`.local-sync/full_interaction_foveation_20261106/analysis.json`
- 类别＋位置 grounding 小试与逐帧查看器：`.local-sync/full_interaction_foveation_grounding_20261106/index.html`
- 冻结协议、源图哈希、逐条原始输出、输入消费审计、预测冻结文件和 GT 后评估均在两个上述运行目录中。
- foveation 的构造代码：[foveation_attention.py](../MultiAgent/foveation_attention.py)；运行与评分：[run_foveation_grounding_pilot.py](../tools/run_foveation_grounding_pilot.py)。
