# RGB 区域组织扩大实验与 Exact-set（2026-11-09）

> **重要任务设置更正（2026-11-09）**：以下实验虽让模型在RGB上输出2D点，但随后用场景世界坐标anchor和相机姿态将点解码为anchor ID，并以ID集合评价；其样本/输出/指标均不是当前的 projected-2D setting。所谓“仅语言候选”更不是2D视觉基线：它通过完整场景anchor清单按类别唯一匹配ID。32条结果现仅保留为独立的 RGB→3D-anchor-set 诊断；撤回所有关于当前2D任务的性能、gaze对照和语言基线结论。冻结响应/原历史数值未改。正式更正与下一轮2D协议要求见 [运行目录更正记录](../.local-sync/referent_rgb_expansion32_20261109_repaired/TASK_SETTING_CORRECTION.json)。

用户要求扩大规模，并重新利用已有 referent / 时间片段选择的权衡，重点改进 Exact-set。当前为独立研究入口，不改动此前 8 条冻结结果或默认 agent。

## 已复核的旧工作

- `mention_mapping.py`：原文 occurrence / span、角色、数量、scene-local 类型/别名、部件父对象、共指、显式 ID、嵌套父对象去重。
- `preprocess.plan_frames`：固定全事件覆盖地板，动作 clause 的宽阶段提示，ASR 只补充而不删除基础帧。时钟不可靠时回退。
- `temporal_focus.focus_spec`：突出当前 occurrence、动作上下文、最多五张主帧与前后文；窗口只是软先验。旧四条局部时间实验 Exact 未提升，不能直接把窗口硬收窄当成已证明的优化。
- `weighted_fusion` 与 focus coordinator：单数/数量约束、语言唯一候选与显式 ID 保留，最终逐 mention 合并；不得借另一 occurrence 的视觉证据任意重分配。
- V4 坐标审计：模型整张复合图 0..1000 坐标已可复现，不能改回让模型自行解释子面板坐标。旧 decoder 的双帧均值与 anchor-center 近似是另一层限制。

前一轮 RGB pilot 只传了这些语言字段的简化子集，未充分利用 count、候选映射和软阶段。这轮恢复它们，但不声称所有旧模块已自动证明有效。

## 本轮实现

新模块 `MultiAgent/referent_rgb_selection.py`、入口 `tools/run_referent_rgb_expansion.py`。

1. 共享原 ReferentialAgent 的完整输出，而非只保留文本/ID。真实部件/代词只通过有原文依据的共指链继承；重复同词不强制相同或不同。`its seat` 的 parser parent 字段仅在原文 antecedent、component 和共指都支持时规范化，不猜实例。
2. 复用原 `plan_frames` 的动作阶段与 `focus_spec`。本地尚无可靠语音词时间，这轮只用视频内容时段中的软 clause 阶段，不运行 ASR、不把它称作语音对齐。首轮仍覆盖完整交互的 12 张全视野帧；复核从这些真实帧中取最多 8 张，保留四个全事件覆盖位置，再补当前 mention 的主帧/上下文。没有将当前查询收窄成 GT 条件对象局部任务。
3. RGB 输入仍是完整源图 + 三个脚本组织的 RGB 细节，热图不进入模型。四条件 FULL_RGB / UNIFORM_RGB / GAZE_RGB / MIRRORED_GAZE_RGB 保留。每张图的既有 header 标明真实 Vxxx 与 PTS，完整 RGB 像素不变。
4. 保留整张复合图 0..1000 坐标；允许模型在任一 RGB 面板选点。脚本识别面板、排除标题/letterbox、严格映射回源图。输出引用单个实际 Vxxx 源帧，不再拿一组两帧的投影均值对点。source-frame 支持标志仅表示 ID decoder 可用性，不表示可见性或目标。
5. 原几何距离 0.15 / 次近 margin 0.02 不变。语言类型池仅来自全场景公开 anchor；不读 GT。模糊 here/there 无语言默认身份，只能在动作/空间关系支持具体对象、且有合法视觉点时尝试完整场景的类别兼容解码；否则 unresolved，不能直接删掉或自动补 ID。
6. 集合逐 mention 合并。显式语言 ID、唯一兼容实体可沿用旧规则保留，明确标记语言来源、不宣称视觉可见。单数不合并多个身份；局部坏行/漏行不清空其他合法对象。最终不得由无限制全局 coordinator 再混合身份。
7. 对遗漏、数量不足/冲突、解码失败和同类实例身份，最多一次有全局覆盖的复核。复核不能修改未指定的 occurrence；没有返回的行保留初次记录。额外视觉预算单独计量，复核收益不能全部归为 gaze。

旧交互接口小试曾复现固定首帧/坐标复制，因此在正式新样本之前去掉数值定位示例，并保留 V4 整图坐标约定。原始试读保存在 `.local-sync/referent_rgb_dev_smoke_20261109*/`，未运行这批开发样本的 GT 评分。工程接口通过不代表视觉点正确。

## 冻结样本与边界

种子 202611091；每个八子场景各四条，共 32 条，排除本地可见历史地址及前轮 8 条。选择先于新视频支持/模型响应/本轮 GT，不按结果换样本。32/32 CPU 准备完成；23 条有至少一帧注意支持，9 条 RGB 回退，全部保留分母。

“新”只指当前可用本地清单中未出现；不可用原服务器历史和预训练暴露不能认证。scene4 的四房间合计占样本一半，因此需分场景报告，不能用池化分数代表原 4000 条总体。

这份独立 ID-set 诊断历史上报告三个阶段：`visual_only`（需要合法视觉点和 3D anchor 对应）、`language_assembled`（加入语言唯一/显式/共指集合规则）、`recovered`（有预算上限的复核后），并计算场景 anchor 词汇先验。它们只描述这项 RGB→3D anchor-set 流程，不是 projected-2D 的比较条件。当前 2D setting 的有效视觉对照应使用同一批原始 Experiment 2 面板、同一 point/panel 输出契约和原 evaluator，对比 RGB 与热图/注意处理；不需要“仅语言”基线。

输入/源码/提示在首个新交互视觉调用前锁定，全部预测与响应哈希冻结后才读取 GT，沿用原 evaluator。当前代码 81 项针对与回归测试通过，包含新坐标/集合/冻结边界及旧 referent/time/focus 测试。

## 已完成但不适用于当前 2D setting 的扩大实验

正式运行目录：`.local-sync/referent_rgb_expansion32_20261109_repaired/`。32条语言解析已完成，31通过、1条因生成不存在的重复原文 occurrence 而失败；失败保留分母。128条件结果全部完成并冻结，之后才加载GT。101组不同初始输入中4组对应同一解析失败交互而跳过视觉；实际执行97次初始视觉＋66次复核＝163次模型视觉调用，全部vision消费核验通过，无视觉运行错误。

每个条件的逻辑预算相同：31次有效初始视觉、21次复核；相同输入复用节省了物理调用。首轮11,088视觉token、复核7,392；物理调用合计1,563,408视觉token。复核带来额外预算，不能把跨阶段变化全部归于时间提示或gaze。

准备期工程修复：第一目录的输入核验发现1张中断写入留下的零字节PNG；当时仅完成1条件/2调用，未读取GT。已停止并保存 `ABANDONED.json`，该视觉预测不计入正式评分。新目录沿用相同32条预选地址和全部语言缓存，按源RGB重建并逐像素核验全部输入、修复损坏缓存；缓存必须完整且像素匹配，PNG原子写入，正式推理增加完整输入核验门槛。未改变提示、decoder、模型或按结果换样本。报告页初次生成因未定义统计变量中断，仅修复报告工具后重建；冻结推理源码、响应和评分均未修改。

### 原 evaluator 结果

| 阶段 | 条件 | TP / FP / FN | Micro-F1 | Exact-set |
|---|---|---|---:|---:|
| 视觉点＋解码 | FULL_RGB | 10 / 5 / 59 | 23.81% | 3/32 |
| 视觉点＋解码 | UNIFORM_RGB | 15 / 7 / 54 | 32.97% | 5/32 |
| 视觉点＋解码 | GAZE_RGB | 11 / 6 / 58 | 25.58% | 5/32 |
| 视觉点＋解码 | MIRRORED_GAZE_RGB | 9 / 6 / 60 | 21.43% | 4/32 |
| 加入语言集合规则 | FULL_RGB | 30 / 5 / 39 | 57.69% | 10/32 |
| 加入语言集合规则 | UNIFORM_RGB | 31 / 7 / 38 | 57.94% | 10/32 |
| 加入语言集合规则 | GAZE_RGB | 30 / 6 / 39 | 57.14% | 10/32 |
| 加入语言集合规则 | MIRRORED_GAZE_RGB | 31 / 6 / 38 | 58.49% | 10/32 |
| 一次复核后 | FULL_RGB | 30 / 11 / 39 | 54.55% | 10/32 |
| 一次复核后 | UNIFORM_RGB | 33 / 6 / 36 | **61.11%** | **11/32** |
| 一次复核后 | GAZE_RGB | 31 / 7 / 38 | 57.94% | **11/32** |
| 一次复核后 | MIRRORED_GAZE_RGB | 32 / 10 / 37 | 57.66% | 10/32 |

历史报告曾称作“不使用RGB定位的共享语言先验”：TP29 / FP0 / FN40，Micro-F1 59.18%、Exact 10/32。**该数字不是当前2D setting的语言基线**；它使用场景anchor候选清单并直接按唯一类别匹配anchor ID，评分也是anchor集合。

恢复旧的referent集合规则对本批确实有作用：GAZE_RGB Exact从5升到10，其他组净增5～7条，未损失原有Exact。但该诊断中的anchor词汇先验本身已实现全部10条，且这10条都在scene4；这不是2D视觉实验的基线。因此不能把跨不同样本的旧8条0/8→新32条11/32直接解释为新视觉方法提升。

复核在GAZE_RGB / UNIFORM_RGB各新增1条Exact，均是同一条 `scene1_647` 的 `forklift2`；FULL和MIRROR均无新增Exact。GAZE复核改变6个交互集合，TP+1、FP+1、FN−1，Micro-F1只增加0.80点；FULL复核TP不变、FP增加6，说明无条件追加观察也可能引入误报。

最终GAZE相对FULL的Micro-F1为+3.40点、相对UNIFORM为−3.17点、相对MIRROR为+0.29点。5000次交互配对bootstrap的95%区间分别为[0.00,8.26]、[−8.72,1.51]、[−6.15,6.76]点；离散小样本的零下界不能作为严格阳性证据。Exact相对UNIFORM逐样本完全相同。**该RGB→3D-anchor-set诊断不回答当前2D任务是否有gaze收益；此前“没有正确gaze独立收益证据”的表述对当前2D任务撤回。**

### Exact-set 失败与下一步优先级

GAZE最终11条Exact；其余21条中15条纯漏选、6条漏选和误选并存，没有纯误选型失败。相比该诊断中的anchor-lexical先验，仅新增2个TP但引入7个FP，最终Micro-F1反而低1.24点；这只说明该ID-set流程的行为，不构成当前2D setting的语言基线或方法结论。

- **候选/mention覆盖**：38个GAZE漏掉的GT身份中，7个不在任何字面mention候选池；全部仍存在于完整场景表。这7个分布于5条交互，包含未具体解析的目的地/道路和整条parser失败。这里并非统一删除候选；here/there仍可用全场景视觉解码，但语义和类别需要实际识别。全批共有8个GT身份不在字面池，其中1个经视觉deictic路径恢复。应优先完善关系目的地和上下文类别解析，继续保留全场景搜索与来源记录。
- **视觉点/源帧选择**：GAZE共有111条合法点观测，33条来自缺少几何支持的帧；模型虽然允许在任意RGB面板定位，实际全部选在完整RGB面板，不能据此断言它未利用细节。19个有字面具体候选的mention被模型以non_concrete_reference拒答；明确具体对象与模糊位置的拒答规则仍需独立验证。抽查 `scene2_482` 的V002/V003可见跨类别坐标复用及落点不在对象上；这是助手定性复核，不是人工像素GT。
- **几何/身份可用性**：38个漏选中9个在现有12帧里没有受支持且anchor中心入视野的帧；与候选池缺失可能重叠。111条观测中有28条因anchor中心太远、29条无支持相机/入视野候选、7条同类中心太近而解码失败。这些不是独立错误交互数，也不能由中心入视野证明对象可见。需要可靠源帧与实例对应，单纯改热图样式或窗口宽度不足以解决。
- **时间复用边界**：本轮复用软阶段、highlighted occurrence、primary/context和有覆盖复核，没有可靠ASR词级时间，也没有新增密集帧。不能宣称旧time权衡无用或已被本轮单独证明有效；若下一轮研究时间选择，应在新地址上比较相同额外预算的均匀复核与phase复核。

优先顺序：关系/候选完整性 → 具体对象视觉点及源帧真实性 → 同类实例身份 → 以相同预算隔离时间选择贡献 → 再扩大gaze权重实验。这些已评分32条不应用于当前2D任务结论。后续2D实验须重新按原Experiment 2 panel和projected-2D evaluator定义样本、输出与指标。

### 验证与产物

81项针对与回归测试通过（仅工程接口/该RGB→3D诊断，不验证2D任务设定）；1536/1536输入逐像素重建、完整RGB一致，32条时间计划可复现，源码/提示/响应锁核验通过。初始schema和mention coverage各118/128，复核后有效coverage123/128；4个条件因parser失败未运行视觉，另1个仍有坏行。12组阶段×条件指标与原evaluator一致。

- 原始结果与对照：`analysis.json`、`evaluation/`、`predictions_frozen.json`、`summary.json`、`RESULTS.md`。
- 全输入查看器：`index.html`；逐mention来源/漏多选/源图点复核：`exact_set_review.html`。
- 工程核验：`input_verification.json`、`completion_audit.json`、`tests.log`。
- 代码：`MultiAgent/referent_rgb_selection.py`；`tools/run_referent_rgb_expansion.py`；`tools/report_referent_rgb_expansion.py`；`tools/audit_referent_rgb_expansion.py`。

```bash
source /ai/data/TriModal-Referring/.local-sync/reconstructed_pilot100/env.sh
cd /ai/data/MultiAgent
PYTHONPATH=. python tools/run_referent_rgb_expansion.py select --output 新目录
PYTHONPATH=. python tools/run_referent_rgb_expansion.py prepare-cpu --output 新目录
PYTHONPATH=. python tools/run_referent_rgb_expansion.py prepare --output 新目录
PYTHONPATH=. python tools/run_referent_rgb_expansion.py verify --output 新目录
PYTHONPATH=. python tools/run_referent_rgb_expansion.py infer --output 新目录
PYTHONPATH=. python tools/run_referent_rgb_expansion.py evaluate --output 新目录
PYTHONPATH=. python tools/run_referent_rgb_expansion.py report --output 新目录
PYTHONPATH=. python tools/report_referent_rgb_expansion.py --output 新目录
PYTHONPATH=. python tools/audit_referent_rgb_expansion.py --output 新目录
```

时钟/历史投影、可见性和动态身份仍未绝对标定；增加语义结构不能保证解决视觉身份。不得在这 32 条评分后继续调阈值再当未见验证。
