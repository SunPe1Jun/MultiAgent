# 2026-11-09：逐 referent 选帧退化自查

此前“模型仍明显偏向第一帧”只是统计现象，不能当成原因。用户指出应针对不同 referent 独立选择证据帧，本次据此审查代码和原始响应，并完成 60 次本地 8B BF16 诊断调用。全部使用独立完整 RGB 图片，没有新增四宫格、热图或 3D 候选表。旧 32 条实验、响应、代码锁和分数保持冻结。

正确任务是：在完整指令中识别当前表达，结合该表达的语境、时序与可见注意线索，找出所指实例及其证据帧。一个对象在 P1 可见，不能自动证明 P1 对当前表达的指代最有支持。不同表达应独立处理；证据充分时也允许共享帧，不以强制分配不同帧修复统计分布。共指描述对象身份，不应提前绑定各次表达的证据时刻。

## 查到的具体协议损失

1. **既有时间搜索被压缩。** `referent_rgb_selection.grounding_prompt` 原来保留每次表达的 `highlighted_instruction`、`action_context`、`primary_frame_ids`、`context_frame_ids`。适配 2D 时，`projected2d_routed_rgb.prepare_inputs` 只提取软范围及其与评价面板的交集；`projected2d_main_image` 又沿用了这个缩减版本。单图替换四宫格后，没有重新核对这些信息是否足够。
2. **全交互软时间先验没有适配稀疏面板。** 32 条共 77 个 mention，67 个的主窗口覆盖全部现有面板，不能区分该选哪帧；2 个没有面板落在窗口内，也没有显式覆盖不足标志或调用既有全交互恢复帧。这里保留原 P 面板是为原 2D 评价协议一致，不是无意抽错帧；错误在于把全交互计划降成提示后，没有处理候选帧支持发生的变化。软范围来自指令分句，不是真实语音时间，不能据此声称缺失了真实 GT 时刻。
3. **注意线索没有等价接续。** 原 2D wrapper 的 system/user 提示明确解释采集画面中的 gaze 点只能作为弱线索，并提供 gaze 局部参考；主图版使用通用 JSON system 提示，删除了这段解释与放大参考，本轮又未加入替代的脚本注意辅助。采集画面里原有的可见标记没有被清除。尚未隔离文字说明、局部细节和其他改动各自的影响。
4. **任务说明偏向通用可见性。** 主图提示虽写了比较所有帧、不要默认首帧，但没有把“物体可见”和“当前表达指向此实例/时刻”充分分开。旧响应出现“P1 是最早可见帧”的选取理由；新逐帧检查也把命令中的推动/移动写成画面中已发生的动作，缺乏相应视觉依据。不能假设指令要求的操作已经执行。

上述是已核对的代码和协议变化；它们对性能下降的独立因果贡献仍未量化。既有 referent/共指任务保留，不将 GT expected_count 恢复为前提。

## 排除和控制

`normalize` 没有把缺省值补成 P1，也没有把其他帧静默改成 P1；只允许实际提供的帧。所有新调用均独立重建完整 `input_ids`、`pixel_values`、`image_grid_thw`，与实际 processor trace 的哈希完全一致，且核对了真实 vision forward 消费。正反序图片与相邻的实际帧号/时间保持绑定。因而本次未发现图片被漏送、caption 错配或 parser 硬编码 P1 的证据。

固定 10 条已见开发交互，共 29 个 mention；沿用之前的 8 条多 mention 子集，加上用于覆盖诊断的 scene2/412、scene2/482。样本已被看过和评分，本轮结果不能当作独立留出效果。

| 原 P 面板对照 | 输出定位点 | P1/P2/P3 | 逐 mention 检查覆盖 | Point@100 F1 | Joint@100 F1 |
|---|---:|---|---|---:|---:|
| 顺序中性提示，正序输入 | 23 | 20/3/0 | 未要求 | 0.04 | 0.00 |
| 同提示、同图，倒序输入 | 23 | 16/1/6 | 未要求 | 0.08 | 0.04 |
| 加动作语境及逐帧检查，示例漏字段 | 23 | 20/1/2 | 0/29 | 0.04 | 0.00 |
| 仅修正上述 JSON 示例，包含检查字段 | 23 | 20/3/0 | 29/29 | 0.12 | 0.04 |

23 对在正反序均有单点的相同 mention 中，6 对换帧，4 对在两种顺序下都选输入第一张。存在顺序敏感，但不能把所有 P1 选择归因于呈现顺序。正反序之外的条件还改变了提示或输出结构，不能当作单一顺序效应。

本次新增的“逐帧检查”初稿也有错误：正文要求 `frame_checks`，JSON 示例却未包含，29 个 mention 均没有按要求返回。该版的 COMPLETE 只表示原点坐标 schema 合法，不能算完成逐帧比较。修正示例后的 29/29 只验证三个帧号均列出，不验证观察内容真实。仍有 20/23 点选 P1，说明增加字段并未消除退化，不能把解释文字当作视觉证据。以上四组 Joint Exact@100 均为 0/10；未调整集合策略、未强制分帧。

## 复用已有逐 referent 时间覆盖

使用原来冻结的 `temporal_focus` 主帧及前后文帧，不引入 GT、候选对象 ID 或新的目标提案。两个交互共 10 次单 mention 调用，每次收到 5–7 张原有完整 RGB 帧；不同 referent 的帧池可不同。它们是已有 672×672 全画幅 JPEG，与原 P 面板 768 主图条件不同，只用于覆盖诊断。

scene2/482 的原 P 面板为 3.517、4.962、5.518 秒，而后两个表达的软阶段为 9.9605–16.5403 秒。原全交互帧能覆盖后段。恢复后：

| 表达 | 所选真实源帧 | 视频时间 |
|---|---|---:|
| m0：that colleague | V001 | 0.749 s |
| m1：that chair（push） | V001 | 0.749 s |
| m2：there | V001 | 0.749 s |
| m3：that drawer | V003 | 3.623 s |
| m4：that chair（place） | V010 | 13.664 s |
| m5：over there | V012 | 16.540 s |

这证明新的推理入口确实可为不同表达使用不同时间的独立 RGB；不能证明对象或时刻选对，因为候选帧池本身随表达变化。10 个定位点中 7 个仍选各自池子的首张，且允许选择窗口外的上下文帧，例如 scene2/412 的 file 选了 4.246 s 的 V005。这仍是需要处理的退化，不将阶段先验设成硬筛选或真值。

原评价标签只覆盖 P1/P2/P3，不能给新增 V 帧套用原 GT、对齐到最近 P 帧或宣称原 2D 性能恢复。上述 V 帧尚无匹配的独立 2D 标签，本次不为它们报告官方分数；这是本次诊断的边界，不是后续推进的人工标注前提。主评价仍应在原 P 面板上配对验证。

本次单 mention 探针第一版另有 ID 约束错误：模型在 3 条响应中用 `coreference_id=r1/r3` 替换了 `mention_id=m1/m3`，另 1 条 ambiguous 却附点，全部被严格拒绝。第二版只增加明确的 occurrence ID 绑定及对应 JSON 示例，保持同一 10 个任务、图像与其余设置，最终 10/10 正确解析。首版错误属于本次探针，不将它追溯成旧 32 条 P1 统计的原因；不做响应后 ID 猜测或重映射。

## 已完成与仍需解决的部分

已完成：代码路径自查、67/77 与 2/77 覆盖统计、正反序对照、可核查的逐帧检查输出、原主/上下文帧复用、single-mention ID 绑定。60/60 次模型调用均通过 processor 精确重建及 vision 消费检查；旧主图 64 条记录的源码锁与响应哈希仍一致。

尚未解决：当前表达的注意/指示证据不足，模型会以“可见”或“最早可见”替代真正的指代判断；逐帧解释可能编造操作已发生。没有完成新的 32 条配对收益试验，没有恢复原基线准确度，也没有证明 gaze 的独立收益。

后续主线应保持：完整 RGB 主图、原 P 面板及 2D 坐标/evaluator、现有逐 referent 任务。将动作语境、当前表达、软阶段来源和覆盖情况带入每次推理；把 gaze 用于脚本内部的区域/时段注意辅助，保留全图搜索和低支持回退。对象身份、表达出现时刻和最后 exact-set 合并分开处理，不能先把共指行折成同一帧。先用同一 P 面板上的配对控制核对这些线索，再谈扩大规模；新增 V 帧只能辅助检索或作为独立协议，不能混报成原 benchmark 结果。

## 复现与记录

```bash
source /ai/data/TriModal-Referring/.local-sync/reconstructed_pilot100/env.sh
cd /ai/data/MultiAgent
PYTHONPATH=. python tools/audit_referent_frame_choice.py prepare
PYTHONPATH=. python tools/audit_referent_frame_choice.py infer
PYTHONPATH=. python tools/audit_referent_frame_choice.py evaluate
PYTHONPATH=. python tools/probe_referent_frame_checks_v2.py prepare
PYTHONPATH=. python tools/probe_referent_frame_checks_v2.py infer
PYTHONPATH=. python tools/probe_referent_frame_checks_v2.py report
PYTHONPATH=. python tools/probe_referent_temporal_coverage.py prepare
PYTHONPATH=. python tools/probe_referent_temporal_coverage.py infer
PYTHONPATH=. python tools/probe_referent_temporal_coverage.py report
PYTHONPATH=. python tools/probe_referent_temporal_coverage_v2.py prepare
PYTHONPATH=. python tools/probe_referent_temporal_coverage_v2.py infer
PYTHONPATH=. python tools/probe_referent_temporal_coverage_v2.py report
```

上述 prepare 拒绝覆盖已有清单；复现新 run 需使用 `--output` 指定新目录，并注意后续脚本的 SOURCE 默认指向本次冻结目录。GPU 推理进程顺序运行，CPU 审计可独立运行。已有结果可直接只读核验、重建派生评分：

```bash
PYTHONPATH=. OMP_NUM_THREADS=2 python tools/verify_referent_frame_inputs.py \
  .local-sync/referent_frame_choice_audit_20261109 \
  .local-sync/referent_frame_checks_v2_20261109 \
  .local-sync/referent_temporal_coverage_probe_20261109 \
  .local-sync/referent_temporal_coverage_v2_20261109
PYTHONPATH=. python tools/report_referent_frame_audit.py
```

机器结论：[completion_audit.json](../.local-sync/referent_frame_choice_audit_20261109/completion_audit.json)。逐 mention 覆盖：[coverage_audit.json](../.local-sync/referent_frame_choice_audit_20261109/coverage_audit.json)。四个 run 分别保留输入清单、源码锁、原始响应、请求与模型 trace 摘要、冻结响应哈希和 processor 重建核验。大张量只在核验后清理，记录的哈希与输入消费信息保留。
