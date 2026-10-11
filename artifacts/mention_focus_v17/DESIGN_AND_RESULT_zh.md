# 逐 mention 局部时间视图实验

## 方法

保留指令原文解析及字符位置，每次只突出当前 occurrence，同时提供所属动作片段。有效 ASR phrase 提示且时钟通过保护时，使用提示窗口作为主窗口；其他情况使用宽动作阶段，并标为 soft_instruction_phase，不能解释成真实词时间戳。不同 occurrence 不强制选择不同实体。

每个 mention 最多提供 5 张主帧、前后各 1 张上下文帧，保持时间顺序。图像横条明确标注实际 F#、时间、PRIMARY/CONTEXT、mention ID；投影 UV 相对于横条下方原 RGB，避免局部输入重排后模型把第 3 张错误称作 F3。

局部正向证据必须引用实际提供的主帧。没有 support_score*confidence >= 0.35 的有效证据时，使用原始完整事件帧进行第二次搜索；此阈值只是实验触发条件，非校准概率。扩窗后引用主窗口外证据必须返回 temporal_justification，说明与当前 occurrence 的关系。主窗口不永久删除全视频证据。

最终 coordinator 仍保留原全局上下文，但新编号选择必须有对应 mention 的 Visual 正向证据，不能借用另一 occurrence 的证据。明确 ID 或唯一实体候选仍可由语言确定，不伪造视觉证据。协调器的空集合也不再被融合层静默忽略。全局候选表范围和 30% 时钟保护在该对照中保持原策略。

CLI 新增 --mention-focus，默认关闭。可独立于 --mention-isolation 使用。主要实现为 MultiAgent/temporal_focus.py；audit 记录主窗口、来源、主帧、上下文帧、局部/扩窗调用和拒绝证据。

## 四条冻结输入的真实模型对照

冻结 v15 的解析、时间计划、RGB、投影、Hand 观察，仅重跑局部 Visual 和最终 coordinator。此结果比较的是整套局部搜索及选择保护，不是纯窗口单因素消融。

| 指标 | v15 同四条基线 | v17 局部视图 |
|---|---:|---:|
| Any-hit | 100% | 100% |
| Exact-set | 50% | 50% |
| Micro-F1 | 66.67% | 64% |

| 样本 | 基线输出 | v17 输出 | GT |
|---|---|---|---|
| scene2_664 | file1, desk2 | file1, desk2 | file1, desk7, desk8 |
| scene2_474 | chair1, person1, desk2 | chair1, person2, chair5, desk4 | chair7, person1, chair4, desk4 |
| scene3_148 | person1, truck1 | person1, truck1 | person1, truck1 |
| scene4_room3/104 | vase, cabinet, rug, sofa | vase, cabinet, rug, sofa | vase, cabinet, rug, sofa |

scene2_474 两次 chair 的主帧分别为 F3/F4 与 F7/F8/F9，实际证据分别引用 F3/F4、F7；desk4 修正。但 chair 编号仍错，而且 actor 退化到 person2。scene2_664 当前时钟拒绝，宽阶段主帧存在重叠，两次 desk 仍都是 desk2：软阶段不能恢复真实时间。scene4_104 vase 扩窗后仍无视觉支持，最终由唯一候选的语言规则保留，不应宣称其被看见。

第一版 v16 曾回归：单阶段 5 帧被裁成 4 帧而删掉中心 F3，模型在局部重排后仍引用 F3，且唯一候选被 coordinator 省略。v17 已恢复 5 帧覆盖、增加真实帧 ID 横条及唯一候选保留；不得使用 v16 作为最终方案。

41 项测试通过，涵盖重复字符位置、时钟拒绝后不使用 ASR、主帧覆盖、局部/扩窗、未提供帧引用、跨 occurrence 借证据及唯一候选保留。两轮模型运行均已结束。

## 判断和下一步

该实验限制了串时路径，但尚未提高端到端准确率，保持实验开关而不替换默认。模型能对错误身份给出很高信心，局部正向证据阈值不能检测这种错误；在时间校准和候选身份未经验证时不能继续硬收窄窗口。

优先下一项应是同类身份核查：在主窗口提供可信相机姿态下的逐候选 RGB 对照位置/裁剪，人工抽检投影与真实物体是否对应，要求模型比较可见差异，并把“看见 chair 类别”与“证实 chair5 身份”分开。对局部/global 冲突保留两个假设或扩窗复核，而不是靠模型自报高置信直接停止搜索。随后再拆分音频和 telemetry 的校准可靠性。

重现实验：

```bash
/workspace/usr3/miniconda3/envs/trimodal/bin/python -m MultiAgent.smoke_temporal_focus \
  --baseline MultiAgent/runs/cross_scene_smoke_v15 \
  --output-dir MultiAgent/runs/mention_focus_v17 \
  --events scene2_474 scene2_664 scene3_148 scene4_104
```

结果：summary.json；逐帧证据及扩窗原因：traces/<scene>/<event>/trace.json；日志：../mention_focus_v17.log。
