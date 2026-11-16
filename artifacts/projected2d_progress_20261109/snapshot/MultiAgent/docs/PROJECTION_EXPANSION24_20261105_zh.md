# 扩展24交互的时间/投影诊断（2026-11-05）

本轮从重建pilot100扩展到24个交互：8个子场景各3条，保留原3条检查交互，新增21条。每交互12张真实源帧，固定随机种子20261105；选择发生在投影运行前，不按GT或投影效果筛选。CPU诊断全部完成，没有模型加载或API调用。

当前结果支持继续优先处理运动相关的残余对齐/投影问题。新增交互也存在明显随时刻变化的偏差；右侧/后期帧并非稳定更准。当前没有独立人工像素GT，因此不报告referent定位准确率，也未证明偏差只由时钟造成。

## 运行规模与完整性

- 24/24交互完成，288/288源帧核对SHA-256和时钟等式。
- 254帧有可用选定姿态，34帧缺少合法姿态，保留在总分母；没有边界外补姿态。
- 保存684条referent—帧记录：540条进入视锥但遮挡未知，65条画外，79条缺姿态。逐帧核对全部投影ID严格等于本交互referent集合；没有全场景anchor投影。
- 时钟拟合不读取GT/gaze，GT在拟合完成后才用于审计展示。
- 本轮主程序、时钟拟合参数和选择规则均未修改；关键源码SHA-256与运行开始快照全部一致。未覆盖旧诊断目录。

## 时钟可用性

| 实际采用策略 | 交互数 | 有支持的注意帧 |
| --- | ---: | ---: |
| 候选时钟通过验证 | 8 | 89 |
| 候选未通过，但原时钟有RGB支持 | 5 | 53 |
| 候选与原时钟均缺少足够支持，RGB回退 | 11 | 0 |

共142/288帧生成有时钟支持的注意图，其余保留RGB。候选被拒绝合计16个交互，不等于这16条都没有可用原时钟。候选拒绝原因可重复：时间块不一致12条、低损失offset区间过宽9条、验证旋转残差偏高7条、共同验证未支持5条。

| 子场景 | row_index | 候选通过数 | 原时钟有支持的回退数 | 有支持的注意帧 |
| --- | --- | ---: | ---: | ---: |
| scene1 | 162, 261, 795 | 1 | 0 | 11/36 |
| scene2 | 71, 411, 678 | 2 | 1 | 33/36 |
| scene3 | 109, 577, 747 | 2 | 1 | 35/36 |
| scene4_room1 | 46, 72, 155 | 1 | 1 | 21/36 |
| scene4_room2 | 33, 149, 158 | 0 | 1 | 10/36 |
| scene4_room3 | 19, 94, 102 | 0 | 0 | 0/36 |
| scene4_room4 | 10, 25, 41 | 1 | 0 | 11/36 |
| scene5 | 28, 112, 386 | 1 | 1 | 21/36 |

这些比例描述当前工程门槛下的时钟支持，不是标定准确率。数据来自重建输入，不能当作恢复了原冻结输入。

## 运动分层结果

按当前选定时钟对应的遥测角速度分层；阈值只用于描述，不用于删帧、改时钟或筛选模型输入。

| 运动分组 | 可用帧数 | ±50ms旋转敏感度中位数 | 邻近背景旋转残差中位数 |
| --- | ---: | ---: | ---: |
| 低运动 <10°/s | 154 | 1.10 px | 8.13 px |
| 中等运动 10–45°/s | 59 | 5.62 px | 11.12 px |
| 快速转动 ≥45°/s | 41 | 27.07 px | 27.80 px |

- 敏感度：在当前源帧尺寸下，对九条固定图像射线施加±50ms参考姿态变化，先取九条射线位移中位数，再取两个扰动方向的较大值。±50ms是测试假设，不是估计出的真实时钟误差，也不是实测referent误差。
- 背景残差：复用已保存的RGB特征对，在当前帧±0.4秒内、按特征对中点选局部邻域，取可评价的旋转模型残差中位数；图像最长边为640px。它不包含深度/视差和动态物体保证，不能当作目标定位误差。不同帧邻域可能共享特征对。
- 同一交互内，同时有低运动与快速转动帧的21条中，20条的快速转动局部背景残差更高。该检查为探索性描述，未作显著性检验；支持运动相关问题并非仅原两条交互的现象，不能唯一归因于时间误差。
- 用事件前1/3和后1/3比较旋转敏感度：17条后段较低、6条后段更高或相同、1条不可比。这不是前后段准确率；不能据此只保留后期帧。

## 图像复查中的现象

已浏览8份场景总览，每份固定展示该子场景3个交互的V003/V006/V009/V012；列按时间排列，均使用冻结的时钟规则。以下为定性现象，不构成人工像素精度统计：

- scene5/112、scene5/386：末段仍能看到标记明显漂到天空/远离可见物体，与“后面的帧一定更准”不符。scene5/112虽有原时钟RGB支持，绝对目标投影仍存在问题。
- scene4_room2/33：中段相对接近，末段又明显偏离；扩展到12帧后保留了该现象。
- scene4多个房间及scene3也出现静稳阶段与转动阶段差异；部分候选通过的交互仍有可见偏差。背景运动验证通过不等于相机绝对标定完成。
- 所有对象投影均为已有anchor世界代表点；动态对象、区域语义、遮挡及其与物体可见中心的差异仍需区分。

更合理的下一步是在这批冻结样本中建立少量独立人工像素/区域验证，分开检查静态对象、动态对象和区域referent，再诊断时钟相位与相机模型残差。不能按GT十字是否贴合来调offset。

## 复核入口

产物目录：`/ai/data/MultiAgent/.local-sync/projection_expansion24_20261105/`。

- `review_gallery.html`：离线独立图集，选择交互后可拖动12帧时间滑块；包含实际PTS、时钟状态、角速度和敏感度。全部24×12帧的DOM交互逻辑及脚本语法已检查；未执行真实浏览器渲染测试。
- `scene*_overview.jpg`：8份总览；每行一个交互，四列依次为V003、V006、V009、V012。图内青圈为原时钟、橙色十字为选定时钟，红色为有支持的gaze。
- `summary.json`、`analysis_summary.json`、`within_event_motion_comparison.json`：运行状态、分层统计和同交互内的运动比较。
- `per_frame.csv`、`per_referent_frame.csv`：全部逐帧指标及仅referent投影记录，保留画外UV。
- `selection.json`、`config.json`、`validation.json`：样本选择、冻结配置、源图哈希/时钟/投影范围验证。
- `events/`、`pose_alignment/`：逐样本图像、时钟曲线、姿态契约和背景特征。
- `run_expansion.py`、`summarize_expansion.py`、`write_report.py`：本轮脚本。运行脚本拒绝覆盖已有输出；再次实验应使用新的目录。

本轮是24条开发诊断，不代表全量4000条数据，也未运行新的指代模型性能评价。

## 固定样本清单

row_index均为原loader的零起始行号。

| scene | row_index | event_id | 本交互referents | 候选状态 |
| --- | ---: | --- | --- | --- |
| scene1 | 162 | scene1_163 | helicopter2 | CLOCK_UNCERTAIN |
| scene1 | 261 | scene1_262 | person1, building6 | VALIDATED_CANDIDATE |
| scene1 | 795 | scene1_796 | forklift2, cargo1, forklift1 | CLOCK_UNCERTAIN |
| scene2 | 71 | scene2_72 | file1, desk2 | CLOCK_UNCERTAIN |
| scene2 | 411 | scene2_412 | chair7, desk8, file2, desk4 | VALIDATED_CANDIDATE |
| scene2 | 678 | scene2_679 | chair2, door1, lamp2 | VALIDATED_CANDIDATE |
| scene3 | 109 | scene3_110 | drone2 | CLOCK_UNCERTAIN |
| scene3 | 577 | scene3_578 | van1, building1, car3, car1 | VALIDATED_CANDIDATE |
| scene3 | 747 | scene3_748 | van1, truck1, car3, building2, person1 | VALIDATED_CANDIDATE |
| scene4_room1 | 46 | scene4_47 | curtain1, stove | CLOCK_UNCERTAIN |
| scene4_room1 | 72 | scene4_73 | curtain1, stove | VALIDATED_CANDIDATE |
| scene4_room1 | 155 | scene4_156 | person, cup, dining_table | CLOCK_UNCERTAIN |
| scene4_room2 | 33 | scene4_34 | cloth, chair | CLOCK_UNCERTAIN |
| scene4_room2 | 149 | scene4_150 | chair2, table | CLOCK_UNCERTAIN |
| scene4_room2 | 158 | scene4_159 | basin, cloth, chair | CLOCK_UNCERTAIN |
| scene4_room3 | 19 | scene4_20 | room1 | CLOCK_UNCERTAIN |
| scene4_room3 | 94 | scene4_95 | person, chair | CLOCK_UNCERTAIN |
| scene4_room3 | 102 | scene4_103 | chair, dining_table | CLOCK_UNCERTAIN |
| scene4_room4 | 10 | scene4_11 | curtains1, window | CLOCK_UNCERTAIN |
| scene4_room4 | 25 | scene4_26 | bench, bedroom | VALIDATED_CANDIDATE |
| scene4_room4 | 41 | scene4_42 | pillow, window | CLOCK_UNCERTAIN |
| scene5 | 28 | scene5_29 | drone5, oil_tank2 | VALIDATED_CANDIDATE |
| scene5 | 112 | scene5_113 | truck, warehouseB | CLOCK_UNCERTAIN |
| scene5 | 386 | scene5_387 | drone2, crate, truck | CLOCK_UNCERTAIN |
