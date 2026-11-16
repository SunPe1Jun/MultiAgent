# 脚本利用 gaze 组织 RGB 证据（2026-11-09）

用户本轮要求尝试将热图作为脚本判断重点区域的辅助，而非直接交给大模型的视觉证据。新增独立、可冻结和恢复的完整交互试验入口；默认 agent 流程不变。代码位于 `MultiAgent/gaze_rgb_routing.py`，入口为 `tools/run_gaze_rgb_routing.py`。

## 实现

脚本沿用统一 PTS、时钟、姿态与过去两秒 gaze 的三段时长密度。内部密度只用于分配补充 RGB 细节窗口，不渲染为模型图像，不把注视时长或峰值作为文本分数传给模型。每个时间段最多一个窗口，窗口边长为源图短边的 40%；至少 0.10 秒有效支持才启用该时段窗口。两个参数均为预先固定的工程配置，不是经过标定的最优值或可靠性标准。

选择局部时以窗口累计质量为依据，在同一阶段的近似最优窗口中尝试避免重复。这个预算不保证覆盖所有热点，更不声称获得对象边界、跨帧身份或 referent。完整密度和各窗口支持/剩余量保留供审计。没有时钟/姿态支持或支持不足时，对应细节槽回退完整 RGB。

每帧仍使用 1344×1400 四面板：左上是 672×672 完整源 RGB，另外三块为该帧 RGB 的局部放大或完整 RGB 回退。完整源 RGB 各条件逐像素保留；源图已是上游缩放图，局部放大没有创造原视频未提取的高分辨率细节。模型收到原始完整指令、逐 mention 稳定 ID、完整场景类别词表、视频时间索引和细节窗口在源图中的矩形，不收到内部 gaze 数值、热图、GT、目标框或预选对象。

语言准备优先复用原 ReferentialAgent。对 `its seat` 的旧 parent 字段约束，本轮出现原文 mention 有效但候选父对象结构校验失败；保留原请求/错误，只在此类错误下验证原始 mention 的原文 span、重复 occurrence 和显式 actor 覆盖，不修复或传递父对象身份字段。此修正发生在本轮视觉响应/GT 评分之前，四条件共享同一冻结输入。

模型仍为每个 mention 输出 grounded/unresolved。定位点统一落在左上完整 RGB，沿用 V4 复合图坐标校验和原有 GT-free scene-ID decoder。候选表不按注视筛除，GT 仅在所有响应冻结后用于原评价器。关系位置短语沿用 V4.1 的非具体对象处理原则。

## 冻结四条件

| 条件 | 模型所见补充区域 | 要区分的因素 |
|---|---|---|
| FULL_RGB | 三槽均重复完整画面 | 相同版式/视觉预算的完整 RGB 基线 |
| UNIFORM_RGB | 固定左/中/右区域，支持门控与 gaze 组相同 | 一般局部放大效果；它仍共享 gaze 有效性，不是 gaze-free |
| GAZE_RGB | 脚本根据真实三段 gaze 质量分配区域 | 正确空间注意分配的效果 |
| MIRRORED_GAZE_RGB | 只镜像内部密度后重新选区域；RGB 不镜像 | 空间错误注意分配控制，不保证每条交互都构成足够强的误导 |

条件共用源帧、时间顺序、完整指令、完整候选、模型、输出协议、decoder、画幅和像素预算。提示模板相同，但细节来源矩形随实际输入变化，避免隐去裁剪与全图的坐标关系。因此这轮测量 RGB 证据及其来源元数据的组织方式，不能声称是只改变像素的一因素实验。没有新增直接热图组；不能用旧热图成绩充当同批配对比较。

## 样本与运行

在读取新视频、gaze 支持、模型响应或本轮 GT 前，固定种子 20261109，每子场景随机取一条。保守扫描已有本地 manifest/selection/fixed-samples，排除 198 个已出现的地址。原始指令来源的行号映射与归档 100 条指令逐条核对一致；scene3 原 merged 表中 medium 首条缺失会造成错位，本轮直接读取原 XLSX 的正确指令列。推理准备不读取 referent 列。

本轮称“相对当前可用本地记录的新交互”。不能证明它们未在不可用原服务器运行过，也不声称未进入模型预训练。每子场景只有一条，是工程 pilot，不足以建立可靠泛化收益。

| 子场景 | row_index（零起始） | event_id | 有支持帧 |
|---|---:|---|---:|
| scene1 | 196 | scene1_197 | 0/12 |
| scene2 | 52 | scene2_53 | 11/12 |
| scene3 | 522 | scene3_523 | 11/12 |
| scene4_room1 | 82 | scene4_83 | 11/12 |
| scene4_room2 | 173 | scene4_174 | 0/12 |
| scene4_room3 | 39 | scene4_40 | 11/12 |
| scene4_room4 | 120 | scene4_121 | 11/12 |
| scene5 | 402 | scene5_403 | 0/12 |

全部保留在主评价分母，不根据支持/预测/GT换样本。语言 mention 提取使用原 `ReferentialAgent`，包含原文 span、类型解析与显式执行者覆盖检查，共享于四条件，并在视觉推理前冻结。完全相同的提示与图像哈希只调用一次模型，多个条件保留明确 alias 记录。

运行目录：`.local-sync/gaze_rgb_routing_20261109_mention_first/`。原 `.local-sync/gaze_rgb_routing_20261109/` 保留 CPU 输入与中断版本：简化语言解析漏掉两条指令中的明确 actor，第一条视觉请求尚未保存响应即被终止；没有读取本轮 GT。修正后保持同一批样本、RGB、routing 参数、预算和 decoder，独立目录冻结，并只读复用 CPU 产物。命令：

```bash
source /ai/data/TriModal-Referring/.local-sync/reconstructed_pilot100/env.sh
cd /ai/data/MultiAgent
PYTHONPATH=. python tools/run_gaze_rgb_routing.py select --output 新目录
PYTHONPATH=. python tools/run_gaze_rgb_routing.py prepare --output 新目录
PYTHONPATH=. python tools/run_gaze_rgb_routing.py verify --output 新目录
PYTHONPATH=. python tools/run_gaze_rgb_routing.py infer --output 新目录
PYTHONPATH=. python tools/run_gaze_rgb_routing.py evaluate --output 新目录
PYTHONPATH=. python tools/run_gaze_rgb_routing.py report --output 新目录
```

`select` 拒绝覆盖非空目录；`prepare` 保留 CPU 中间缓存供中断恢复，拒绝覆盖冻结 manifest。`infer` 检查 manifest 和源码锁，已保存条件跳过；模型失败保留原记录并计空集合，不根据 GT 修复。`evaluate` 要求每个条件已有结果，先冻结响应哈希，随后才读 GT。每次响应后仅清理可再生的大张量和处理器 RGB trace，保留原输入、响应、哈希及 `trace.json`。

## 当前验证与限制

38 项针对测试通过（9 项新增 routing + 5 项语言边界 + 3 项新运行冻结/评分边界 + 10 项 V4 合约 + 3 项 decoder + 8 项 attention volume）；核对完整 RGB 不变、支持不足回退、时段保留、边缘窗口、空间错置、无 gaze 数值模型 payload 及原坐标兼容。真实模型完成 32 条件结果、23 次不同视觉输入；32/32 schema 合法、27/32 mention 覆盖完整，32/32 vision 消费核验通过，全部 11,088 视觉 token。5 个覆盖失败包括 scene1 的同一空响应跨四组精确复用，以及 scene4_room4 的 FULL_RGB 空响应；均按冻结策略计空集合。

主要限制仍是 gaze-to-object 对应/历史重投影未绝对标定、局部预算有限、原 RGB 采集标记保留、mention parser 可能遗漏、8B 跨帧实例判断以及近似 anchor-center decoder。此次实现改变信息接口，不预设其收益，也未自动替换默认系统。

## 最终集合结果

全部响应先写入 `predictions_frozen.json`，随后才加载本轮 GT，用原 evaluator 评分。独立集合统计与原 evaluator 的 Micro-F1 一致。

| 条件 | TP/FP/FN | Micro-F1 | Exact-set | Any-hit |
|---|---|---:|---:|---:|
| FULL_RGB | 7/1/13 | 50.00% | 0/8 | 4/8 |
| UNIFORM_RGB | 6/1/14 | 44.44% | 1/8 | 4/8 |
| GAZE_RGB | 8/1/12 | 55.17% | 0/8 | 5/8 |
| MIRRORED_GAZE_RGB | 8/0/12 | 57.14% | 0/8 | 5/8 |

GAZE_RGB 相对 FULL_RGB 的 Micro-F1 差为 **+5.17 个百分点**，按交互配对 bootstrap 5000 次，经验 95% 区间 **[-4.60, 20.81]**；相对固定区域为 +10.73 点、区间 [-14.04, 35.86]。真实引导低于镜像引导 1.97 点、区间 [-6.06, 0.00]。这些是 Micro-F1 的差；`analysis.json` 的 `paired_event_f1_contrasts` 是逐交互 F1 均值差，不混用。

有支持的 5 交互中，四组 Micro-F1 为 35.29% / 25.00% / 44.44% / 47.06%；3 个完全回退的交互跨条件相同。不能只报告有支持子集而省略总体。

**这轮没有证明正确 gaze 引导的独立收益。** 新增正确 referent 只有 room4/120 的 dining_chair，真实和镜像引导都补回；真实引导在 scene2/52 额外输出错误 desk2，镜像没有。真实/固定/镜像三组都去掉了 FULL_RGB 在 scene3/522 的错误 building1；固定裁剪独自恢复 room3/39 的 wall_picture1。现象更符合“补充 RGB 视图会改变选择，但正确 gaze 的作用尚未隔离”，不能称其优于直接热图，也不能宣称 gaze 无效。Exact-set 未随真实引导改善。

下游仍有明显损失：四组累计 69 条有效 grounded 行中，30 条因与近似 anchor 中心距离较远被拒、4 条无相机/候选支持、1 条同类歧义；这些不是已确认的视觉识别错误。32 条件中 5 条 mention 覆盖失败，也不能由区域组织自动解决。没有根据这些 GT 结果修改窗口、阈值、提示、decoder 或候选。

## 真实输入与干预审计

`verification.json` 核验 384 张复合输入：完整 RGB 区域逐像素相同，所有补充面板均可从对应源 RGB 和冻结矩形重建，不含脚本新增热图像素。1152 个三槽记录中 447 个启用细节，按真实/固定/镜像各 149 个；FULL_RGB 始终重复完整画面。

`routing_intervention_audit.json` 中，149/149 个真实窗口不同于固定区域，148/149 不同于镜像控制。启用窗口平均覆盖各自阶段真实密度质量约 97.54%，固定区域 33.61%，镜像区域仍有 57.63%。这仅验证内部密度被不同方式路由，不是目标命中率；镜像保留较多真实密度，所以它也不是完全去除 gaze 信息的控制。

查看器：`.local-sync/gaze_rgb_routing_20261109_mention_first/index.html`；四条件示意 `comparison.jpg`；人读汇总 `RESULTS.md`；机器汇总 `summary.json`；原始集合评价 `evaluation/<condition>/original_summary.json`。查看器 384 个图片引用存在，静态 HTML 检查通过；没有宣称执行完整浏览器自动化。模型 trace 的可再生大张量和 processor RGB 已按响应清理，`trace.json` 和全部原输入/响应保留。

```bash
PYTHONPATH=. python tools/audit_gaze_rgb_routing.py
PYTHONPATH=. python tools/summarize_gaze_rgb_routing.py
```

当前实现已完成且可使用；保留为独立实验入口，默认完整 agent 未替换。进一步判断效果需要更大的独立交互对照和视觉点/scene-ID 分层诊断，不在本批已评分交互上继续调参。
