# 从注视区域到具体对象：自动调研与工程迭代（2026-11-06）

这轮的核心方案是增加可追溯的**可见物体层**：原始指令类别 → 全帧检测/分割 → 实例视觉编号 → 注视与mask的软关联 → 原生帧中的对象历史 → 语言筛选。热图仍以图像提供，脚本内部的注视数值不作为候选文本权重输入大模型。没有把最热物体当作真实 referent。

当前环境没有已安装的 autoResearch 技能，本轮直接执行了原始资料调研、协议冻结、本地实验、错误诊断和后续实现。不是调用某个未安装插件，也没有代替用户创建新的长期目标。

## 1. 调研后的路线选择

| 方法 | 解决的环节 | 对当前任务的限制 | 本轮处理 |
| --- | --- | --- | --- |
| gaze点提示 SAM | 注视位置到一个或多个可见区域 | 点可能落在手、地面、边缘或多个尺度；分割质量不等于目标身份 | 作为候选补充路线；未把点提示mask直接定为目标 |
| Grounding DINO＋SAM2 | 从语言类别找到具体可见实例和边界 | 类别词歧义、小物体、漏检、相互遮挡和嵌套mask | 已实现全帧候选，检测不依赖gaze，保留所有通过阈值的候选 |
| Set-of-Mark | 给模型一个可引用的视觉地址，减少坐标/物体名称漂移 | 编号不是语义身份，也不保证模型选对 | 已实现图内随机O编号、边界、放大的原始物体细节 |
| SAM2视频传播＋对象历史 | 在头动下保留同一物体提案，跨原生帧累计注视 | 漂移、遮挡、新进入物体、跨帧身份仍需核验 | 已实现4交互双向离线传播和独立重检测兼容性检查 |
| 采集端记录射线命中实例 | 直接记录“看见/射线命中哪个场景实例” | 需要修改采集；命中对象仍不等于语言目标；runtime ID不能直接跨会话使用 | 审计现有字段，提出稳定对象ID、父对象和同步实例mask的采集建议，未伪造已有数据 |

Grounding DINO、SAM2和SoM均参考原作者论文/仓库，没有把第三方排行榜当作当前数据的效果依据。本轮使用可公开获取的 Grounding DINO tiny 与 SAM2.1 hiera tiny，模型仓库revision及文件SHA保存于 `.local-sync/models/*/download_provenance.json`。选择它们是为了在当前24GB GPU上完成可审阅实验，不声称它们是所有模型中最优。

## 2. 最相关的文献，及不能直接照搬的部分

Materia等人的 *Leveraging Gaze and Set-of-Mark in VLLMs for Human-Object Interaction Anticipation from Egocentric Videos* 于2026-04-04提交，是很贴近本项目的工作。它联合分割遮罩、最近15个gaze点的轨迹和时间采样，做HD-EPIC下一交互对象的五选一类别判断。**该实现省略了字母/数字标签**，不是具体实例ID指代定位；视频还根据交互/gaze时刻预裁切，不能直接套用“越近结尾越相关”的先验到本项目多mention指令。

论文表2在相同采样参数下，VLLM-only为24.9%，SoM＋gaze为27.2%；这是组件组合的结果，不能把跨不同采样配置的总体提升全归因于热图。它支持尝试“对象边界＋注意线索”，并不证明当前球面热图或本项目最终指代必然有效。

本轮采用SoM视觉地址思想，但另加图内实例编号；采用SAM2的mask传播，但没有将其等同于已验证对象身份。更符合当前头动动机的工程推论是：**先在每次观测的原生帧内把gaze关联到对象提案，再对该提案累计时间证据**，可以减少让语言模型自己从跨帧漂移的热点重建对象对应的工作。该推论仍需本地验证。

## 3. 对象提案与图形注意关联

`MultiAgent/object_attention.py`与`tools/run_object_attention_study.py`实现：

- 仅从原始指令词语生成本pilot的检测类别；`file`扩展为document/folder等。词表是有限的工程适配，不是通用语义解析，更不读取scene anchor类别或对象GT。
- 原始支持帧按首帧、中间索引、末帧选取，不根据gaze或GT取图。Grounding DINO阈值0.25、文本阈值0.2；同类框IoU0.6去重。SAM2按每个检测框生成mask，保留全部通过检测规则的实例。
- O编号和页面顺序确定性打乱，与置信度、注视热量和GT顺序无关。模型提示不含检测类别/数值分数，只给原始指令、图像及页面可选ID。
- 最近2秒的有效gaze复用已支持的时钟、原生帧相机姿态和三时间层。对于重叠mask，按每个像素的mask数量平分质量；未落入mask的质量单独保留，不分给最近实例。
- `OBJECT_HEAT`把分配到物体的支持时长以整mask橙色着色，未承接的gaze以蓝色显示。填色只是对象关联图，不表示物体每个像素都被看过，更不是目标置信度。
- 图片同时保留全帧上下文、白色对象边界、随机编号与放大的原始细节。细节保留长宽比；几何准备时发现小crop没有放大，已在任何模型调用前修正。无模型的首版留在`study_geometry_v0`。

模型返回`selected`与`uncertain`两组实例编号；所有原始候选始终保留，未把单次模型拒绝接成自动硬删除，也未接默认Coordinator/evaluator。

## 4. 原始资料与可复核入口

文献与官方实现：

1. [Set-of-Mark Visual Prompting for GPT-4V](https://arxiv.org/abs/2310.11441)，[Microsoft官方实现](https://github.com/microsoft/SoM)。
2. [Grounding DINO](https://arxiv.org/abs/2303.05499)，[官方实现](https://github.com/IDEA-Research/GroundingDINO)。
3. [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714)，[Meta官方实现](https://github.com/facebookresearch/sam2)。
4. [Grounded SAM 2官方组合实现](https://github.com/IDEA-Research/Grounded-SAM-2)。
5. [Gaze＋SoM的交互预测论文](https://arxiv.org/abs/2604.03667)，[作者实现](https://github.com/fpv-iplab/leveraging_gaze_som_vllms_human_obj_anticipation)。
6. [Unity射线命中的Collider](https://docs.unity3d.com/ScriptReference/RaycastHit-collider.html)，[GetInstanceID](https://docs.unity3d.com/ScriptReference/Object.GetInstanceID.html)。命中对象可通过Collider关联到GameObject；采集端应另存稳定应用ID/父对象映射，不依赖跨会话变化的runtime整数。 当前GetInstanceID参考页已标记移除并指向GetEntityId，具体访问接口需按采集工程的Unity版本确认；本轮没有假定或修改其Unity版本。

检索及原始资料快照在 `.local-sync/object_attention_research_20261106/sources/`。实际阅读时核对了页面标题；两次不相关arXiv编号响应单独标记`excluded_unrelated_*`，未作为研究依据。模型权重传输走镜像，模型来源/版本仍按原作者仓库记录。

本地入口和本轮结果见 [对象注意实验记录](OBJECT_ATTENTION_EXPERIMENT_20261106_zh.md)。
