# 数据与方法核查

## 范围与复现

本结果按固定实施说明执行五组同帧呈现对照 R/H_inst/H_local/F_inst/F_local。样本 manifest seed=20261021；选择依据：pilot20 preserved; proportional scene quotas with seeded random supplements; no prediction-based filtering。该 pilot 位于现有 evaluation pool，没有官方开发划分。全部为离线完整交互诊断；每组每样本输入相同的 3 个事件覆盖帧。R 是 8B 同流程下的 raw-RGB 参照条件，不是历史 30B 基线复现。未实现/未声称全景、空间图集或 B2 深度重投影。

入口仍为 `python -m MultiAgent.run_v3`，示例：

```bash
cd /media/kunyu/16TB/spj/Multiagent
source /home/kunyu/miniconda3/etc/profile.d/conda.sh
conda activate trimodal
source /media/kunyu/16TB/spj/setup/use_prepared_inputs.sh
python -m MultiAgent.run_v3 --manifest runs/visual_evidence_8b_20261021/pilot100_relaxed_clock/manifest.json --output-dir runs/visual_evidence_8b_20261021/pilot100_relaxed_clock_rerun --cache-dir runs/visual_evidence_8b_20261021/cache --experiment V3 --visual-study temporal --study-config /media/kunyu/16TB/spj/Multiagent/configs/temporal_gaze_8b_pilot100_clock075_20261021.json --qwen-model Qwen3-VL-8B-Instruct --base-url http://100.124.154.87:8000/v1 --api-key-env VRTRIREF_API_KEY
```

CPU 准备命令：

```bash
python -m MultiAgent.run_v3 --manifest runs/visual_evidence_8b_20261021/pilot100_relaxed_clock/manifest.json --output-dir runs/visual_evidence_8b_20261021/pilot100_relaxed_clock_cpu --cache-dir runs/visual_evidence_8b_20261021/cache --experiment V3 --visual-study temporal --cpu-only --study-config /media/kunyu/16TB/spj/Multiagent/configs/temporal_gaze_8b_pilot100_clock075_20261021.json
```

在线调用需设置 `VRTRIREF_API_KEY=EMPTY`；日志不包含密钥。此目录保留固定样本 `manifest.json`、配置、逐样本预测、输入图、原始请求/响应缓存和本报告。

仅从已经保存的预测重建配对与指标报告（不发模型请求）：

```bash
python -m MultiAgent.run_v3 --manifest runs/visual_evidence_8b_20261021/pilot100_relaxed_clock/manifest.json --output-dir runs/visual_evidence_8b_20261021/pilot100_relaxed_clock --cache-dir runs/visual_evidence_8b_20261021/cache --finalize-study
```

## 数据、时钟与几何

- 本轮时钟比例阈值 0.75（项目默认仍为 0.30）。100 条计划样本；时钟/退化窗口保护拒绝 gaze/video 对齐 10 例，event window 经视频范围裁剪 99 例。未标定起点误差不会因放宽阈值而消失。拒绝样本继续使用原始 RGB 并进入评价分母。
- 至少一帧存在可投影热图证据的样本：instant 90/100，local_history 90/100；对齐有效但没有投影证据的帧同样回退原图。
- gaze telemetry 中有效字段比例均值：1.000。按真实 ISO 时间戳计算的采样率中位数：8.99 Hz。
- 世界 gazePoint 是有限世界坐标求交点；gazeVector 是世界方向向量，作为无限远方向投影。所有样本均未记录碰撞 anchor ID。
- 旋转一致性检查：world gazeVector 经过记录 xyzw 四元数逆旋转，与 cameraGazeDirection 比较，样本 p95 的最大值 1.26372e-05°。这验证内部旋转约定一致，不是相机外参标定。
- 记录相机位置在每个样本内的最大位移范围 0.071–0.268 Unity world units；物理单位、录制相机与中心眼偏移未知。无同步深度、可重放 mesh 或动态物体变换。遮挡保持 `visibility_unknown`；历史点不跟踪移动对象。
- 可核实代码约定：四元数 (x,y,z,w)，逆旋转 world→camera，相机 +z 前向、+y 上方、左上像素原点；标量 `cameraFOV` 按垂直 FOV、方形像素投影。Unity 左/右手性、逐帧内参和畸变未独立标定。
- 视频实际由现有 ffmpeg 抽帧器按秒 seek，保留抽取时间；短边约束为最大 672 像素，不裁切/letterbox。视觉服务 processor 会否再次缩放以及最终视觉网格无法从兼容 API 验证。
- 热图核以角宽度配置后换算输入分辨率像素；无有效 gaze 时 heatmap 为零、H/F 保留 RGB。无遮挡判断，不归一化弱证据；foveation 使用同一 heatmap 驱动的 2x/4x/8x 降采样/上采样金字塔并保留清晰 RGB 下限，不是 Scene-aware Foveated Rendering 论文复现，也未声称降低视觉 token 或推理耗时。

## 冻结路径与推理

- 同一 8B 服务先完成 Referential、候选映射、直接事件时间计划及原 RGB Hand 观察；五组共用冻结哈希、候选映射、Hand 图像/观察和每 mention 的 Visual 调用粒度。Coordinator 使用原 prompt、规则与原始 RGB 图像，仅 Visual evidence 变化。GT 只用于独立 evaluator；输入构造仅写入 GT 文件哈希，不把标签值送入模型。
- 每组计划 100 条，成功完成上游与模型链路 98 条；另 2 条在 Visual/Coordinator 前失败，仍计入原评估分母。每组完成 98 次 Visual 和 Coordinator 逻辑调用；相同 Coordinator 输入由确定性缓存复用。调用审计记录实际 HTTP、缓存命中、stage、耗时、图片哈希/尺寸、原始响应和 usage。远端 processor resize 未知，因此视觉 token 是 32px merged-grid 估计，服务端 prompt_tokens 同时包含文本/图像，不当作视觉 token。
- 服务端模型 ID：`Qwen3-VL-8B-Instruct`；vLLM OpenAI-compatible `/v1` 服务。实际模型 HTTP 961 次，缓存复用 222 次，模型累计 HTTP 等待 1630.2s；累计服务端 prompt_tokens 2840317（非纯视觉）。
- 全部阶段复用已部署 Qwen3-VL-8B-Instruct API；本轮未启动或停止推理服务。

## 结果读取与解释

- 本轮 N=100：H_local Any-hit 0.670 vs R 0.670，Micro-F1 0.511 vs 0.500；F_local Micro-F1 0.487。各组 Exact-set 以 RESULTS.md 表格为准；本轮探索观察不视作显著性结论。
- 各 gaze 组相对 R 实际改变 Visual 图像像素的样本数：H_inst=88/100, H_local=88/100, F_inst=88/100, F_local=88/100；其余输入为全 RGB 回退或上游失败。
- `paired_comparison.json` 对 98/100 条完成样本核对同源帧哈希；剩余 2 条为上游失败，没有模型输入图。
- `RESULTS.md` 给出原 evaluator 汇总；`paired_comparison.json` 给出相对 R 的逐样本变化；`failure_analysis.json` 对照 Visual 融合前与 Coordinator 后集合，包含漏选、额外对象、同类实例候选混淆。无 mention 级 GT，不报告 mention accuracy。
- Referential 校验失败 2 条；API 失败调用 0，已返回但语义抽取为空的响应记录在 failures.json。时钟拒绝样本属于明确 RGB 回退，不能解释成热图成功。
- 已验证：API `/models` 返回配置模型；关键几何合成测试通过；已完成 98/100 条真实模型链路，逐组原 evaluator 分母均保留 100 条。
- 仅实现/待验证：gaze 热图/历史窗口和 blur blend 是否改善对象选择；基于文件时间的 gaze/video 起点估计；动态对象历史 gaze 的身份对应。当前缺少深度/mesh，故未做 B2。
