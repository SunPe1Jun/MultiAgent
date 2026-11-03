# 数据与方法核查

## 范围与复现

本结果按固定实施说明执行五组同帧呈现对照 R/H_inst/H_local/F_inst/F_local。样本 manifest seed=20260922；选择依据：scene stratified seeded sampling with multi-reference / relation coverage; no accuracy-based filtering。该 pilot 位于现有 evaluation pool，没有官方开发划分。全部为离线完整交互诊断；每组每样本输入相同的 3 个事件覆盖帧。R 是 8B 同流程下的 raw-RGB 参照条件，不是历史 30B 基线复现。未实现/未声称全景、空间图集或 B2 深度重投影。

入口仍为 `python -m MultiAgent.run_v3`，示例：

```bash
cd /media/kunyu/16TB/spj/Multiagent
python -m MultiAgent.run_v3 --manifest runs/visual_evidence_8b_20261021/pilot20_corrected/manifest.json --output-dir runs/visual_evidence_8b_20261021/pilot20_corrected_rerun --cache-dir runs/visual_evidence_8b_20261021/cache --experiment V3 --visual-study temporal --study-config configs/temporal_gaze_8b_20261021.json --qwen-model Qwen3-VL-8B-Instruct --base-url http://100.124.154.87:8000/v1 --api-key-env VRTRIREF_API_KEY
```

CPU 准备命令：

```bash
python -m MultiAgent.run_v3 --manifest runs/visual_evidence_8b_20261021/pilot20_corrected/manifest.json --output-dir runs/visual_evidence_8b_20261021/cpu_rebuild --cache-dir runs/visual_evidence_8b_20261021/cache --experiment V3 --visual-study temporal --cpu-only --study-config configs/temporal_gaze_8b_20261021.json
```

在线调用需设置 `VRTRIREF_API_KEY=EMPTY`；日志不包含密钥。此目录保留固定样本 `manifest.json`、配置、逐样本预测、输入图、原始请求/响应缓存和本报告。

仅从已经保存的预测重建配对与指标报告（不发模型请求）：

```bash
python -m MultiAgent.run_v3 --manifest runs/visual_evidence_8b_20261021/pilot20_corrected/manifest.json --output-dir runs/visual_evidence_8b_20261021/pilot20_corrected --cache-dir runs/visual_evidence_8b_20261021/cache --finalize-study
```

## 数据、时钟与几何

- 样本数：20；时钟保护拒绝 gaze/video 对齐 13 例，其余 7 例使用文件名/metadata 起点估计（仍非像素级校准）。所有拒绝样本继续使用原始 RGB 并进入评价分母。
- 至少一帧存在可投影热图证据的样本：instant 7/20，local_history 7/20；对齐有效但没有投影证据的帧同样回退原图。
- gaze telemetry 中有效字段比例均值：1.000。按真实 ISO 时间戳计算的采样率中位数：8.99 Hz。
- 世界 gazePoint 是有限世界坐标求交点；gazeVector 是世界方向向量，作为无限远方向投影。所有样本均未记录碰撞 anchor ID。
- 旋转一致性检查：world gazeVector 经过记录 xyzw 四元数逆旋转，与 cameraGazeDirection 比较，样本 p95 的最大值 1.26372e-05°。这验证内部旋转约定一致，不是相机外参标定。
- 记录相机位置在每个样本内的最大位移范围 0.080–0.268 Unity world units；物理单位、录制相机与中心眼偏移未知。无同步深度、可重放 mesh 或动态物体变换。遮挡保持 `visibility_unknown`；历史点不跟踪移动对象。
- 可核实代码约定：四元数 (x,y,z,w)，逆旋转 world→camera，相机 +z 前向、+y 上方、左上像素原点；标量 `cameraFOV` 按垂直 FOV、方形像素投影。Unity 左/右手性、逐帧内参和畸变未独立标定。
- 视频实际由现有 ffmpeg 抽帧器按秒 seek，保留抽取时间；短边约束为最大 672 像素，不裁切/letterbox。视觉服务 processor 会否再次缩放以及最终视觉网格无法从兼容 API 验证。
- 热图核以角宽度配置后换算输入分辨率像素；无有效 gaze 时 heatmap 为零、H/F 保留 RGB。无遮挡判断，不归一化弱证据；foveation 使用同一 heatmap 驱动的 2x/4x/8x 降采样/上采样金字塔并保留清晰 RGB 下限，不是 Scene-aware Foveated Rendering 论文复现，也未声称降低视觉 token 或推理耗时。

## 冻结路径与推理

- 同一 8B 服务先完成 Referential、候选映射、直接事件时间计划及原 RGB Hand 观察；五组共用冻结哈希、候选映射、Hand 图像/观察和每 mention 的 Visual 调用粒度。Coordinator 使用原 prompt、规则与原始 RGB 图像，仅 Visual evidence 变化。GT 只用于独立 evaluator；输入构造仅写入 GT 文件哈希，不把标签值送入模型。
- Coordinator 逻辑调用每组 20 次；相同输入由确定性缓存复用。调用审计记录实际 HTTP、缓存命中、stage、耗时、图片哈希/尺寸、原始响应和 usage。远端 processor resize 未知，因此视觉 token 是 32px merged-grid 估计，服务端 prompt_tokens 同时包含文本/图像，不当作视觉 token。
- 服务端模型 ID：`Qwen3-VL-8B-Instruct`；vLLM OpenAI-compatible `/v1` 服务。实际模型 HTTP 171 次，缓存复用 70 次，模型累计 HTTP 等待 189.8s；累计服务端 prompt_tokens 451220（非纯视觉）。
- GPU 推理服务已在本机可访问的远端 vLLM 进程中提供；当时未发现 PID 16413，实际服务 PID 163774。远端真实多图请求满足实验吞吐，本轮未切换本地推理。

## 结果读取与解释

- pilot20 结果：H_local Any-hit 0.800 vs R 0.750，Micro-F1 0.571 vs 0.532；F_inst Micro-F1 0.568，F_local 0.514。五组 Exact-set 均为 0.200。四个 gaze 变体各有 7/20 例实际改变 Visual 图像像素，其余 13/20 样本回退 RGB；不能视作稳定提升或显著结果。
- 各 gaze 组相对 R 实际改变 Visual 图像像素的样本数：H_inst=7/20, H_local=7/20, F_inst=7/20, F_local=7/20；其余输入明确为全 RGB 回退。
- `RESULTS.md` 给出原 evaluator 汇总；`paired_comparison.json` 给出相对 R 的逐样本配对变化及同源帧哈希检查；`failure_analysis.json` 对照 Visual 融合前与 Coordinator 后集合，包含漏选、额外对象、同类实例候选混淆。无 mention 级 GT，不报告 mention accuracy。
- 未发现运行/API/解析失败；13 个时钟拒绝样本属于明确 RGB 回退，不能解释成热图成功。当前差异是视觉呈现探索性信号，小样本无显著性结论。
- 已验证：API `/models` 返回配置模型；文本 JSON 与两张不同真实帧的多图请求成功；关键几何合成测试通过；20 个样本、五组真实模型对照与原 evaluator 完成。
- 仅实现/待验证：gaze 热图/历史窗口和 blur blend 是否改善对象选择；基于文件时间的 gaze/video 起点估计；动态对象历史 gaze 的身份对应。当前缺少深度/mesh，故未做 B2。
