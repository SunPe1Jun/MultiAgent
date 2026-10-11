# 最终验证

32 项回归测试通过，8 条真实 scene4 smoke 完成。19 个 mention 均有候选，chair 候选包含 chair2，table 的组件沿用 dining_table，bedroom side 能恢复 bedroom。所有场景均触发 30% guard，禁用音频时间和未校准 telemetry 空间提示；每条至少有 5 张不同 raw RGB。room4/11 已由片尾重复帧恢复为全片覆盖。

采用同一份修正后 GT：

|预测版本|Any-hit|Exact-set|Micro-F1|
|---|---:|---:|---:|
|旧 v12 预测重评|100%|62.5%|83.33%|
|最终 v14|100%|62.5%|83.33%|

指标持平，不能宣称算法精度提升。room4/11 由 curtains2/window 变成 curtains1/window2；窗帘选对而 window 实例仍错。room1/47 仍选 curtain2，room2/150 仍选 chair。正确候选已不再被提前剔除，但拒绝错位 telemetry 后，仅凭 RGB 区分相似实例仍受限。

30% 是异常保护阈值，不是精确同步校准；本批所有样本回退为指令+RGB+可见手势。后续若要恢复有效 gaze/投影，需要独立验证 video 与 telemetry 的相对原点。

新标注仅修正用户确认的两条，其余扫描疑点见 ../gt_audit_20261010/GT_AUDIT_REPORT_zh.md。历史输出仍保留原 GT 快照；对比使用独立重评分文件，不改写历史结果。
