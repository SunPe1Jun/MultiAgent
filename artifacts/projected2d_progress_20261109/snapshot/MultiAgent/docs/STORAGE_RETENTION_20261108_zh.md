# 存储保留与清理记录（2026-11-08）

本次清理针对 `/ai/data` 存储紧张问题。清理前 `/ai/data` 仅剩约 47 GiB，`MultiAgent/.local-sync` 占约 92 GiB。

已删除的是模型视觉 trace 中可重新生成的重复输入张量和 RGB 中间图：

- `processor_inputs.pt`：2028 个，约 86.51 GiB；
- `processor_rgb_*.png`：4580 个，约 1.79 GiB；
- 合计释放约 88.29 GiB。

保留内容包括：

- 2026-11-05 和 2026-11-06 的所有实验根目录；
- manifest、protocol lock、source 快照、request/response、analysis、RESULTS、verification 和 frozen predictions；
- `trace.json`、模型调用审计、原始输入与证据图；
- 初始版本和失败版本，便于回退；
- `.local-sync/models` 中的本地模型权重。

没有修改代码、manifest、模型响应、评估结果或默认流程，也没有新增模型调用。删除清单、删除前的核心文件哈希和完整路径记录保存在：

`.local-sync/storage_cleanup_20261108.json`
