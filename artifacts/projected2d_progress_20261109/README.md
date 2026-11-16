# 2026-11-09 Projected-2D 实验进度档案

入口：`RESUME_STATE.json`（恢复上下文）、`RESULTS_TABLE.csv`（最终分数）、`SAMPLE_INVENTORY.json`（32条/95帧与分组）。正式实验文档与最终机器报告都在 `snapshot/MultiAgent/` 下按原相对路径保存。

已复制当前代码、文档、各run的清单/配置/冻结响应/trace JSON和评分；源码为归档时快照，历史准备脚本漂移见 `SOURCE_LOCK_DRIFT.json`。318条冻结响应哈希逐一验证，归档没有新增模型调用。

原RGB、gaze NPZ、视频和模型权重仍在原目录；`REFERENCED_VISUAL_INPUTS.json` 保存图片/热图哈希。将本目录移到另一机器不等于已经迁移这些大文件。需要原环境和数据才能重新推理。

校验复制文件：

```bash
python tools/archive_projected2d_progress.py verify --output ARCHIVE_DIRECTORY
```

同时校验原位置图片/热图可加 `--external`。正式文件清单与SHA-256在 `ARCHIVE_MANIFEST.json`；其自身哈希在 `ARCHIVE_MANIFEST.sha256`。
