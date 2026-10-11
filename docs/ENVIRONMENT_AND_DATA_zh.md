# 环境、数据和复现说明

## 目录约定

默认工作区：`/workspace/usr3`。

- 代码：`/workspace/usr3/MultiAgent`
- 原始 TriModal-Referring：`/workspace/usr3/TriModal-Referring`
- 视频/遥测：`/workspace/usr3/V3dMD`
- Qwen3-VL：`/workspace/usr3/Qwen3-VL-30B-A3B-Instruct`
- GroundingDINO：`/workspace/usr3/grounding-dino-base`
- SAM：`/workspace/usr3/sam-base`
- ASR：`/workspace/usr3/ASR/faster-whisper-small.en`
- Conda：`/workspace/usr3/miniconda3`

数据和权重未上传 GitHub。需要把这些目录恢复到相同路径，或修改入口参数和 `MultiAgent/baseline.py` 中的路径。

## 环境

```bash
source /workspace/usr3/miniconda3/etc/profile.d/conda.sh
conda activate trimodal
```

仓库提供：

- `environment/trimodal-history.yml`：Conda 历史依赖
- `environment/trimodal-pip-freeze.txt`：当前 pip 包快照

如果没有原环境，可先创建 Python 3.10 环境，再按 pip 快照安装；CUDA、PyTorch 和本地模型路径需按目标机器调整。

## 运行顺序

```bash
python -m unittest discover -s MultiAgent -t . -p 'test*.py'
python -m MultiAgent.run_v3 --help
python -m MultiAgent.run_v3 \
  --manifest /workspace/usr3/MultiAgent/runs/pilot20/manifest.json \
  --output-dir /workspace/usr3/MultiAgent/runs/reproduce_v3 \
  --cache-dir /workspace/usr3/MultiAgent/runs/reproduce_v3/cache/qwen \
  --limit 1
```

完整运行前检查：模型目录、视频路径、音频路径、anchor table、GT Excel 和 `ffmpeg/ffprobe`。运行输出包含 prompt/cache、trace、逐行预测、summary 和评估结果。
