# 容器镜像和工作区备份

## 已有镜像归档

本地归档位于：

```text
/workspace/usr3/container-image-backups/131-vip-gpu45-sunpeijun-pre-reclaim-20260901-194731.tar.gz
```

- 大小：8,034,467,368 bytes（约 7.5 GiB）
- SHA-256：`4d4cbcc983cc409c73c56c15df4cc624f15c2abc5f094254b7ab2b96ba73847b`
- 校验文件：同目录 `.sha256`
- 归档未上传 GitHub，需要单独复制到安全存储。

当前服务器没有 Docker CLI，因此无法在本次环境中重新执行 `docker save` 或验证 registry tag。归档来自此前的 pre-reclaim 容器备份；恢复时先检查内容类型。

## 备份文件复制和校验

```bash
mkdir -p /backup/usr3
cp --progress /workspace/usr3/container-image-backups/131-vip-gpu45-sunpeijun-pre-reclaim-20260901-194731.tar.gz /backup/usr3/
sha256sum /backup/usr3/131-vip-gpu45-sunpeijun-pre-reclaim-20260901-194731.tar.gz
```

SHA-256 必须与上面一致。建议同时备份：`MultiAgent_repo`、`TriModal-Referring` 的 git bundle、环境快照和所有数据/模型目录。

## 在有 Docker 的机器上恢复

先确认归档是 `docker save` 格式：

```bash
tar -tzf 131-vip-gpu45-sunpeijun-pre-reclaim-20260901-194731.tar.gz | head
```

如果列表中有 `manifest.json` 和 layer 文件：

```bash
gunzip -c 131-vip-gpu45-sunpeijun-pre-reclaim-20260901-194731.tar.gz | docker load
# 或：docker load -i image.tar
 docker images
```

如果它是 rootfs/export 格式而不是 image archive：

```bash
docker import 131-vip-gpu45-sunpeijun-pre-reclaim-20260901-194731.tar.gz trimodal:restored
```

`docker import` 不会恢复原镜像的 ENTRYPOINT、CMD、环境变量和挂载点，需要按照原运行命令手工补齐。优先尝试 `docker load`，失败后再确认是否应使用 `docker import`。

## 复用代码

```bash
git clone git@github.com:SunPe1Jun/MultiAgent.git
cd MultiAgent
git lfs install  # 仅当后续决定使用 Git LFS 存放大文件
```

代码仍需要外部挂载 `/workspace/usr3/V3dMD`、模型目录和 ASR 目录；Git 仓库只保存代码、文档、清单和小型结果摘要。
