# Readraft 生产部署与升级

本文以 Debian/Ubuntu 系 Linux、systemd、Nginx 和标准目录
`/opt/readraft` 为例。其他发行版可以使用等价组件，但必须保持单个
Uvicorn worker 和单实例 SQLite 写入。

## 1. 准备主机

安装 Git、Python 3.12+、`venv`、Nginx 和 `curl`，然后创建不可登录的
专用账号：

```bash
sudo useradd --system --home /opt/readraft --shell /usr/sbin/nologin readraft
sudo install -d -o readraft -g readraft -m 700 /opt/readraft
sudo install -d -o readraft -g readraft -m 700 /var/backups/readraft
```

公网入口必须使用 HTTPS。不要直接开放 Uvicorn 的 8010 端口。

## 2. 克隆和安装

```bash
sudo -u readraft git clone https://github.com/ReBeating/Readraft.git /opt/readraft
LATEST_TAG="$(
  sudo -u readraft git -C /opt/readraft tag --list 'v[0-9]*' \
    --sort=-version:refname | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | head -n 1
)"
test -n "$LATEST_TAG"
sudo -u readraft git -C /opt/readraft checkout --detach "$LATEST_TAG"
sudo install -d -o readraft -g readraft -m 700 /opt/readraft/data
sudo -u readraft python3.12 -m venv /opt/readraft/.venv
sudo -u readraft /opt/readraft/.venv/bin/python -m pip install \
  --no-cache-dir --require-hashes -r /opt/readraft/requirements.lock
sudo install -o root -g readraft -m 640 \
  /opt/readraft/.env.example /opt/readraft/.env
sudoedit /opt/readraft/.env
```

生产环境至少设置：

```dotenv
APP_NAME=Readraft
APP_ENV=production
APP_SECRET_KEY=<至少 32 字符的随机值>
APP_CREDENTIAL_ENCRYPTION_KEY=<至少 32 字符且长期保持不变>
APP_DATA_DIR=/opt/readraft/data
APP_DATABASE_PATH=/opt/readraft/data/readraft.db
APP_COOKIE_SECURE=true
# 首次启动时临时设为 true，创建首个账号后立即改回 false。
APP_ALLOW_REGISTRATION=true
DEEPSEEK_API_KEY=
```

可以用以下命令生成两个独立随机值：

```bash
python3.12 -c 'import secrets; print(secrets.token_urlsafe(48))'
```

`APP_CREDENTIAL_ENCRYPTION_KEY` 一旦投入使用就必须与数据库一起长期
保管；丢失或修改后，数据库中的个人 API Key 将无法解密。

## 3. 安装 systemd 和 Nginx

```bash
sudo install -m 644 /opt/readraft/deploy/readraft.service \
  /etc/systemd/system/readraft.service
sudo systemctl daemon-reload
sudo systemctl enable --now readraft
curl --fail --silent --show-error http://127.0.0.1:8010/healthz
```

复制 `deploy/nginx.conf`，将示例域名和证书路径替换为真实值，再检查并
重载 Nginx：

```bash
sudo install -m 644 /opt/readraft/deploy/nginx.conf \
  /etc/nginx/sites-available/readraft
sudoedit /etc/nginx/sites-available/readraft
sudo ln -s /etc/nginx/sites-available/readraft \
  /etc/nginx/sites-enabled/readraft
sudo nginx -t
sudo systemctl reload nginx
```

确认 HTTPS 正常后再创建首个账号；随后把
`APP_ALLOW_REGISTRATION=false` 写回 `.env` 并重启服务。

## 4. 日常升级

生产实例默认只更新到带注释的稳定 `vMAJOR.MINOR.PATCH` Tag，不会自动
运行 `main` 的开发提交。管理员可以先只读检查，再手动执行一次更新：

```bash
sudo /opt/readraft/deploy/update.sh --check
sudo /opt/readraft/deploy/update.sh
```

脚本会：

1. 使用系统级互斥锁拒绝并发升级，并拒绝未提交改动、非快进历史、
   非稳定格式或未带注释的版本 Tag；
2. 在当前服务仍正常运行时，为目标提交创建隔离的 Python 环境并安装
   带哈希的锁定依赖；
3. 停止服务，在 `/var/backups/readraft` 创建完整备份并实际解包校验；
4. 以 detached HEAD 切换到目标版本，并原子切换 Python 环境；
5. 更新 systemd 单元、启动服务并等待 `/healthz` 返回成功；
6. 若新版本启动失败，自动切回旧提交和旧依赖、恢复更新前备份，再验证
   旧服务已经恢复健康。

数据库迁移由新版本启动时自动执行。不要直接使用会覆盖本地历史的
`git reset --hard`，也不要跳过备份强行降级数据库。

更新失败时 systemd 任务仍会记录为失败，以便监控发现；自动回滚成功不
会把这次失败伪装成成功。脚本会保留备份并输出诊断命令。先查看：

```bash
sudo systemctl status readraft
sudo journalctl -u readraft -n 200 --no-pager
```

只有自动回滚也失败时才需要手工恢复。完整恢复要求停止服务：

```bash
sudo systemctl stop readraft
cd /opt/readraft
sudo -u readraft .venv/bin/python -m app.backup verify \
  /var/backups/readraft/<备份文件>.zip
sudo -u readraft .venv/bin/python -m app.backup restore \
  /var/backups/readraft/<备份文件>.zip --replace
sudo systemctl start readraft
```

恢复数据库前还应把代码切回与备份兼容的正式版本。若不确定，请保留故障
现场和备份，不要反复启动不同版本。

## 5. 启用自动更新

先成功执行至少一次手动检查或更新，再显式安装并启用 Timer：

```bash
sudo /opt/readraft/deploy/install-auto-update.sh
systemctl list-timers readraft-update.timer --all
```

Timer 每日检查一次，并随机延迟最多一小时，避免所有实例同时访问 GitHub。
错过计划时间时，`Persistent=true` 会在服务器恢复运行后补做一次检查。
没有新版本时不会停止 Readraft，也不会创建空备份。

默认配置位于 `/etc/readraft/update.env`：

```dotenv
READRAFT_UPDATE_CHANNEL=release
READRAFT_BACKUP_RETENTION_DAYS=30
READRAFT_VENV_RETENTION=3
```

`release` 只接受最新稳定版本 Tag。只有可以容忍每个开发提交的测试机才应
改成 `main`。修改配置后无需重启应用，下一次更新任务会读取新值。可以
手动触发和查看日志：

```bash
sudo systemctl start readraft-update.service
sudo journalctl -u readraft-update.service -n 200 --no-pager
```

暂停自动更新不会影响手动更新：

```bash
sudo systemctl disable --now readraft-update.timer
```

脚本只自动清理自身创建且超过保留天数的 `readraft-update-*.zip`，不会
删除手工备份。每次更新保留当前与近期隔离依赖环境，以便失败回滚。

## 6. 从非 Git 部署迁移

通过复制或 rsync 安装、且目录内没有 `.git` 的旧实例不能直接
`git pull`。一次性迁移时：

1. 停止旧服务并创建、校验完整备份；
2. 在新目录按本文重新 `git clone`；
3. 安全复制原 `.env` 和数据目录，保持属主及权限；
4. 检查其中的 `APP_NAME`、路径和 systemd 单元；
5. 启动新实例并验证 `/healthz` 后，再移除旧目录。

不要把 `.env`、数据目录或备份提交到 Git。

早于 Readraft 0.1.0 的内部开发版本使用过未公开的品牌和归档协议。首次
迁移到 Readraft 时可以继续通过 `APP_DATABASE_PATH` 指向原数据库，但
需要重新保存个人模型 API Key；旧品牌扩展名的作品归档和完整备份不会被
新协议导入。迁移前请保留原代码与已校验备份，确认新实例可用后再清理。
