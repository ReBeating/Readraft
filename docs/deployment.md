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
sudo install -d -o readraft -g readraft -m 700 /opt/readraft/data
sudo -u readraft python3.12 -m venv /opt/readraft/.venv
sudo -u readraft /opt/readraft/.venv/bin/python -m pip install \
  --require-hashes -r /opt/readraft/requirements.lock
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

标准安装以后不需要维护者登录服务器。管理员只需执行：

```bash
sudo /opt/readraft/deploy/update.sh
```

脚本会：

1. 拒绝未提交改动、缺失上游分支和非快进历史；
2. 获取远端并预先安装目标版本的锁定依赖；
3. 停止服务并在 `/var/backups/readraft` 创建完整备份；
4. 执行 `git pull --ff-only` 并再次核对实际版本依赖；
5. 更新 systemd 单元并启动服务；
6. 等待 `/healthz` 返回成功。

数据库迁移由新版本启动时自动执行。不要直接使用会覆盖本地历史的
`git reset --hard`，也不要跳过备份强行降级数据库。

如果健康检查失败，脚本会保留备份并输出诊断命令。先查看：

```bash
sudo systemctl status readraft
sudo journalctl -u readraft -n 200 --no-pager
```

完整恢复要求停止服务：

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

## 5. 从非 Git 部署迁移

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
