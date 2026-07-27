# 为 Readraft 贡献

感谢你愿意改进 Readraft。小型修复可以直接提交 Pull Request；涉及数据
模型、归档格式、权限边界或主要交互的改动，请先开 Issue 说明目标、用户
流程和兼容性影响。

## 开发环境

需要 Python 3.12 或更高版本：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements-dev.lock
cp .env.example .env
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

自动化测试不调用真实模型，也不需要 API Key：

```bash
python -m ruff check app tests
python -m pytest
python -m pip_audit --disable-pip -r requirements.lock
git ls-files -z | xargs -0 detect-secrets-hook --baseline .secrets.baseline
bash -n deploy/update.sh
```

## 改动要求

- 不要提交 `.env`、真实 API Key、数据库、作品正文、备份或生产日志。
- 测试材料必须是自行编写的合成文本，不能复制仍受保护的小说内容。
- AI 写入必须保持服务端权限校验、作者确认和不可变版本边界，不能依赖
  模型自述来保证安全。
- 已发布的数据库迁移只可向前追加；不要改写已执行迁移的含义。
- 改动作品归档或完整备份协议时，必须同步提升格式版本并补充导入、篡改
  校验和配额测试。
- 新增 Provider 时必须声明能力矩阵，并测试鉴权、错误、超时、模型目录
  和自定义 Base URL 边界。
- 删除界面时也应删除对应路由、模板、链接和测试，避免保留无法维护的
  兼容页面。

依赖范围维护在 `requirements.txt` 和 `requirements-dev.txt`。更新后
必须使用 Python 3.12 与 `pip-tools` 重新生成带哈希的锁文件，不能手工
修改锁文件。

## Pull Request

一个 Pull Request 尽量只解决一个问题，并说明：

1. 改了什么以及为什么；
2. 对用户、数据格式和部署的影响；
3. 执行过哪些验证；
4. 界面改动的桌面与手机截图。

根据 Apache License 2.0 第 5 节，除非你明确另行声明，主动提交并被
项目接纳的贡献将按 Apache License 2.0 授权。
