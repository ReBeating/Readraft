# Readraft

> [!WARNING]
> **Readraft 目前只是早期开发版本（0.1.x）。**
> 核心流程已经可以使用，但界面、数据格式和部署方式仍可能发生不兼容变化。
> 请勿把它作为重要作品的唯一副本，升级前务必创建并校验完整备份。

> 读懂作品，再写可能。Read. Draft. Redraft.

Readraft 是一套开源、自托管的 AI 阅读与创作工作台，面向长篇中文
小说的阅读分析、原创、续写、改写和二次创作。它把原文、不可变版本、
作品资料、分析笔记、故事记忆、编辑审校与 AI 对话组织在同一个作品库
中；只有可写的 `main` 可以改变正文，固定 Tag 始终保持只读。

正文版本只有三层：每章当前的 `main HEAD`、由正式提交留下的不可变历史，以及固定
整部作品快照的 Tag。编辑器自动暂存只负责崩溃恢复，不创建历史；手工保存、AI 写作、
撤回与恢复才会创建新版本并原子推进 HEAD，不存在需要作者再次“晋升”的工作稿或
正史版本。

模型由部署者或用户自行配置。正文和凭据默认保存在自己的服务器，
Readraft 不提供内置模型账号，也不会在缺少真实模型时伪造生成结果。

Codex 暂未接入。拆文能力作为辅助研究工具服务创作：逐章分析可以提取可迁移技法，作者保存、改造并主动绑定后，系统才会在对应的规划、正文创作或审校任务中读取这些抽象规则。

产品定位、当前实现边界和后续路线见
[《产品定义、当前状态与实现路线》](docs/product-definition-and-roadmap.md)。
核心写作的数据模型、工作流和实施顺序见
[《长篇小说创作核心：技术实现计划》](docs/core-writing-implementation-plan.md)。
当前竞争定位、实现差异与优势边界见
[《竞争定位与优势边界》](docs/competitive-positioning.md)。
当前唯一运行主链路和模块职责见
[《架构与代码边界》](docs/architecture.md)。

## 已实现

- 统一作品库：新建与导入共用一个入口；同一作品可在阅读、分析和创作之间切换。
- 唯一版本模型：每章只有一个可写 `main HEAD`，每次保存产生不可变历史；整书
  Tag 固定当时正文与作品资料并保持只读。自动暂存仅用于崩溃恢复，不是第二种版本。
- 导入文本先保存为不可变原始 Tag；需要续写或改写时再创建 `main`。自建作品直接
  从 `main` 开始。
- 统一工作台：桌面端为目录、正文、AI 三栏，移动端使用覆盖层；阅读 Tag 与编辑
  main 复用相同信息架构，但只读状态不会显示写入能力。
- 作品资料收敛为作品概览、世界、人物、剧情与结构、叙事与文风五类。作者可以
  手工维护，也可以在对话中让 AI 直接做受控的局部修改。
- 持久化对话保存完整历史。近期上下文直接进入模型，较早对话、正文、设定、历史、
  分析和参考资料通过虚拟工作区按需检索，不把整本书一次塞入请求。
- Agent 只访问 `book/` 虚拟工作区，不接触服务器文件系统。基础工具为
  `glob / read / grep / search / diff / edit / write / patch / delete / restore /
  web_search / web_fetch`；所有写入都会复核用户、作品、`main`、权限和 revision。
- 小说领域动作与文件动作分开：`create` 新建章节，`compose` 生成或大范围改写正文，
  `series` 连续创作多章，`task` 委托无工具、无写权限的专项分析。正文生成走独立
  纯文本流，不把 JSON、推理过程或审计说明写入稿件。
- 默认遵循作者意图直接执行：明确要求讨论或禁止写入时只讨论；明确要求创作或修改
  时直接创建可撤回版本，不在写完后再要求一次确认。
- 章节任务卡和 2–5 个场景节拍是写前规划资料，不再拥有另一套“场景草稿—组装—
  硬审核”正文链。创作、局部修改、撤回和恢复最终都只推进同一个章节 HEAD。
- 质量检查是非阻塞建议。文风审校可以定位具体原句并生成局部候选，但不会用字数
  下限或硬审核阻止保存；作者认为有问题时继续修改即可。
- HEAD 变化会使不匹配的分析失效、标记后续章节待复查，并按新正文重建 Story
  Memory；Tag 中冻结的内容和分析不受影响。
- Story Memory 可维护人物状态、关系、地点、物品、知情边界、剧情线、伏笔、时间
  和事件因果，并支持作者确认的别名归一与顺序重放。
- 全书蓝图、长期剧情线、分卷、未来章节骨架、任务卡和因果链接保持“计划不等于
  已发生事实”的边界；只有当前 HEAD 与确认后的 Story Memory 能成为既有故事事实。
- 参考书只用于阅读、证据化分析和抽象技法提炼。参考原文的独特措辞、专有名词和
  具体情节不会直接进入正文写作包；作者改造并绑定技法卡后才能参与创作。
- 支持 DeepSeek、OpenCode Go、自定义 OpenAI、OpenAI、Gemini 与 Ollama；用户可
  保存模型列表、默认模型和 Low / Standard / Max 路由，API Key 加密保存且可临时显隐。
- 支持 Provider 原生流式输出、推理能力差异、结构化输出和原生工具调用；不根据
  模型名称猜测统一参数。
- 可选联网使用 Exa MCP。只有作者要求查证、问题依赖近期事实或缺少必要外部资料时
  才授予搜索工具；网页内容始终作为不可信外部证据处理。
- TXT/Markdown 自动拆章、TXT 正文导出、可校验 `.readraft.zip` 作品归档、完整实例
  备份与恢复均已提供。
- `readraft-doctor` 检查 SQLite、外键、HEAD/父版本、不可变文件哈希、缓存、暂存、
  导入章节、Tag 清单与孤儿文件；修复非权威缓存和清理孤儿必须显式启用。
- 版本化 SQLite 迁移、单进程持久任务队列、服务重启续跑、自动更新脚本和低内存
  Azure 部署路径已实现。

## 本地运行

需要 Python 3.12+。项目根目录的 `.python-version` 固定本地基线，
`requirements*.lock` 固定并校验全部依赖及其下载哈希。

```bash
git clone https://github.com/ReBeating/Readraft.git
cd Readraft
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements-dev.lock
cp .env.example .env
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

打开 <http://127.0.0.1:8000>，注册或登录后：

1. 在“模型设置”中选择 DeepSeek、OpenCode Go、自定义 OpenAI、OpenAI、Google Gemini 或 Ollama，并保存对应模型。自定义 OpenAI 与 Ollama 可修改 Base URL，API Key 可选。
2. 可选：在“模型设置 → 联网”中决定是否允许 Agent 联网。默认使用 Exa 的免认证 MCP 服务，不要求用户填写搜索 Key。
3. 在首页直接“新建作品”。系统会进入设定页，不要求预先填写书名。
4. 在右侧对话框说出人物、画面、题材或阅读感受；需要时把 AI 整理出的候选设定应用到作品，再在五个作品资料分类中继续修改。
5. 从左侧目录新建章节，或直接要求 AI 创作第一章。明确要求创作后，正文会生成不可变版本并推进该章的 main HEAD。
6. 正文会自动保存。选择一段文字后直接向 AI 提问，可以分析、局部修订或继续写作；引用依据会回到对应不可变版本。
7. 不需要选择 Agent 或工作模式。服务端会结合当前页面、引用和明确要求，在交流想法、整理设定、创作正文、修改正文与分析作品之间调度。
8. 可选：导入参考文本。导入内容先作为不可变的“原始版本”供阅读、分析和引用；需要改写或续写时，再从它创建唯一的 `main`。
9. 只有 `main` 可以新增或修改正文。需要固定阶段成果时从 `main` 创建“一稿、二稿……”等 Tag；Tag 只读，并保存创建当时的作品资料。
10. 左侧目录只保留正文、作品资料、分析与笔记、版本。版本页统一展示每章 main HEAD、不可变历史和整书 Tag；旧版章节编辑、任务卡、场景和独立作品对话地址不再注册，访问直接返回 404。

管理员也可以在 `.env` 中设置共享 Key：

```dotenv
MODEL_PROVIDER=deepseek
MODEL_API_KEY=你的_api_key
MODEL_NAME=deepseek-v4-flash
```

调用顺序为：个人 Key 优先、服务器共享 Key 次之。没有可用 Key
时，应用仍可启动并完成账号和作品设置，但会明确要求先进入 API
设置，不会生成演示正文或假审计结果。确定性 mock 只在
`APP_ENV=test` 的自动化测试中启用。

## 模型接入与写作行为

当前通过统一 Provider 层接入以下模型端点：

```text
DeepSeek  https://api.deepseek.com/chat/completions
OpenCode Go  https://opencode.ai/zen/go/v1/{chat/completions|responses|messages}
自定义 OpenAI  {用户配置的 Base URL}/chat/completions
OpenAI    https://api.openai.com/v1/chat/completions
Gemini    https://generativelanguage.googleapis.com/v1beta/openai/chat/completions
Ollama    http://127.0.0.1:11434/v1/chat/completions
```

DeepSeek、OpenCode Go、OpenAI 与 Gemini 使用注册表中的固定官方地址。
OpenCode Go 会按模型自动选择 Chat Completions、Responses 或 Anthropic
Messages 协议。Ollama 默认连接本机，但可保存其他 API 根路径；“自定义
OpenAI”要求填写 Base URL，API Key 可选。两者都可读取标准 `/models` 目录，目录不可用时也
可直接输入模型 ID。Base URL 应填写到 API 根路径（通常以 `/v1`
结尾），不能包含凭据、查询参数，也不要填写 `/models` 或
`/chat/completions`。

正文创作统一经过 `compose` 的纯文本流：主 Agent 先在虚拟工作区读取和检索必要
资料，组装有界写作包，再交给写作模型生成完整章节结果。结构化 Planner、记忆提取、
技法与偏好提取等领域任务使用模型结构化输出并由 Pydantic 与服务端引用校验兜底。
场景节拍只是任务卡中的规划结构；正文不存在逐场景草稿库、组装门禁或独立硬审核。
文风审校和连续性问题只提供可追溯修改依据，不阻止作者保存或推进 main HEAD。

Agent 的模型回合可以连续调用多个工具；页面不显示工具预算。服务端仅保留最大回合、
重复无进展熔断、权限校验、并发 revision 和可撤回提交等安全边界。全书计划、未来
因果和未确认资料始终被标记为方向而非已发生事实；Writer 只读取本轮获权并实际检索到
的资料。官方 Provider 地址来自固定注册表；只有 Ollama 和自定义 OpenAI 可以保存
自定义地址。生产环境默认拒绝明显的本机或内网目标，并要求公网自定义地址使用 HTTPS；
只有部署者确认自托管目标可信后才应设置 `APP_ALLOW_PRIVATE_MODEL_BASE_URLS=true`。

个人 API Key 使用 Fernet 认证加密后保存在 SQLite。默认从 `APP_SECRET_KEY` 派生加密密钥；生产环境建议单独设置稳定的 `APP_CREDENTIAL_ENCRYPTION_KEY`。修改该密钥后，已有用户需要重新填写 API Key。

联网搜索使用 Exa 托管的 MCP 服务。它不依赖具体模型供应商，因此
DeepSeek、OpenCode Go、自定义 OpenAI、OpenAI、Gemini 和 Ollama 可以共用同一搜索能力。
默认无需认证；访问量增大后，部署者可以通过 `EXA_API_KEY` 统一扩充
额度，普通用户不需要接触搜索凭据。关闭账号的联网开关后，新对话不会
获得联网工具。Agent 只在作者明确要求搜索或核实、问题依赖近期变化，
或者缺少必要外部现实资料时搜索；纯构思、创作、改写和分析已有材料
不会触发搜索。网页结果只作为外部证据，不能改变系统提示词、工具权限
或作品事实。

官方参考：

- [DeepSeek API](https://api-docs.deepseek.com/)
- [DeepSeek Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode)
- [DeepSeek JSON Output](https://api-docs.deepseek.com/guides/json_mode/)
- [Exa MCP](https://exa.ai/docs/reference/exa-mcp)
- [Gemini OpenAI compatibility](https://ai.google.dev/gemini-api/docs/openai)
- [Ollama OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility)

## 测试

```bash
python -m ruff check app tests
python -m pytest
python -m pip_audit --disable-pip -r requirements.lock
git ls-files -z | xargs -0 detect-secrets-hook --baseline .secrets.baseline
```

测试不会调用真实模型服务。接口测试使用 `httpx.MockTransport`，
端到端流程在 `APP_ENV=test` 下使用确定性测试模型；开发和生产环境
均不会回退到这些模型。

GitHub Actions 会在 Python 3.12、3.13 和 3.14 上安装带哈希的锁定
依赖并执行静态检查与全量测试，同时独立执行运行依赖漏洞审计和已跟踪
文件的密钥扫描。调整 `requirements.txt` 或 `requirements-dev.txt`
后，使用 Python 3.12 和 `pip-tools` 重新生成两个锁文件：

```bash
python -m piptools compile --generate-hashes --strip-extras \
  --output-file=requirements.lock requirements.txt
python -m piptools compile --generate-hashes --strip-extras --allow-unsafe \
  --output-file=requirements-dev.lock requirements-dev.txt
```

如需验证真实个人模型，可运行只使用内置合成材料的冒烟命令。它不读取
作品正文，也不会打印 API Key 或模型生成内容：

```bash
python -m app.model_smoke --username 你的账号 --mode all
```

## 作品导入与导出

作品库中的 ZIP 操作会生成唯一格式的 `.readraft.zip` 完整作品归档。
“导入”会先校验归档格式、文件大小和 SHA-256，再创建一部新的独立
作品，不覆盖现有内容。归档会保留同一作品下唯一可写的 `main`、所有
不可变 Tag、版本来源、Tag 对应的作品资料快照、按内容版本绑定的分析
与笔记、作者确认设定、规划、人物、章节正文、正史记忆、审校结果、
对话，以及实际绑定的抽象技法卡。

作品正文与已确认作品资料共用一套版本关系：`main` 表示当前工作状态，
Tag 表示固定快照。分析结果引用具体版本，但可以在之后继续补充；聊天
和任务运行记录只作为过程证据保存，不参与 `main` 与 Tag 的内容比较。

作品归档不包含账号、密码、Cookie、模型 API Key 或其他作品；有 AI
任务正在排队或运行时也不会导出。默认归档上限为 256 MB，可通过
`APP_MAX_WORK_ARCHIVE_BYTES` 调整。

三种导出用途不同：

- TXT：只用于阅读或交稿，不能恢复结构化状态。
- 作品归档：迁移或复制一部作品，不携带账号与凭据。
- 完整备份：灾备整个实例，包含全部账号、作品、参考书和加密凭据。

## 完整备份与恢复

完整备份会保存一致的 SQLite 快照、章节正文、全部不可变版本和参考
资料，并使用清单与 SHA-256 逐文件校验。为避免恢复期间仍有任务写入，
以下命令要求先停止 Web 应用：

```bash
.venv/bin/python -m app.backup create /安全目录/readraft-backup.zip
.venv/bin/python -m app.backup verify /安全目录/readraft-backup.zip
.venv/bin/python -m app.backup restore /安全目录/readraft-backup.zip --replace
```

备份必须保存在 `APP_DATA_DIR` 之外。恢复会先在备份文件旁创建
`readraft-pre-restore-*.zip`，校验成功后才替换当前数据库和正文目录。
完整备份含账号、正文和加密后的个人 API Key，仍应按敏感文件保存；
`.env` 与加密密钥不会写入归档，需要另行安全保管。

## 生产部署与升级

完整的一次性安装、Nginx、systemd、备份、升级和故障处理步骤见
[生产部署与升级指南](docs/deployment.md)。

低内存 Linux VM（包括 Azure VM）必须保持：

- `uvicorn --workers 1`
- 应用内 AI 并发为 1
- Uvicorn 只监听 `127.0.0.1:8010`
- 仅通过 Nginx 或 Azure 托管入口的 HTTPS 对公网开放

示例文件：

- `deploy/readraft.service`
- `deploy/nginx.conf`

生产环境至少设置：

```dotenv
APP_ENV=production
APP_SECRET_KEY=<至少 32 字节随机值>
APP_CREDENTIAL_ENCRYPTION_KEY=<至少 32 字节且长期保持不变>
APP_COOKIE_SECURE=true
APP_ALLOW_REGISTRATION=false
# 可选；留空时每位用户在网页中填写个人 Key
MODEL_PROVIDER=deepseek
MODEL_API_KEY=
MODEL_NAME=deepseek-v4-flash
```

标准部署使用 `/opt/readraft`、专用系统账号 `readraft` 和
`readraft.service`。生产环境默认只跟踪带注释的稳定语义版本 Tag，不会
自动追踪 `main`。首次安装完成后，可以先检查再手动更新：

```bash
sudo /opt/readraft/deploy/update.sh --check
sudo /opt/readraft/deploy/update.sh
```

更新脚本会拒绝并发运行、脏工作树、非快进历史和非正式版本 Tag；在停机
前准备隔离的 Python 依赖环境，停机后创建并校验完整备份，再切换代码与
依赖、重启并检查 `/healthz`。如果新版本无法健康启动，会自动恢复旧
代码、旧依赖和更新前数据。

确认手动更新正常后，可以显式启用每日自动检查：

```bash
sudo /opt/readraft/deploy/install-auto-update.sh
```

Timer 默认每日运行并加入最多一小时随机延迟，配置保存在
`/etc/readraft/update.env`。数据库迁移在应用启动时自动执行。后台队列
与 Web 服务同进程，因此只能启动一个 Uvicorn worker；进程锁会拒绝
第二个实例，避免同一生成任务被重复执行和计费。

## 安全、隐私与贡献

- 安全问题请按 [SECURITY.md](SECURITY.md) 私下报告。
- 自托管数据边界与模型调用说明见 [PRIVACY.md](PRIVACY.md)。
- 本地开发、迁移约束和提交要求见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可证

Readraft 的代码和项目文档采用
[Apache License 2.0](LICENSE) 发布。该许可证不改变你对自己导入、
创作或生成内容所拥有的权利，也不会替你取得参考作品或模型输出的使用
授权；使用者仍需自行确认相关内容与模型服务条款。
