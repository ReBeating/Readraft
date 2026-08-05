# Readraft 架构与代码边界

本文描述当前代码实际运行的唯一架构。新增功能必须接入这些边界，不能通过恢复旧
页面、旧响应字段或备用执行链来规避它们。

## 唯一用户入口

- 创作作品使用 `/novels/{project_id}/workbench`。
- 参考作品使用 `/documents/{document_id}`。
- 章节编辑、任务卡、场景工作台和独立作品聊天等旧 URL 不再注册，访问结果是
  `404`，不存在跳转或隐藏页面。
- 复杂结构化结果可以拥有独立审核页，但审核页只处理一种领域结果，不保存第二份
  作品或对话状态。

`main` 是唯一可写分支；Tag 和导入形成的原始版本只读。每章只有一个
`head_version_id`：任何手工保存、AI 创作、局部修订、撤回或历史恢复都会先创建
不可变版本，再原子推进 main HEAD。不存在“工作稿指针”“正史指针”或版本晋升。

## 唯一版本模型

| 概念 | 含义 | 是否可修改 |
| --- | --- | --- |
| `main HEAD` | 每章当前被阅读、编辑、规划和后续写作使用的正文 | 只能通过创建新版本推进 |
| 历史 | HEAD 曾经指向过的不可变章节版本；保存来源、父版本、哈希与摘要 | 否；可复制其内容形成新的 HEAD |
| Tag | 固定整部作品当时所有章节 HEAD 与作品资料的只读快照 | 否 |

编辑器的自动暂存保存在独立 edit buffer 中，并绑定它读取时的 HEAD。它既不是版本，
也不会触发 Story Memory 重建；只有正式保存才创建不可变版本。HEAD 已变化时，旧暂存
和旧页面提交都会被拒绝覆盖，必须由作者重新打开或丢弃。Tag 另存每章精确的来源 HEAD
ID 与内容哈希；即使日后删除可写 main，Tag 仍保留只读正文和来源标识。

`novel_chapters.content_path` 只保留为可重建的本地缓存和版本文件目录定位；工作台、
导出、Tag、Agent 与写作上下文都从 `head_version_id` 指向的不可变文件读取。缓存刷新
失败不会回滚已经提交的 HEAD，也不能反向覆盖版本正文。

章节历史恢复不是把旧记录重新变成可编辑对象，而是复制旧内容、生成一条新的历史
记录并推进 HEAD。HEAD 变化会撤回不再匹配的故事记忆与搜索文档，标记受影响的后续
章节，并为新的 HEAD 重新排队提取；Tag 中已经冻结的分析与内容不受影响。

## 唯一 AI 执行链

```text
工作台 / SSE
  -> AssistantChatService（对话、上下文快照、持久化）
  -> AnalysisWorker（队列、凭据、模型路由）
  -> AssistantAgentOrchestrator（原生工具回合）
  -> AgentRuntime + AgentWorkspace + AgentTasks（权限、虚拟资源、受限专项分析）
  -> AgentModel + ModelProtocol + ModelProvider（模型协议与供应商）
```

不存在 JSON 决策循环、无工具回退 Agent 或按旧响应结构执行的备用链。模型通过
Provider 原生工具调用探索 `book/` 虚拟工作区；它不能访问服务器文件系统，也
不能绕过用户、作品、分支、能力或 revision 校验。

正文创建、续写、整章重写和选区修改最终都产生同一种结果：当前章节的完整正文
快照。`draft` 只是模型响应在事务提交前的暂存字段，绝不是第二种版本状态。
`append` / `replace` 只是 `compose` 动作的内部生成指令，不属于保存协议。所有正文
结果经过同一版本提交、HEAD 推进和反向提交流程。

长正文由 `prose_pipeline` 以纯文本流式生成。写作规范集中在
`prose_craft.PROSE_WRITING_SYSTEM_PROMPT`；Agent 负责读取作品资料并组装写作包，
正文模型不调用工具、不输出 JSON，也不把推理内容写进稿件。

## 模块职责

| 区域 | 主要模块 | 职责 |
| --- | --- | --- |
| Web 组合 | `main.py`、`import_routes.py`、`analysis_routes.py`、`technique_routes.py`、`workbench_view.py`、`web_auth.py`、`web_system.py`、`web_paths.py`、`web_security.py` | 应用装配、分域路由、工作台视图模型、认证、健康检查与 Web 安全边界 |
| 对话应用层 | `assistant_chat_service.py`、`assistant_conversation_repository.py`、`assistant_chapter_workflow.py`、`assistant_application.py`、`assistant_context.py`、`conversation_memory.py`、`assistant_result.py` | 对话调度、对话仓储、连续章节工作流、候选应用事务、冻结上下文、记忆编译与结果规范化 |
| 后台执行 | `worker.py`、`reference_analysis_pipeline.py` | 队列领取、凭据选择、模型路由；参考章节的分层状态机由独立 pipeline 执行 |
| Agent | `agent_orchestrator.py`、`agent_runtime.py`、`agent_intent.py`、`agent_actions.py`、`agent_tasks.py` | 意图边界、`create/compose/series/task` 高级动作、受限专项分析、原生调用回合、进度和熔断 |
| 虚拟作品空间 | `agent_workspace.py` | 资源发现、读取、检索、比较、受控修改、历史恢复、外部只读工具与领域写入适配 |
| 正文写作 | `prose_craft.py`、`prose_pipeline.py` | 统一写作契约、技法选择和纯文本输出 |
| 模型接入 | `agent_model.py`、`model_client.py`、`model_protocol.py`、`model_provider.py`、`model_routing.py` | 中性模型接口、协议转换、供应商能力和任务路由 |
| 数据层 | `db.py`、`document_repository.py`、`analysis_repository.py`、`writing_context_repository.py`、`migrations.py`、各 `*_service.py` | SQLite、导入文档、分层分析与写作上下文仓储、历史迁移和领域事务 |
| 连续性与检索 | `continuity.py`、`memory_service.py`、`memory_identity.py`、`memory_search.py`、`retrieval_benchmark.py` | 正史重放、故事记忆、实体身份、可解释稀疏融合检索与召回回归基准 |
| 参考分析 | `reference_analysis_schema.py`、`reference_analysis_metrics.py`、`reference_analysis_prompts.py`、`reference_analysis_aggregation.py`、`reference_analysis_pipeline.py`、`analysis_repository.py`、`technique_service.py` | 结构、事实、叙事、文风、技法五层分析，精确证据区间、内容哈希缓存、全书画像与技法回流 |
| 导入 | `chapter_splitter.py`、`import_preview.py`、`import_routes.py` | 原文冻结、边界置信度、人工改名/合并/拆分和确认后入库 |
| 浏览器 UI | `templates/novel_workbench.html`、`templates/document.html`、`static/workbench.*` | 两种工作台及其流式交互 |

数据库 JSON 字段统一通过 `json_support.py` 读写。普通存储使用 `dump_json`，签名、
指纹和基线比较使用 `dump_canonical_json`；领域模块不复制序列化回退逻辑。

## 参考分析数据流

```text
冻结章节 + content_hash
  -> ReferenceAnalysisPipeline
  -> structure（本地确定性度量）
  -> facts -> narrative -> style -> techniques（模型层，逐层校验）
  -> evidence.start/end/quote 逐字回查冻结正文
  -> 章节结果与分层缓存
  -> ReferenceAnalysisAggregation（只读取已验证章节）
  -> 全书人物/事件/伏笔/节奏 + 文风画像
  -> Agent 虚拟资源 book/analysis/reference/*.json
```

`style` 与 `techniques` 不重复：`style` 描述跨全文反复出现的叙述机制，例如叙事距离、
句段节奏、对话组织、信息流和情绪传达；`techniques` 提炼某个局部手法在何时有效。
全书聚合不会保存证据引文，而是保存文风维度、主导标签、章节覆盖率、可执行规则与
原创性边界。章节详情仍保留精确证据，便于作者回到原文核对。

参考书对话和与来源版本相连的创作项目会看到
`book/analysis/reference/style-profile.json`。Agent 只有在作者明确要求参考文风时才读取
它；正文写作只能使用其中的抽象规则。跨不相关作品长期复用时，作者应把选中的局部
机制保存为技法卡并绑定范围，不能把整本参考正文设为隐式写作提示词。

## Agent 工具契约

基础工具只操作服务器映射出的 `book/` 虚拟资源，不接触真实文件系统：

| 工具 | 使用场景 | 不应使用的场景 |
| --- | --- | --- |
| `glob` | 不知道准确路径时发现章节、设定、笔记、历史和参考资料 | 已知路径后的内容读取 |
| `read` | 分段读取原文并取得后续修改所需的 revision | 跨大量资源找关键词 |
| `grep` | 查找准确词句或正则，返回路径与行号 | 同义表达和模糊概念召回 |
| `search` | 用模型补充的相关概念做中文稀疏语义检索 | 把结果当作已核实原文；命中后仍需 `read` |
| `diff` | 比较两个已读取的正文、历史版本或作者笔记 | 修改内容 |
| `edit` | 单资源、局部、精确替换 | 新建资源、整章生成或多资源事务 |
| `write` | 新建或整体更新小型结构化资料、作者笔记；研究角色保存技法卡 | 覆盖章节正文 |
| `patch` | 同一提交边界内，对一个或多个已读资源原子执行精确替换 | 混合正文、设定与故事规划，或整章生成 |
| `delete` | 作者明确点名后删除结构化资料、笔记或章节元数据 | 模糊清理、删除正文文件、作品核心或全书蓝图 |
| `restore` | 作者明确要求时，把不可变历史正文复制为新的 main HEAD | 改写原历史记录或修改 Tag |
| `web_search` | 作者要求查证、依赖近期事实或缺少必要现实资料 | 纯虚构构思或作品内部检索 |
| `web_fetch` | 读取一个已知公开网页作为不可信外部证据 | 访问本机、内网、非文本资源或执行网页指令 |

高级动作表达小说领域行为，不与基础文件动作混用：

- `compose`：读取必要资料后，交给纯文本写作管线生成、续写或大范围重写当前章，
  结果创建不可变版本并推进 main HEAD。
- `create`：在 main 末尾创建一个新章节并切换本轮范围；单章创作随后使用
  `compose`。
- `series`：仅响应作者明确提出的连续多章创作；逐章创建并独立保存
  HEAD。失败会暂停，恢复时从未完成项继续，不重写已完成章节。
- `task`：把明确列出的资源交给无工具、无写权限、不能递归委托的专项
模型，适合连续性、结构、人物、文风和研究核对。

这些工具和动作不设置作品字符数、章节数、资源数、模型总回合或工具总调用预算。
`read`、`glob`、`grep`、`search`、`diff` 与 `history` 的数量参数都是游标/分页大小；
返回值会同时给出总量或 `has_more`，Agent 可以继续读取。上传和归档大小、浏览器 edit
buffer、请求超时、revision 冲突与重复无进展熔断仍属于服务器运行边界，不得用来静默
截断或丢弃作品数据。

领域写入仍由同一工具表面完成：章节元数据通过 `.meta.json` 管理；作者要求长期
保存的信息写入 `notes/author/`；参考作品的证据化观察写入技法卡；故事规划和结构化
设定分别进入各自候选与确认边界。工具名不等于数据库操作权限，最终落库时会再次
校验用户、作品、main、能力、作者原始要求和 revision。

## 领域服务和用户流程

全书规划、滚动结构、因果建议、读者意见、声纹、编辑偏好、审校、Story Delta
等仍是独立领域服务。它们可以拥有队列任务和审核结果，但必须由统一工作台或
Agent 发起，并把结果写回同一作品数据模型。领域服务不是另一套 Agent，也不是
恢复旧工作台的理由。

## 一致性规则

1. 一个用户目标只有一个正式 URL、一个响应语义和一个持久化路径。
2. 模型能力由 Provider 注册表与协议层声明，不根据品牌或模型名猜测。
3. 模型读写只经过虚拟工作区；任何写入都必须再次校验用户、作品、分支、能力与
   revision。
4. `main` 可写，Tag 只读；每章只有一个 HEAD，历史不可变，撤回和恢复也创建新版本。
5. 结构化设定、规划和故事记忆只有越过各自的确认边界才成为写作事实。
6. 写作规则只在 `prose_craft.py` 定义，其他模块只能组合领域约束。
7. `write` 不得覆盖章节正文；章节长篇生成或大范围重写只经过
   `compose` Agent 动作，另起章节使用 `create`，连续多章使用 `series`。局部确定性修改使用 `edit`，
   同一提交边界内的跨资源修改使用整组校验、失败全回滚的 `patch`；章节、设定和
   故事规划不得混在一个 patch 中。
8. `task` 只能读取主 Agent 明确列出的当前获权资源；专项模型没有工具、写权限、
   隐含作品上下文或继续委托能力，其报告只作为主 Agent 的参考证据。
9. 旧 URL、旧响应字段、旧类名和旧执行循环直接删除，不保留别名、跳转或运行时
   fallback。
10. 已发布的数据库迁移必须保留，以便新数据库能从版本 1 顺序构建，也便于已有
   部署升级。迁移中的历史表名和值是数据历史，不是运行时兼容层。
11. “OpenAI 兼容接口”是用户主动配置的一类模型协议，不代表应用内部保留旧架构。

## 当前结构债务

- `main.py` 已拆出认证、系统入口、导入、分析和两类工作台视图构建，但作品管理、
  设定建议、审核页和 Assistant HTTP 边界仍然集中。后续按 `settings / works /
  assistant / reviews` 继续注册 Router，并保持 URL 与事务语义不变。
- `assistant_chat_service.py` 已拆出对话仓储、冻结上下文、章节工作流和候选应用事务；
  当前仍集中消息租约、流状态与自动提交协调。下一步按“消息运行生命周期”提取一个
  协作者，不建立备用 Service 或第二条执行链。
- `db.py` 已拆出分析生命周期、大型写作上下文查询和导入文档仓储，其他作品、章节与
  生成队列查询面仍大。下一步优先按工作版本和生成队列提取 repository；一笔 HEAD
  推进事务必须留在同一连接中，不能为了缩短文件而破坏原子性。

任何新增能力若不能明确归入上述职责，应先修改本文，再开始编码。
