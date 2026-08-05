from __future__ import annotations

from .memory_identity import IDENTITY_TYPE_LABELS


def _human_size(value: int) -> str:
    if value < 1_000:
        return f"{value} 字"
    if value < 1_000_000:
        return f"{value / 1_000:.1f} 千字"
    return f"{value / 1_000_000:.2f} 百万字"


def _status_label(value: str) -> str:
    return {
        "ready": "等待分析",
        "planned": "待创作",
        "written": "已写",
        "final": "已定稿",
        "queued": "排队中",
        "running": "分析中",
        "completed": "已完成",
        "partial": "部分完成",
        "failed": "失败",
    }.get(value, value)


def _writing_status_label(value: str) -> str:
    return {
        "queued": "排队中",
        "running": "写作中",
        "completed": "已完成",
        "failed": "失败",
    }.get(value, value)


def _writing_operation_label(value: str) -> str:
    return {
        "draft": "生成初稿",
        "continue": "续写",
        "rewrite": "整章重写",
        "polish": "润色",
        "manual": "手动保存",
        "extract_story_delta": "提取故事记忆",
        "plan_chapter": "规划章节任务卡",
        "plan_scene_beats": "只拆分场景节拍",
        "audit_ai_style": "定位 AI 味问题",
        "rewrite_style_issue": "定点改写",
        "targeted_rewrite": "定点改写候选",
        "propose_reader_branches": "评估读者意见",
    }.get(value, value)


def _style_issue_label(value: str) -> str:
    return {
        "abstract_emotion": "抽象概括情绪",
        "over_explanation": "过度解释",
        "uniform_rhythm": "句段节奏过齐",
        "generic_atmosphere": "通用氛围",
        "cliche": "陈词滥调",
        "dialogue_convergence": "人物对话趋同",
        "over_complete_paragraph": "段落过度完整",
        "unnecessary_summary": "无必要总结",
        "repetition": "重复信息",
        "non_specific_detail": "伪具体细节",
    }.get(value, value)


def _voice_suggestion_status_label(value: str) -> str:
    return {
        "queued": "排队中",
        "running": "正在分析",
        "ready": "等待作者审核",
        "applied": "已应用",
        "rejected": "已放弃",
        "failed": "提取失败",
    }.get(value, value)


def _story_plan_status_label(value: str) -> str:
    return {
        "queued": "排队中",
        "running": "正在规划",
        "completed": "可比较与采纳",
        "failed": "生成失败",
    }.get(value, value)


def _story_planning_mode_label(value: str) -> str:
    return {
        "create": "从项目资料建立结构",
        "refine": "优化已确认方向",
        "rethink": "重想未来结构",
    }.get(value, value)


def _chapter_structure_role_label(value: str) -> str:
    return {
        "setup": "建立",
        "escalation": "升级",
        "reversal": "反转",
        "payoff": "兑现",
        "transition": "转场",
    }.get(value, value or "待定义")


def _voice_dimension_label(value: str) -> str:
    return {
        "narration": "叙述距离",
        "rhythm": "句段节奏",
        "dialogue": "对话声音",
        "sensory": "感官与意象",
        "metaphor": "比喻策略",
        "omission": "省略与留白",
    }.get(value, value)


def _edit_preference_category_label(value: str) -> str:
    return {
        "diction": "用词",
        "sentence_rhythm": "句段节奏",
        "narration_distance": "叙述距离",
        "dialogue": "对话",
        "emotional_expression": "情绪表达",
        "sensory_detail": "感官细节",
        "metaphor": "比喻",
        "omission": "留白",
        "paragraph_structure": "段落结构",
        "other": "其他",
    }.get(value, value)


def _edit_preference_status_label(value: str) -> str:
    return {
        "queued": "排队中",
        "running": "正在分析",
        "ready": "等待作者审核",
        "applied": "已确认偏好",
        "rejected": "已放弃",
        "failed": "提取失败",
    }.get(value, value)


def _story_arc_type_label(value: str) -> str:
    return {
        "main": "主线",
        "subplot": "支线",
        "character": "人物弧光",
        "relationship": "关系线",
        "mystery": "谜团线",
        "world": "世界线",
    }.get(value, value)


def _story_arc_lifecycle_label(value: str) -> str:
    return {
        "planned": "计划中",
        "active": "正在推进",
        "paused": "暂缓推进",
        "resolved": "计划收束",
        "abandoned": "已放弃",
    }.get(value, value)


def _reader_request_type_label(value: str) -> str:
    return {
        "pace": "节奏",
        "character": "人物",
        "relationship": "关系",
        "plot": "剧情",
        "world": "世界设定",
        "payoff": "回报 / 爽点",
        "other": "其他",
    }.get(value, value)


def _reader_scope_label(value: str) -> str:
    return {
        "next_chapter": "下一章",
        "next_three": "未来三章",
        "current_volume": "当前分卷",
        "long_term": "长期主线",
    }.get(value, value)


def _reader_status_label(value: str) -> str:
    return {
        "draft": "待评估",
        "proposing": "正在生成方案",
        "reviewing": "等待作者选择",
        "adopted": "已采纳",
        "dismissed": "已归档",
        "failed": "生成失败",
    }.get(value, value)


def _impact_item_type_label(value: str) -> str:
    return {
        "chapter": "后续正史章节",
        "fact": "后续事实",
        "knowledge": "人物知情",
        "plot_thread": "剧情线",
        "foreshadowing": "伏笔",
    }.get(value, value)


def _technique_dimension_label(value: str) -> str:
    return {
        "plot": "剧情",
        "structure": "结构",
        "scene": "场景",
        "pacing": "节奏",
        "information": "信息释放",
        "character": "人物",
        "dialogue": "对话",
        "language": "语言",
        "suspense": "悬念",
    }.get(value, value)


def _reference_style_axis_label(value: str) -> str:
    return {
        "point_of_view": "叙事视角",
        "narrative_distance": "叙事距离",
        "sentence_rhythm": "句子节奏",
        "paragraph_rhythm": "段落节奏",
        "dialogue": "对话组织",
        "description": "描写策略",
        "information_flow": "信息流",
        "emotion": "情绪传达",
        "diction": "用词语域",
        "figurative_language": "修辞与意象",
        "transition": "转场方式",
        "scene_entry_exit": "场景出入",
    }.get(value, value)


def _technique_scope_label(value: str) -> str:
    return {
        "project": "全书",
        "volume": "分卷",
        "chapter": "章节",
        "scene": "场景",
    }.get(value, value)


def _technique_usage_label(value: str) -> str:
    return {
        "plan": "规划",
        "write": "正文",
        "audit": "审校",
    }.get(value, value)


def _continuity_issue_label(value: str) -> str:
    return {
        "state_before_mismatch": "人物状态前后不一致",
        "relationship_before_mismatch": "人物关系前后不一致",
        "location_before_mismatch": "地点连续性冲突",
        "item_holder_mismatch": "物品持有者冲突",
        "item_after_destroyed": "已毁物品再次出现",
        "story_time_mismatch": "故事时间衔接冲突",
        "missing_baseline": "缺少可核对的前置状态",
        "knowledge_without_baseline": "遗忘缺少知情基线",
        "plot_thread_without_setup": "剧情线缺少建立记录",
        "plot_thread_duplicate_open": "剧情线重复建立",
        "plot_thread_reopened": "已关闭剧情线重新开启",
        "plot_thread_after_closed": "已关闭剧情线继续推进",
        "foreshadow_without_setup": "伏笔缺少埋设记录",
        "foreshadow_duplicate_setup": "伏笔重复埋设",
        "foreshadow_reopened": "已关闭伏笔重新埋设",
        "foreshadow_after_closed": "已关闭伏笔继续推进",
        "duplicate_event_identity": "事件身份重复",
        "causal_self_reference": "事件因果自指",
        "causal_reference_missing": "直接原因事件缺失",
    }.get(value, value)


def _memory_identity_type_label(value: str) -> str:
    return IDENTITY_TYPE_LABELS.get(value, value)


def _story_memory_label(value: str) -> str:
    return {
        "status": "状态",
        "location": "位置",
        "physical": "身体",
        "emotional": "情绪",
        "goal": "目标",
        "ability": "能力",
        "possession": "持有",
        "other": "其他",
        "created": "产生",
        "acquired": "获得",
        "lost": "丢失",
        "transferred": "转移",
        "used": "使用",
        "destroyed": "毁坏",
        "changed": "变化",
        "knows": "知道",
        "suspects": "怀疑",
        "believes_false": "误信",
        "forgets": "遗忘",
        "main": "主线",
        "subplot": "支线",
        "relationship": "关系线",
        "mystery": "谜团",
        "promise": "承诺线",
        "opened": "建立",
        "advanced": "推进",
        "paused": "暂停",
        "resolved": "解决",
        "abandoned": "放弃",
        "setup": "埋设",
        "payoff": "回收",
    }.get(value, value)


def _knowledge_state_label(value: str) -> str:
    return {
        "knows": "知道",
        "suspects": "怀疑",
        "believes_false": "相信错误信息",
        "forgets": "已经遗忘",
    }.get(value, value)


def _plot_status_label(value: str) -> str:
    return {
        "open": "已建立",
        "active": "推进中",
        "paused": "暂缓",
        "resolved": "已解决",
        "abandoned": "已放弃",
    }.get(value, value)


def _foreshadow_status_label(value: str) -> str:
    return {
        "setup": "已埋设",
        "advanced": "已推进",
        "payoff": "已回收",
        "abandoned": "已放弃",
    }.get(value, value)
