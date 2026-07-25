import pytest
from pydantic import ValidationError

from app.style_editor import locate_style_issues, numbered_paragraphs
from app.style_schema import StyleAuditResult


def test_style_issue_locations_are_exact_and_hallucinations_are_dropped():
    text = (
        "林岚把信放到窗边。显然，这个邮戳不可能来自昨天。\n\n"
        "她没有解释，只把日期抄进笔记。"
    )
    paragraphs = numbered_paragraphs(text)
    assert len(paragraphs) == 2
    assert text[
        paragraphs[1]["start_offset"] : paragraphs[1]["end_offset"]
    ] == paragraphs[1]["text"]

    result = StyleAuditResult.model_validate(
        {
            "summary": "一条有效定位，一条模型臆造定位。",
            "issues": [
                {
                    "paragraph_index": 1,
                    "quote": "显然，这个邮戳不可能来自昨天。",
                    "issue_type": "over_explanation",
                    "severity": "medium",
                    "evidence": "直接替读者下结论。",
                    "reader_impact": "削弱读者依据证据推断的参与感。",
                    "rewrite_direction": "删去判断标签，让动作和证据承担推理。",
                },
                {
                    "paragraph_index": 2,
                    "quote": "正文里并不存在的句子。",
                    "issue_type": "generic_atmosphere",
                    "severity": "low",
                    "evidence": "这条定位是模型臆造。",
                    "reader_impact": "无法核对。",
                    "rewrite_direction": "不应保存。",
                },
            ],
        }
    )
    located, dropped = locate_style_issues(text, result)
    assert dropped == 1
    assert len(located) == 1
    issue = located[0]
    assert text[issue["start_offset"] : issue["end_offset"]] == issue["quote"]


def test_repeated_quote_in_one_paragraph_is_dropped_as_ambiguous():
    text = "她说“我知道”。他没有回答。她又说“我知道”。"
    result = StyleAuditResult.model_validate(
        {
            "summary": "同一段中的引文无法唯一定位。",
            "issues": [
                {
                    "paragraph_index": 1,
                    "quote": "我知道",
                    "issue_type": "repetition",
                    "severity": "medium",
                    "evidence": "同一句话在段内重复。",
                    "reader_impact": "重复削弱了对话推进。",
                    "rewrite_direction": "只保留有动作后果的一次。",
                }
            ],
        }
    )

    located, dropped = locate_style_issues(text, result)

    assert located == []
    assert dropped == 1


def test_style_audit_schema_enforces_prompt_issue_limit():
    issue = {
        "paragraph_index": 1,
        "quote": "可定位原句",
        "issue_type": "over_explanation",
        "severity": "low",
        "evidence": "直接解释。",
        "reader_impact": "减少读者参与。",
        "rewrite_direction": "让动作承担信息。",
    }

    with pytest.raises(ValidationError):
        StyleAuditResult.model_validate(
            {
                "summary": "模型返回的问题数量超过审校契约。",
                "issues": [issue for _ in range(13)],
            }
        )
