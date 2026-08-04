from pathlib import Path

import pytest

from app.assistant_chat_schema import AssistantStructuredSettingEdit
from app.assistant_chat_schema import AssistantSettingsPatch
from app.db import Database
from app.security import hash_password
from app.structured_settings import (
    StructuredSettingsEditor,
    filter_structured_edits,
    normalize_changes,
    preview_structured_edits,
)


def _build_project(tmp_path: Path):
    database = Database(tmp_path / "structured-settings.db")
    database.initialize()
    user_id = database.create_user(
        "structured-settings-author",
        hash_password("password-123"),
    )
    project_id = "structured-settings-project"
    database.create_novel_project(
        user_id=user_id,
        project_id=project_id,
        title="潮汐盲区",
        genre="悬疑",
        premise="记者追查被海雾抹去的港口事故。",
        world_setting="当代海港。",
        style_guide="克制、具体。",
        point_of_view="第三人称限知",
        target_chapter_chars=3000,
    )
    character_id = database.add_novel_character(
        user_id=user_id,
        project_id=project_id,
        name="林悦",
        role="调查记者",
        traits="谨慎",
        background="在雾港长大",
        character_arc="从独自查案到学会合作",
        external_goal="查明事故真相",
        internal_need="证明自己没有判断失误",
        central_conflict="越接近真相越会伤害家人",
        secret="曾删除一段关键录音",
        speech_style="句子短，先追问证据",
        initial_state="不信任任何线人",
    )
    return database, user_id, project_id, character_id


def _apply(
    editor: StructuredSettingsEditor,
    *,
    user_id: int,
    project_id: str,
    edit: AssistantStructuredSettingEdit,
    baseline: dict,
):
    with editor.database.connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        result = editor.apply_in_connection(
            connection,
            user_id=user_id,
            project_id=project_id,
            edits=[edit],
            baseline_snapshot=baseline,
        )
        connection.commit()
    return result


def test_character_micro_edit_preserves_unselected_fields_and_allows_unrelated_change(
    tmp_path: Path,
):
    database, user_id, project_id, character_id = _build_project(tmp_path)
    editor = StructuredSettingsEditor(database)
    baseline = editor.snapshot(user_id=user_id, project_id=project_id)
    original = baseline["characters"][0]

    assert database.update_novel_character(
        user_id=user_id,
        project_id=project_id,
        character_id=character_id,
        name=original["name"],
        role="首席调查记者",
        traits=original["traits"],
        background=original["background"],
        character_arc=original["character_arc"],
        external_goal=original["external_goal"],
        internal_need=original["internal_need"],
        central_conflict=original["central_conflict"],
        secret=original["secret"],
        speech_style=original["speech_style"],
        initial_state=original["initial_state"],
    )

    applied = _apply(
        editor,
        user_id=user_id,
        project_id=project_id,
        baseline=baseline,
        edit=AssistantStructuredSettingEdit(
            entity_type="character",
            action="update",
            target_id=character_id,
            changes={"internal_need": "承认自己害怕再次误判"},
            reason="让人物动机更具体。",
        ),
    )
    assert applied[0]["fields"] == [
        {
            "field": "internal_need",
            "label": "内在需求与动机",
            "before": "证明自己没有判断失误",
            "after": "承认自己害怕再次误判",
            "changed": True,
        }
    ]
    updated = editor.snapshot(
        user_id=user_id,
        project_id=project_id,
    )["characters"][0]
    assert updated["role"] == "首席调查记者"
    assert updated["internal_need"] == "承认自己害怕再次误判"
    assert updated["secret"] == "曾删除一段关键录音"


def test_character_micro_edit_rejects_stale_change_to_same_field(
    tmp_path: Path,
):
    database, user_id, project_id, character_id = _build_project(tmp_path)
    editor = StructuredSettingsEditor(database)
    baseline = editor.snapshot(user_id=user_id, project_id=project_id)
    original = baseline["characters"][0]

    assert database.update_novel_character(
        user_id=user_id,
        project_id=project_id,
        character_id=character_id,
        name=original["name"],
        role=original["role"],
        traits=original["traits"],
        background=original["background"],
        character_arc=original["character_arc"],
        external_goal=original["external_goal"],
        internal_need="已经决定向搭档坦白",
        central_conflict=original["central_conflict"],
        secret=original["secret"],
        speech_style=original["speech_style"],
        initial_state=original["initial_state"],
    )

    with pytest.raises(ValueError, match="在讨论后已经变化"):
        _apply(
            editor,
            user_id=user_id,
            project_id=project_id,
            baseline=baseline,
            edit=AssistantStructuredSettingEdit(
                entity_type="character",
                action="update",
                target_id=character_id,
                changes={"internal_need": "继续掩盖自己的失误"},
            ),
        )


def test_structured_candidate_can_apply_only_selected_fields(
    tmp_path: Path,
):
    database, user_id, project_id, character_id = _build_project(tmp_path)
    editor = StructuredSettingsEditor(database)
    baseline = editor.snapshot(user_id=user_id, project_id=project_id)
    edit = AssistantStructuredSettingEdit(
        entity_type="character",
        action="update",
        target_id=character_id,
        changes={
            "internal_need": "学会把判断交给可以信任的人复核",
            "speech_style": "避免连续追问，关键时刻才短句发问",
        },
    )
    previews = preview_structured_edits([edit], baseline)
    assert [field["field"] for field in previews[0]["fields"]] == [
        "internal_need",
        "speech_style",
    ]
    selected = filter_structured_edits(
        [edit],
        {"structured:0:internal_need"},
    )
    assert selected[0].changes == {
        "internal_need": "学会把判断交给可以信任的人复核"
    }

    _apply(
        editor,
        user_id=user_id,
        project_id=project_id,
        baseline=baseline,
        edit=selected[0],
    )
    updated = editor.snapshot(
        user_id=user_id,
        project_id=project_id,
    )["characters"][0]
    assert updated["internal_need"] == "学会把判断交给可以信任的人复核"
    assert updated["speech_style"] == "句子短，先追问证据"


def test_structured_changes_normalize_common_model_shapes():
    assert normalize_changes(
        "character",
        {"traits": ["谨慎", "克制", "责任感强"]},
    ) == {"traits": "谨慎、克制、责任感强"}
    assert normalize_changes(
        "story_blueprint",
        {"major_turns": "发现异常；潜入广播塔；做出选择"},
    ) == {
        "major_turns": ["发现异常", "潜入广播塔", "做出选择"]
    }
    assert normalize_changes(
        "world_entry",
        {"entry_type": "科技规则", "name": "声纹档案系统"},
    ) == {"entry_type": "rule", "name": "声纹档案系统"}
    assert normalize_changes(
        "plot_arc",
        {
            "arc_type": "主线",
            "title": "声纹迷踪",
            "lifecycle_status": "planning",
            "priority": "high",
        },
    ) == {
        "arc_type": "main",
        "title": "声纹迷踪",
        "lifecycle_status": "planned",
        "priority": 1,
    }


def test_plot_arc_candidate_rejects_missing_confirmable_fields():
    edit = AssistantStructuredSettingEdit(
        entity_type="plot_arc",
        action="create",
        target_name="声纹迷踪",
        changes={
            "arc_type": "main",
            "title": "声纹迷踪",
            "dramatic_question": "谁篡改了港口广播？",
            "target_payoff": "沈砚找到篡改者并理解哥哥的选择。",
        },
    )

    with pytest.raises(ValueError, match="对读者的承诺"):
        preview_structured_edits([edit], {})


def test_settings_patch_omits_blank_optional_text():
    patch = AssistantSettingsPatch.model_validate(
        {
            "title": "第七码头的回声",
            "ending_constraint": "",
            "style_guide": "   ",
        }
    )

    assert patch.title == "第七码头的回声"
    assert patch.ending_constraint is None
    assert patch.style_guide is None
