from app.version_diff import build_version_diff


def test_whole_chapter_diff_tracks_additions_removals_and_blocks():
    original = "第一段没有变化。\n她解释自己很害怕。\n中间标记。\n结尾。"
    replacement = (
        "第一段没有变化。\n她把杯子推远，手指仍压着杯沿。\n中间标记。\n"
        "门外传来脚步声。\n结尾。"
    )

    result = build_version_diff(original, replacement)

    assert result["changed_blocks"] == 2
    assert result["added_chars"] > 0
    assert result["removed_chars"] > 0
    assert result["delta_chars"] == (
        result["replacement_chars"] - result["original_chars"]
    )
    assert any(
        segment["kind"] == "removed" and "解释自己很害怕" in segment["text"]
        for segment in result["before"]
    )
    assert any(
        segment["kind"] == "added" and "手指仍压着杯沿" in segment["text"]
        for segment in result["after"]
    )


def test_whole_chapter_diff_normalizes_line_endings():
    result = build_version_diff("第一段。\r\n第二段。", "第一段。\n第二段。")

    assert result["identical"]
    assert result["changed_blocks"] == 0
    assert result["added_chars"] == 0
    assert result["removed_chars"] == 0


def test_large_replacement_uses_bounded_block_fallback():
    original = "旧" * 80
    replacement = "新" * 90

    result = build_version_diff(
        original, replacement, detailed_block_limit=10
    )

    assert result["changed_blocks"] == 1
    assert result["removed_chars"] == 80
    assert result["added_chars"] == 90
    assert result["before"] == [{"kind": "removed", "text": original}]
    assert result["after"] == [{"kind": "added", "text": replacement}]
