from pathlib import Path

import pytest

from app.chapter_splitter import split_chapters
from app.import_preview import ImportPreviewStore


def make_preview(tmp_path: Path):
    text = "第一章 来信\n雨夜收到旧信。\n第二章 回声\n寄信人早已失踪。\n"
    store = ImportPreviewStore(tmp_path / "previews")
    preview = store.create(
        user_id=1,
        title="雨夜来信",
        original_filename="rain.txt",
        source_encoding="utf-8",
        source_text=text,
        chunks=split_chapters(text),
    )
    return store, preview


def test_preview_keeps_immutable_source_and_exact_boundaries(tmp_path: Path):
    store, preview = make_preview(tmp_path)

    loaded = store.get(user_id=1, preview_id=preview.id)

    assert loaded is not None
    assert loaded.source_text == preview.source_text
    assert "".join(chunk.text for chunk in store.to_chunks(loaded)) == preview.source_text
    assert all(item.reason for item in loaded.boundaries)


def test_preview_can_rename_merge_and_split_without_changing_text(tmp_path: Path):
    store, preview = make_preview(tmp_path)
    original = preview.source_text

    renamed = store.rename(
        user_id=1, preview_id=preview.id, index=0, title="第一章 · 雨中信"
    )
    assert renamed.boundaries[0].title == "第一章 · 雨中信"
    merged = store.merge(user_id=1, preview_id=preview.id, index=0)
    assert len(merged.boundaries) == 1
    cut = original.index("第二章")
    split = store.split(
        user_id=1,
        preview_id=preview.id,
        index=0,
        source_offset=cut,
    )

    assert len(split.boundaries) == 2
    assert "".join(chunk.text for chunk in store.to_chunks(split)) == original
    assert split.boundaries[0].end == split.boundaries[1].start == cut


def test_preview_is_user_scoped_and_rejects_invalid_split(tmp_path: Path):
    store, preview = make_preview(tmp_path)

    assert store.get(user_id=2, preview_id=preview.id) is None
    with pytest.raises(ValueError, match="正文内部"):
        store.split(
            user_id=1,
            preview_id=preview.id,
            index=0,
            source_offset=preview.boundaries[0].start,
        )
