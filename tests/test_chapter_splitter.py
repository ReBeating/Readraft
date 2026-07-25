from app.chapter_splitter import (
    decode_upload,
    joined_text,
    normalize_text,
    split_chapters,
)


def test_detects_numbered_and_special_headings_without_losing_text():
    text = (
        "\ufeff书名：测试\r\n作者：某人\r\n"
        "序章\r\n这是序章。\r\n"
        "第１２章　雨夜\r\n雨下得很大。\r\n"
        "第一百零三回：归来\r\n他终于回来。\r\n"
        "尾声\r\n故事结束。\r\n"
    )
    chunks = split_chapters(text, target_chars=1000, max_chars=2000)

    assert [chunk.title for chunk in chunks] == [
        "书前内容",
        "序章",
        "第12章 雨夜",
        "第一百零三回:归来",
        "尾声",
    ]
    assert joined_text(chunks) == normalize_text(text)


def test_does_not_split_inline_chapter_phrases():
    text = "他翻到第一章。\n这是第十章的内容。\n“第七回？”她问。\n"
    chunks = split_chapters(text, target_chars=1000, max_chars=2000)
    assert len(chunks) == 1
    assert chunks[0].title == "全文"
    assert chunks[0].text == text


def test_splits_long_chapter_at_safe_boundaries_and_preserves_content():
    paragraph = "这是一个用于测试的长段落。" * 30 + "\n\n"
    text = "第一章 开始\n" + paragraph * 12
    chunks = split_chapters(text, target_chars=500, max_chars=700)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 700 for chunk in chunks)
    assert chunks[0].part_number == 1
    assert chunks[-1].part_number == chunks[-1].part_count
    assert joined_text(chunks) == text


def test_merges_volume_heading_with_following_chapter():
    text = "第一卷 风起\n第1章 初见\n正文。\n第2章 再会\n正文。\n"
    chunks = split_chapters(text)
    assert len(chunks) == 2
    assert chunks[0].title == "第一卷 风起 · 第1章 初见"
    assert chunks[0].text.startswith("第一卷 风起\n第1章 初见")
    assert joined_text(chunks) == text


def test_decode_upload_supports_gb18030():
    original = "第一章 开始\r\n中文正文。"
    decoded, encoding = decode_upload(original.encode("gb18030"))
    assert encoding == "gb18030"
    assert decoded == "第一章 开始\n中文正文。"


def test_empty_text_has_no_chunks():
    assert split_chapters("\ufeff \r\n\t") == []


def test_suppresses_dense_table_of_contents_boundaries():
    catalog = (
        "目录\n"
        "第一章 雨夜\n"
        "第二章 徽记\n"
        "第三章 旧案\n"
        "第四章 县衙\n"
        "第五章 真相\n"
        + "\n" * 20
    )
    body = (
        "第一章 雨夜\n" + "雨夜正文。" * 70 + "\n"
        "第二章 徽记\n" + "徽记正文。" * 70 + "\n"
    )
    text = catalog + body
    chunks = split_chapters(text, target_chars=5000, max_chars=10000)

    assert [chunk.title for chunk in chunks] == ["书前内容", "第一章 雨夜", "第二章 徽记"]
    assert joined_text(chunks) == text
