from app.json_support import (
    dump_canonical_json,
    dump_json,
    json_fingerprint,
    load_json,
    load_json_dict,
    load_json_list,
)


def test_compact_json_preserves_chinese_and_insertion_order():
    assert dump_json({"乙": 2, "甲": 1}) == '{"乙":2,"甲":1}'


def test_canonical_json_and_fingerprint_ignore_mapping_order():
    left = {"乙": [2], "甲": 1}
    right = {"甲": 1, "乙": [2]}

    assert dump_canonical_json(left) == '{"乙":[2],"甲":1}'
    assert json_fingerprint(left) == json_fingerprint(right)


def test_load_json_returns_exact_fallback_for_invalid_values():
    fallback: list[str] = []

    assert load_json('{"有效":true}', fallback) == {"有效": True}
    assert load_json("", fallback) is fallback
    assert load_json(None, fallback) is fallback


def test_typed_loaders_reject_valid_json_with_the_wrong_shape():
    assert load_json_dict('{"名称":"雾港"}') == {"名称": "雾港"}
    assert load_json_dict("[]") == {}
    assert load_json_list('["线索"]') == ["线索"]
    assert load_json_list("{}") == []
