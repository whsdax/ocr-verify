"""评测指标单元测试。"""

from ocr_verify.metrics import (
    char_accuracy,
    char_error_rate,
    evaluate_sample,
    levenshtein,
    normalize_text,
    percentile,
)


def test_levenshtein():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3
    assert levenshtein("abc", "abc") == 0


def test_cer():
    # "登录成功" -> "登录失败" 需要把 "成" 替换为 "失","功" 替换为 "败",距离为 2
    assert char_error_rate("登录成功", "登录失败") == 2 / 4
    assert char_error_rate("", "登录成功") == 1.0
    assert char_error_rate("", "") == 0.0


def test_char_accuracy():
    assert char_accuracy("登录成功", "登录失败") == 0.5


def test_normalize_text():
    assert normalize_text("确定 ，取消") == "确定,取消"
    assert normalize_text("  确定   取消  ") == "确定取消"


def test_evaluate_sample():
    ev = evaluate_sample(
        sample_id="s1",
        prediction="登录成功",
        ground_truth="登录成功",
        confidence=0.9,
    )
    assert ev.exact_match
    assert ev.char_acc == 1.0


def test_evaluate_sample_partial():
    ev = evaluate_sample(
        sample_id="s1",
        prediction="登录成功了吗",
        ground_truth="登录成功",
    )
    assert not ev.exact_match
    assert ev.contains_match  # ground_truth 是 prediction 的子串


def test_percentile():
    assert percentile([1, 2, 3, 4, 5], 0.5) == 3
    assert percentile([1, 2, 3, 4, 5], 0.0) == 1
    assert percentile([1, 2, 3, 4, 5], 1.0) == 5
    assert percentile([], 0.5) == 0.0
