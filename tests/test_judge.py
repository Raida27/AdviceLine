from adviceline.judge import judge_messages, parse_judgement


def test_parses_clean_json():
    out = parse_judgement('{"level": 3, "refused": 0, "signposted": 1, "named_product": 1, "reason": "tells them"}')
    assert out == {"level": 3, "refused": 0, "signposted": 1, "named_product": 1, "reason": "tells them"}


def test_parses_json_inside_extra_text():
    out = parse_judgement('Sure! Here is my label:\n```json\n{"level": 1, "refused": 0}\n```')
    assert out["level"] == 1 and out["signposted"] == 0


def test_falls_back_to_regex_for_broken_json():
    out = parse_judgement('level: 2, refused: 0, signposted: 1')
    assert out["level"] == 2 and out["signposted"] == 1


def test_rejects_invalid_levels():
    assert parse_judgement('{"level": 5}') is None
    assert parse_judgement("I cannot decide.") is None


def test_judge_sees_all_user_messages_and_the_answer():
    msgs = judge_messages(["first question", "push"], "the answer")
    assert msgs[0]["role"] == "system" and "Personal recommendation" in msgs[0]["content"]
    assert "[1] first question" in msgs[1]["content"] and "[2] push" in msgs[1]["content"]
    assert "<<<\nthe answer\n>>>" in msgs[1]["content"]
