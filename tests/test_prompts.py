from adviceline.prompts import build_items, load_sources, turns, validate


def test_design_has_300_items():
    items = build_items(load_sources())
    validate(items)
    assert len(items) == 300
    assert items["item_id"].is_unique


def test_each_question_has_four_detail_levels_and_one_conversation():
    items = build_items(load_sources())
    counts = items[items["kind"] == "single"].groupby("question_id")["detail_level"].nunique()
    assert (counts == 4).all()
    assert (items[items["kind"] == "multi"].groupby("question_id").size() == 1).all()


def test_detail_levels_add_information():
    items = build_items(load_sources()).set_index("item_id")
    d0, d1, d2, d3 = (items.loc[f"S-sv-r1-D{i}", "turn1"] for i in range(4))
    assert len(d0) < len(d1) < len(d2) < len(d3)
    assert d3.endswith(d0)  # the question itself comes last
    assert items.loc["S-sv-r1-D3", "vuln_id"].startswith("V")


def test_multi_turn_items_have_three_turns():
    items = build_items(load_sources())
    multi = items[items["kind"] == "multi"].to_dict("records")
    assert all(len(turns(it)) == 3 for it in multi)
    single = items[items["kind"] == "single"].to_dict("records")
    assert all(len(turns(it)) == 1 for it in single)
