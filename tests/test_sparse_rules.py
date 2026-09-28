from sparse_rules import decide_sparse_card


def test_keeps_past_order():
    assert decide_sparse_card("крем", "крем для ног", "свобода", 2) == "keep"


def test_drops_single_generic_word():
    assert decide_sparse_card("крем", "крем для ног с мочевиной", "свобода", 0) == "drop"


def test_keeps_brand_phrase():
    assert decide_sparse_card("невская косметика мыло", "мыло дегтярное", "невская косметика", 0) == "keep"


def test_keeps_long_overlap():
    assert (
        decide_sparse_card(
            "крем для ног мочевина",
            "крем для ног с мочевиной",
            "",
            0,
        )
        == "keep"
    )


def test_reviews_partial_phrase():
    assert decide_sparse_card("крем для лица", "крем для ног с мочевиной", "", 0) == "review"
