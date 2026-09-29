from sparse_rules import decide_sparse_card


def test_keeps_past_order():
    assert (
        decide_sparse_card("крем", orders_past=2, freq_past=4, views_past=2) == "keep"
    )


def test_reviews_single_generic_word_without_order():
    assert (
        decide_sparse_card("крем", orders_past=0, freq_past=500, views_past=40)
        == "review"
    )


def test_reviews_brand_phrase_without_historical_text():
    assert (
        decide_sparse_card(
            "невская косметика мыло",
            orders_past=0,
            freq_past=100,
            views_past=10,
        )
        == "review"
    )


def test_reviews_long_query_without_order():
    assert (
        decide_sparse_card(
            "крем для ног мочевина",
            orders_past=0,
            freq_past=100,
            views_past=10,
        )
        == "review"
    )


def test_reviews_empty_phrase():
    assert decide_sparse_card("", orders_past=0, freq_past=0, views_past=0) == "review"
