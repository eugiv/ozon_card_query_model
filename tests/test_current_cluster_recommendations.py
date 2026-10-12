import pandas as pd
import pytest

from seo_recommendation_policy import review_top_k


def test_presence_and_conflicts_are_removed_before_top_k_budget():
    frame = pd.DataFrame([
        ["1", "present", 10., "already_present"], ["1", "foreign", 9., "exclude_candidate"],
        ["1", "z", 1., "review"], ["1", "a", 1., "review"], ["2", "only", 0., "review"],
    ], columns=["sku", "normquery", "score", "decision"])
    before = frame.copy()
    result = review_top_k(frame, 1)
    assert result[["sku", "normquery", "rank"]].values.tolist() == [["1", "a", 1], ["2", "only", 1]]
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("limit", [True, 0, -1, 1.5])
def test_k_validation(limit):
    with pytest.raises(ValueError):
        review_top_k(pd.DataFrame(), limit)