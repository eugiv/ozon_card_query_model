import copy

import numpy as np
import pandas as pd
import pytest

from research_diagnostics import (paired_recall_intervals, perturb_history,
                                   random_configurations, summarize_search)


@pytest.fixture
def history():
    frame = pd.DataFrame({'sku': ['a', 'a', 'b', 'b', 'c', 'c'],
                          'normquery': ['x', 'z', 'y', 'z', 'x', 'y'],
                          'orders': [3., 0., 2., 0., 4., 1.], 'views': [9, 2, 3, 4, 5, 6]})
    frame.attrs['train_end'] = '2026-08-21'
    return frame


@pytest.mark.parametrize('scenario,removed,added', [('drop_edges', 1, 0), ('false_edges', 0, 1),
                                                   ('rewire_edges', 1, 1)])
def test_noise_changes_binary_edges_not_only_counts(history, scenario, removed, added):
    original = history.copy(deep=True)
    damaged, audit = perturb_history(history, scenario, seed=7, fraction=.25)
    before = set(map(tuple, history.loc[history.orders.gt(0), ['sku', 'normquery']].to_numpy()))
    after = set(map(tuple, damaged.loc[damaged.orders.gt(0), ['sku', 'normquery']].to_numpy()))
    assert len(before - after) == audit['removed_edges'] == removed
    assert len(after - before) == audit['added_edges'] == added
    assert audit['edges_before'] == 4 and audit['edges_after'] == 4 - removed + added
    assert audit['changed_fraction'] == (removed + added) / 4
    assert set(damaged.sku) <= set(history.sku)
    assert set(damaged.normquery) <= set(history.normquery)
    assert not damaged.duplicated(['sku', 'normquery']).any()
    pd.testing.assert_frame_equal(history, original)
    assert damaged.attrs == history.attrs
    pd.testing.assert_frame_equal(damaged, perturb_history(history, scenario, seed=7, fraction=.25)[0])


def test_false_edges_leave_unselected_zeros_unchanged_and_add_missing_pairs(history):
    damaged, audit = perturb_history(history, 'false_edges', fraction=1)
    assert audit['added_edges'] == 4
    assert len(damaged) >= len(history)
    joined = history.merge(damaged, on=['sku', 'normquery'], suffixes=('_old', '_new'))
    np.testing.assert_array_equal(joined.views_old, joined.views_new)
    unchanged = joined.orders_old.gt(0)
    np.testing.assert_array_equal(joined.loc[unchanged, 'orders_old'], joined.loc[unchanged, 'orders_new'])


@pytest.mark.parametrize('fraction', [0, -1, 1.1, float('nan'), float('inf'), True])
def test_noise_rejects_invalid_fraction(history, fraction):
    with pytest.raises(ValueError):
        perturb_history(history, 'drop_edges', fraction=fraction)


def test_noise_rejects_duplicate_keys_invalid_orders_and_impossible_insertions(history):
    with pytest.raises(ValueError):
        perturb_history(pd.concat([history, history.iloc[:1]]), 'drop_edges')
    with pytest.raises(ValueError):
        perturb_history(history.assign(orders=-1), 'drop_edges')
    complete = pd.DataFrame({'sku': ['a', 'a'], 'normquery': ['x', 'y'], 'orders': [1, 1]})
    with pytest.raises(ValueError):
        perturb_history(complete, 'false_edges')
    with pytest.raises(ValueError):
        perturb_history(history, 'not a scenario')


def test_bootstrap_is_paired_by_sku_and_keeps_missing_targets():
    candidates = pd.DataFrame({'sku': ['a', 'a', 'a', 'a'], 'normquery': ['x', 'q1', 'q2', 'q3']})
    positives = pd.DataFrame({'sku': ['a', 'b'], 'normquery': ['x', 'outside']})
    result = paired_recall_intervals(candidates, positives, [4, 3, 2, 1], [4, 3, 2, 1], seed=9)
    assert result['n_skus'] == 2 and result['n_targets'] == 2
    assert result['system'] == result['baseline'] == .5
    assert result['difference'] == result['lower'] == result['upper'] == 0
    assert result['baseline_lower'] == 0 and result['baseline_upper'] == 1
    assert result == paired_recall_intervals(candidates, positives, [4, 3, 2, 1], [4, 3, 2, 1], seed=9)


def test_bootstrap_uses_macro_recall_and_lexicographic_ties():
    candidates = pd.DataFrame({'sku': ['a'] * 11, 'normquery': [f'q{i:02d}' for i in range(11)]})
    positives = pd.DataFrame({'sku': ['a', 'a', 'b'], 'normquery': ['q00', 'q10', 'missing']})
    result = paired_recall_intervals(candidates, positives, np.ones(11), np.arange(11))
    assert result['system'] == .25 and result['baseline'] == .25
    assert result['difference'] == 0
    empty = paired_recall_intervals(candidates, positives.iloc[:0], np.ones(11), np.ones(11))
    assert empty['n_skus'] == empty['n_targets'] == 0
    assert empty['system'] is None and empty['lower'] is None


@pytest.mark.parametrize('scores', [[1], [float('nan'), 1], [[1], [2]]])
def test_bootstrap_rejects_invalid_scores(scores):
    candidates = pd.DataFrame({'sku': ['a', 'a'], 'normquery': ['x', 'y']})
    with pytest.raises(ValueError):
        paired_recall_intervals(candidates, candidates.iloc[:1], scores, [1, 2])


def test_random_search_draws_unique_joint_reproducible_configurations():
    space = {'hidden_dim': [64, 128], 'batch_size': [64, 128, 256]}
    rows = random_configurations(space, 6)
    assert len({tuple(row.items()) for row in rows}) == 6
    assert rows == random_configurations(space, 6)
    assert rows != random_configurations(space, 6, seed=42)
    with pytest.raises(ValueError):
        random_configurations(space, 7)
    with pytest.raises(ValueError):
        random_configurations({'x': []}, 1)


def search_rows():
    return [{'model': 'adapter', 'parameters': {'hidden_dim': size, 'batch_size': 128},
             'seed': seed, 'validation_ndcg': value, 'validation_recall': value, 'seconds': 2.}
            for size, values in [(64, [.10, .12, .14]), (128, [.11, .13, .15])]
            for seed, value in zip([7, 17, 27], values, strict=True)]


def test_search_compares_means_and_std_instead_of_best_seed():
    rows = search_rows()
    original = copy.deepcopy(rows)
    result = summarize_search(rows)
    assert result['selected']['adapter']['hidden_dim'] == 64
    assert all(row['equivalent_to_best'] for row in result['configurations'])
    assert sum(row['selected'] for row in result['configurations']) == 1
    smaller = next(row for row in result['configurations'] if row['parameters']['hidden_dim'] == 64)
    assert smaller['mean_ndcg'] == pytest.approx(.12)
    assert smaller['std_ndcg'] == pytest.approx(.02)
    assert rows == original
    assert summarize_search(rows[::-1])['selected'] == result['selected']


def test_search_does_not_discard_clear_difference():
    rows = search_rows()
    for row in rows:
        if row['parameters']['hidden_dim'] == 128:
            row['validation_ndcg'] += .3
    assert summarize_search(rows)['selected']['adapter']['hidden_dim'] == 128


def test_search_requires_complete_identical_seed_sets_and_finite_metrics():
    rows = search_rows()
    for invalid in [rows[:-1], rows + rows[:1], [{**rows[0], 'validation_ndcg': float('nan')}, *rows[1:]]]:
        with pytest.raises(ValueError):
            summarize_search(invalid)