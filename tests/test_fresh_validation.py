import copy

import numpy as np
import pandas as pd
import pytest

from fresh_validation import evaluate_closed_window


@pytest.fixture
def fresh_study():
    columns = ['date', 'sku', 'query', 'unique_search_users', 'unique_view_users', 'order_count']
    rows = [[day, '100', 'history', 10, 1, 1]
            for day in pd.date_range('2026-09-20', '2026-10-03')]
    rows += [[day, '100', 'history', 10, 1, 0]
             for day in pd.date_range('2026-10-04', '2026-10-17')]
    rows += [[pd.Timestamp('2026-10-04'), '100', 'candidate', 10, 1, 1],
             [pd.Timestamp('2026-10-05'), '100', 'outside', 10, 1, 1]]
    facts = pd.DataFrame(rows, columns=columns)
    dictionary = pd.DataFrame({'query': ['history', 'candidate', 'outside'],
        'normquery': ['q1', 'q2', 'q3'], 'status_code': 200, 'error': '',
        'loaded_at': '2026-06-01T00:00:00Z'})
    frozen = {'catalog': np.array(['q1', 'q2']),
              **{name: np.array([[1., 0.], [.8, .6]], dtype=np.float32)
                 for name in ['e5', 'adapter', 'graph']}}
    protocol = {'train_end': '2026-09-04', 'selection_end': '2026-09-18',
        'dictionary_as_of': '2026-06-26T23:59:59.999999999Z', 'catalog_clusters': 2,
        'per_source': 5, 'max_views': 5, 'max_seeds': 32,
        'bootstrap_samples': 40, 'bootstrap_seed': 7,
        'weights': {'Полные четыре': {name: .25 for name in
            ['pretrained_embeddings', 'co_sales', 'semantic_adapter', 'lightgcn']},
            'База: E5 + co-sales': {'pretrained_embeddings': .5, 'co_sales': .5}}}
    return facts, dictionary, frozen, protocol


def test_full_window_keeps_targets_outside_catalog_and_does_not_train(fresh_study):
    facts, dictionary, frozen, protocol = fresh_study
    original_facts = facts.copy(deep=True)
    original_vectors = copy.deepcopy(frozen)
    result = evaluate_closed_window(facts, dictionary, frozen, protocol,
                                    previously_observed_through='2026-10-07')
    assert result['history_start'] == '2026-09-20' and result['decision'] == '2026-10-03'
    assert result['target_start'] == '2026-10-04' and result['target_end'] == '2026-10-17'
    assert result['target_days'] == 14 and result['new_source_target_days'] == 10
    assert not result['independent_test']
    assert not result['training_performed'] and not result['hyperparameter_selection_performed']
    assert result['targets_not_retrieved'] == 1
    for name in ['База: E5 + co-sales', 'Полные четыре']:
        assert result['models'][name]['positive_pairs'] == 2
        assert result['models'][name]['recall@10'] == .5
    assert result['interval']['difference'] == 0
    assert result['interval']['lower'] == 0 and result['interval']['upper'] == 0
    assert result['channel_count_interval']['n_targets'] == 2
    assert result['channel_count_interval']['difference'] == (
        result['models']['Полные четыре']['recall@10'] - result['models']['Четыре без обучения']['recall@10'])
    assert sum(row['n_targets'] for row in result['slices']) == 2
    assert sum(row['n_skus'] for row in result['slices']) == 1
    pd.testing.assert_frame_equal(facts, original_facts)
    for name in frozen:
        np.testing.assert_array_equal(frozen[name], original_vectors[name])


def test_future_orders_do_not_change_candidates_scores_or_catalog(fresh_study):
    facts, dictionary, frozen, protocol = fresh_study
    first = evaluate_closed_window(facts, dictionary, frozen, protocol,
                                   previously_observed_through='2026-10-07')
    altered = facts.copy()
    altered.loc[altered.date.gt('2026-10-03'), 'order_count'] = 999
    second = evaluate_closed_window(altered, dictionary, frozen, protocol,
                                    previously_observed_through='2026-10-07')
    assert first['prediction_sha256'] == second['prediction_sha256']
    assert first['candidate_pairs'] == second['candidate_pairs']
    assert first['catalog_clusters'] == second['catalog_clusters'] == 2


@pytest.mark.parametrize('day', ['2026-09-20', '2026-10-10'])
def test_missing_history_or_target_day_blocks_evaluation(fresh_study, day):
    facts, dictionary, frozen, protocol = fresh_study
    with pytest.raises(ValueError, match='полные 14'):
        evaluate_closed_window(facts.loc[facts.date.ne(day)], dictionary, frozen, protocol,
                               previously_observed_through='2026-10-07', cutoff='2026-10-17')


def test_no_positive_targets_is_not_reported_as_zero_quality_or_an_effect(fresh_study):
    facts, dictionary, frozen, protocol = fresh_study
    facts.loc[facts.date.gt('2026-10-03'), 'order_count'] = 0
    result = evaluate_closed_window(facts, dictionary, frozen, protocol,
                                    previously_observed_through='2026-10-07')
    assert result['status'] == 'no_positive_targets'
    assert result['interval']['n_targets'] == 0
    assert result['interval']['difference'] is None
    assert result['effect_confirmed_on_window'] is False


@pytest.mark.parametrize('changed', ['weights', 'train_end', 'selection_end', 'vectors', 'catalog'])
def test_invalid_or_mismatched_frozen_model_is_rejected(fresh_study, changed):
    facts, dictionary, frozen, protocol = fresh_study
    if changed == 'weights':
        protocol['weights']['Полные четыре']['co_sales'] = 0
    elif changed in {'train_end', 'selection_end'}:
        protocol[changed] = '2026-10-10'
    elif changed == 'vectors':
        frozen['adapter'][0, 0] = np.nan
    else:
        frozen['catalog'] = np.array(['q1', 'q1'])
    with pytest.raises(ValueError):
        evaluate_closed_window(facts, dictionary, frozen, protocol,
                               previously_observed_through='2026-10-07')


def test_explicit_cutoff_never_silently_accepts_later_target_records(fresh_study):
    facts, dictionary, frozen, protocol = fresh_study
    with pytest.raises(ValueError, match='даты после'):
        evaluate_closed_window(facts, dictionary, frozen, protocol,
                               previously_observed_through='2026-10-07', cutoff='2026-10-16')