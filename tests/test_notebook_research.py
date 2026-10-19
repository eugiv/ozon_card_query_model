import numpy as np
import pandas as pd
import pytest
import hashlib
import json
from pathlib import Path

from notebook_research import (data_overview, evaluate_prospective, profile_scorer,
                               prospective_status, semantic_review_sample, semantic_review_summary, tune_fusion)


def test_eda_missing_calendar_is_not_zero_and_zero_iqr_is_not_removed():
    facts = pd.DataFrame({'date': ['2026-01-01', '2026-01-03'], 'sku': ['a', 'a'],
                          'query': ['Запрос', 'нет'], 'order_count': [0, 3],
                          'unique_view_users': [0, 0], 'unique_search_users': [2, 4]})
    snapshot = pd.DataFrame({'clean_query': ['запрос'], 'normquery': ['запрос']})
    result = data_overview(facts, snapshot)
    assert result['missing_days'] == ['2026-01-02']
    assert result['daily'][1]['orders'] is None
    assert result['row_mapping_pct'] == 50
    assert result['orders_without_views_rows'] == 1
    assert result['rows'] == 2


def test_profiles_use_only_passed_history_and_preserve_cold_zero_signal():
    candidates = pd.DataFrame({'sku': ['a', 'a'], 'normquery': ['x', 'cold']})
    history = pd.DataFrame({'sku': ['a'], 'normquery': ['seed'], 'orders': [2.]})
    vectors = np.array([[1., 0.], [0.5, 0.5], [0., 0.]])
    for weighted in [True, False]:
        scorer = profile_scorer(candidates, history, ['seed', 'x', 'cold'], weighted=weighted)
        np.testing.assert_allclose(scorer(vectors), [.5, 0.])
        with pytest.raises(ValueError):
            scorer(vectors[:1])


def test_semantic_accuracy_never_invented_and_sample_uses_observed_queries():
    facts = pd.DataFrame({'query': ['Known', 'unknown']})
    snapshot = pd.DataFrame({'clean_query': ['known', 'not observed'], 'normquery': ['cluster', 'x']})
    sample = semantic_review_sample(facts, snapshot)
    assert sample.clean_query.tolist() == ['known']
    summary = semantic_review_summary(sample, sample)
    assert summary['accuracy_on_decided'] is None and not summary['complete']
    reviewed = sample.assign(correct='0')
    assert semantic_review_summary(reviewed, sample)['accuracy_on_decided'] == 0
    with pytest.raises(ValueError):
        semantic_review_summary(sample.assign(normquery='changed'), sample)
    with pytest.raises(ValueError):
        semantic_review_summary(sample.assign(correct='maybe'), sample)


def test_grid_selects_validation_and_keeps_all_missing_targets():
    candidates = pd.DataFrame({'sku': ['a', 'a'], 'normquery': ['x', 'y'],
                               'embedding_mean': [1., 0.], 'graph_score': [0., 1.]})
    positives = pd.DataFrame({'sku': ['a', 'b'], 'normquery': ['y', 'missing']})
    result = tune_fusion(candidates, positives, ['pretrained_embeddings', 'co_sales'], units=10)
    assert result['trials'] == 11
    assert result['weights']['co_sales'] > result['weights']['pretrained_embeddings']
    assert result['ndcg@10'] == .5
    with pytest.raises(ValueError):
        tune_fusion(candidates, positives.iloc[:0], ['co_sales'])


def test_four_untrained_channels_include_popularity_without_fake_catboost():
    candidates = pd.DataFrame({'sku': ['a', 'a'], 'normquery': ['x', 'y'],
                               'embedding_mean': [1., 0.], 'graph_score': [0., 1.],
                               'lexical_mean': [1., .1], 'log_cluster_orders': [0., 2.]})
    positives = pd.DataFrame({'sku': ['a'], 'normquery': ['y']})
    result = tune_fusion(candidates, positives,
                         ['pretrained_embeddings', 'co_sales', 'lexical', 'popularity'])
    assert result['trials'] == 35
    assert 'catboost_pu' not in result['weights']
    assert result['ndcg@10'] == 1


def test_prospective_test_rejects_reused_window_and_requires_complete_future():
    protocol = {'decision': '2026-10-16', 'frozen_at': '2026-10-15T00:15:00+03:00', 'window_days': 14}
    result = prospective_status(protocol, pd.date_range('2026-01-01', '2026-10-07'),
                                model_training_end='2026-08-21', selection_end='2026-09-18')
    assert result['status'] == 'awaiting_unseen_data'
    assert len(result['missing_target_days']) == 14
    complete = prospective_status(protocol, pd.date_range('2026-10-03', '2026-10-30'),
                                  model_training_end='2026-08-21', selection_end='2026-09-18')
    assert complete['status'] == 'ready'
    missing_history = prospective_status(protocol, pd.date_range('2026-10-17', '2026-10-30'),
                                         model_training_end='2026-08-21', selection_end='2026-09-18')
    assert missing_history['status'] != 'ready'
    assert len(missing_history['missing_history_days']) == 14
    with pytest.raises(ValueError):
        prospective_status({**protocol, 'decision': '2026-09-18'}, [],
                           model_training_end='2026-08-21', selection_end='2026-09-18')
    with pytest.raises(ValueError):
        prospective_status(protocol, [], model_training_end='2026-10-17', selection_end='2026-09-18')


def test_frozen_future_evaluation_has_no_training_and_keeps_targets_outside_catalog():
    records = [[day, 'a', 'seed', 1, 10, 1] for day in pd.date_range('2026-10-03', '2026-10-30')]
    records += [[pd.Timestamp('2026-10-17'), 'a', query, 1, 1, 1] for query in ['new', 'outside']]
    facts = pd.DataFrame(records, columns=['date', 'sku', 'query', 'order_count',
                                           'unique_view_users', 'unique_search_users'])
    dictionary = pd.DataFrame({'query': ['seed', 'new', 'outside'], 'normquery': ['seed', 'new', 'outside'],
                               'loaded_at': ['2026-06-26T00:00:00Z'] * 3,
                               'status_code': [200] * 3, 'error': [''] * 3})
    vectors = np.array([[.8, .6], [1., 0.]], dtype=np.float32)
    frozen = {'catalog': np.array(['new', 'seed']), 'e5': vectors, 'adapter': vectors, 'graph': vectors}
    protocol = {'decision': '2026-10-16', 'frozen_at': '2026-10-15T00:15:00Z', 'window_days': 14,
                'train_end': '2026-08-21', 'selection_end': '2026-09-18',
                'dictionary_as_of': '2026-06-26T23:59:59.999999999Z', 'per_source': 5, 'max_views': 5,
                'weights': {'Полные четыре': {'semantic_adapter': .5, 'lightgcn': .5},
                            'База: E5 + co-sales': {'pretrained_embeddings': 1.}}}
    original = facts.copy(deep=True)
    result = evaluate_prospective(facts, dictionary, frozen, protocol)
    assert result['status'] == 'evaluated_frozen_protocol'
    assert result['training_performed'] is False
    assert result['hyperparameter_selection_performed'] is False
    for metric in result['models'].values():
        assert metric['positive_pairs'] == 2
        assert metric['recall@10'] == .5
        assert metric['retrieval_recall'] == .5
    pd.testing.assert_frame_equal(facts, original)
    with pytest.raises(ValueError, match='отсортированным'):
        evaluate_prospective(facts, dictionary, {**frozen, 'catalog': frozen['catalog'][::-1]}, protocol)
    with pytest.raises(ValueError, match='не полностью'):
        evaluate_prospective(facts.loc[facts.date.ne('2026-10-20')], dictionary, frozen, protocol)


def test_public_research_has_all_repeats_negative_results_and_frozen_protocol():
    root = Path(__file__).resolve().parents[1]
    report_path = root / 'reports/notebook_research.json'
    protocol_path = root / 'reports/notebook_research_protocol.json'
    report = json.loads(report_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    assert report['protocol_sha256'] == hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    assert report['eda']['rows'] == 3203109
    assert len(report['eda']['missing_days']) == 4
    assert [row['seed'] for row in report['seeds']] == [7, 17, 27, 37, 47]
    assert any(row['Прирост к базе, п.п.'] < 0 for row in report['seeds'])
    assert len(report['training_noise']) == 6
    assert len(report['sensitivity']) == 9
    assert len(report['architecture']) == 5
    test_split = next(part for part in report['study']['splits'] if part['name'] == 'test')
    assert sum(row['SKU'] for row in report['slices']) == test_split['target_skus']
    assert sum(row['Целей'] for row in report['slices']) == test_split['targets']
    for curve in report['curves'].values():
        assert curve['optimizer_steps'] > 0
        assert len(curve['validation_per_epoch']) == curve['epochs_completed'] + 1
        assert curve['validation_per_epoch'][curve['best_epoch']] == max(curve['validation_per_epoch'])
    assert report['semantic_review']['decided'] == 0
    assert report['semantic_review']['accuracy_on_decided'] is None
    assert report['prospective']['status'] == 'awaiting_unseen_data'
    assert protocol['no_tuning_on_future'] is True
    assert protocol['weights'] == report['weights']
    assert protocol['selected_parameters'] == report['selected_parameters']
    for path, digest in protocol['code_sha256'].items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest
    for text in [report_path.read_text(), protocol_path.read_text()]:
        assert '/Users/' not in text and 'DBT_CLICKHOUSE' not in text
        assert 'normquery' not in text and 'clean_query' not in text
