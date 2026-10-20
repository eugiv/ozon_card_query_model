import json

import numpy as np
import pandas as pd
import pytest

from cluster_data import sha256
from retrospective_holdout import (
    TemporalHoldoutStudy, default_protocol, evaluate_holdout, fit_predict_holdout,
    split_holdout, verify_packet,
)


@pytest.fixture
def small_holdout():
    protocol = default_protocol()
    protocol.update(seeds=[7, 17, 27], search_budget=1, epochs=2, patience=1,
                    per_source=4, bootstrap_samples=40)
    queries = [f'q{i:02}' for i in range(16)]
    dictionary = pd.DataFrame({
        'query': queries, 'normquery': queries, 'status_code': 200, 'error': '',
        'loaded_at': '2026-06-26T10:00:00Z',
    })
    rows = []
    for day in pd.date_range('2026-07-25', '2026-09-18'):
        for i in range(4):
            rows.append([day, str(i), queries[2 * i], 10, 1, 1])
            rows.append([day, str(i), queries[2 * i + 1], 10, 1, 1])
    for day, offset in [('2026-08-08', 8), ('2026-08-22', 9), ('2026-09-05', 10)]:
        for i in range(4):
            rows.append([pd.Timestamp(day), str(i), queries[offset + i], 10, 1, 1])
    facts = pd.DataFrame(rows, columns=[
        'date', 'sku', 'query', 'unique_search_users', 'unique_view_users', 'order_count',
    ])
    vectors = np.random.default_rng(77).normal(size=(len(queries), 384)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return facts, dictionary, queries, vectors, protocol


def test_protocol_keeps_all_selection_windows_before_heldout_period():
    protocol = default_protocol()
    assert protocol['train_end'] == '2026-08-07'
    assert protocol['epoch_validation_end'] == '2026-08-21'
    assert protocol['decision'] == '2026-09-04'
    assert protocol['target_start'] == '2026-09-05'
    assert protocol['target_end'] == '2026-09-18'
    assert protocol['seeds'] == [7, 17, 27, 37, 47]
    assert protocol['independent_of_research_history'] is False
    assert protocol['old_trained_weights_used'] is False


def test_split_excludes_holdout_and_every_later_row_from_past(small_holdout):
    facts, _, _, _, protocol = small_holdout
    original = facts.copy(deep=True)
    later = facts.iloc[:1].assign(date=pd.Timestamp('2026-10-17'), order_count=999)
    past, heldout, audit = split_holdout(pd.concat([facts, later]), protocol)
    assert past.date.max() == pd.Timestamp('2026-09-04')
    assert heldout.date.min() == pd.Timestamp('2026-09-05')
    assert heldout.date.max() == pd.Timestamp('2026-09-18')
    assert set(past.date).isdisjoint(heldout.date)
    assert audit['heldout_days'] == 14 and audit['ignored_later_rows'] == 1
    assert audit['training_last_day'] == '2026-08-07'
    pd.testing.assert_frame_equal(facts, original)


@pytest.mark.parametrize('day', ['2026-08-01', '2026-08-15', '2026-08-29', '2026-09-12'])
def test_incomplete_history_validation_or_test_is_not_evaluated(small_holdout, day):
    facts, _, _, _, protocol = small_holdout
    with pytest.raises(ValueError, match='полные 14'):
        split_holdout(facts.loc[facts.date.ne(day)], protocol)


@pytest.mark.parametrize('field,value', [
    ('train_end', '2026-09-06'), ('epoch_validation_end', '2026-08-22'),
    ('target_start', '2026-09-04'), ('target_end', '2026-09-17'),
    ('dictionary_as_of', '2026-09-05T00:00:00Z'), ('seeds', [7, 7, 17]),
    ('seeds', [7, 17]), ('search_budget', 0), ('bootstrap_samples', True),
])
def test_invalid_protocol_fails_before_training(small_holdout, field, value):
    facts, _, _, _, protocol = small_holdout
    protocol[field] = value
    with pytest.raises(ValueError):
        split_holdout(facts, protocol)


def test_training_api_rejects_heldout_rows_and_catalog_from_future(small_holdout):
    facts, dictionary, catalog, vectors, protocol = small_holdout
    with pytest.raises(ValueError, match='отложенн'):
        TemporalHoldoutStudy(facts, dictionary, catalog, vectors, protocol)
    past, _, _ = split_holdout(facts, protocol)
    changed = dictionary.copy()
    changed.loc[changed.normquery.eq(catalog[-1]), 'loaded_at'] = '2026-09-06T00:00:00Z'
    with pytest.raises(ValueError, match='каталог'):
        TemporalHoldoutStudy(past, changed, catalog, vectors, protocol)


@pytest.fixture
def trained_packet(small_holdout, tmp_path):
    facts, dictionary, catalog, vectors, protocol = small_holdout
    past, heldout, audit = split_holdout(facts, protocol)
    folder = tmp_path / 'packet'
    fit_predict_holdout(past, dictionary, catalog, vectors, protocol, folder)
    return folder, past, heldout, dictionary, protocol, audit


def test_real_models_train_from_scratch_and_predictions_precede_labels(trained_packet):
    folder, past, heldout, dictionary, protocol, _ = trained_packet
    manifest = verify_packet(folder, protocol)
    assert manifest['training_performed'] is True
    assert manifest['old_trained_weights_used'] is False
    assert manifest['targets_read_during_training'] is False
    assert manifest['seeds'] == protocol['seeds']
    assert manifest['training']['last_day'] == protocol['train_end']
    assert manifest['selection']['last_day'] == protocol['decision']
    assert manifest['catalog_clusters'] == 16
    assert len(manifest['search']['configurations']) == 2
    for row in manifest['training_runs']:
        audit = row['audit']
        assert audit['validation_per_epoch']
        assert audit['best_epoch'] == int(np.argmax(audit['validation_per_epoch']))
        assert audit['best_validation_score'] == max(audit['validation_per_epoch'])
    result = evaluate_holdout(folder, heldout, dictionary, protocol)
    assert result['status'] == 'evaluated_retrospective_holdout'
    assert result['training_performed'] and result['test_excluded_from_training']
    assert not result['independent_of_research_history']
    assert len(result['seeds']) == 3
    assert result['interval']['n_targets'] == 4
    assert sum(row['n_targets'] for row in result['slices']) == 4
    assert result['channel_count_interval']['n_targets'] == 4
    assert result['prediction_packet_sha256'] == sha256(folder / 'manifest.json')
    assert result['seed_summary']['count'] == 3
    assert set(result['ablation']) >= {'База: E5 + co-sales', 'Полные четыре', 'Четыре без обучения'}
    public = json.dumps(result, ensure_ascii=False, allow_nan=False)
    assert str(folder) not in public
    assert 'q00' not in public
    assert len(past) > len(heldout)


def test_future_changes_cannot_change_fit_selection_or_predictions(small_holdout, tmp_path):
    facts, dictionary, catalog, vectors, protocol = small_holdout
    past, heldout, _ = split_holdout(facts, protocol)
    changed = facts.copy()
    changed.loc[changed.date.gt(protocol['decision']), 'order_count'] = 999
    changed = pd.concat([changed, pd.DataFrame([{
        'date': pd.Timestamp('2026-09-07'), 'sku': '0', 'query': 'future_only_query',
        'unique_search_users': 999, 'unique_view_users': 0, 'order_count': 999,
    }])], ignore_index=True)
    second_past, second_targets, _ = split_holdout(changed, protocol)
    pd.testing.assert_frame_equal(past, second_past)
    first, second = tmp_path / 'first', tmp_path / 'second'
    fit_predict_holdout(past, dictionary, catalog, vectors, protocol, first)
    fit_predict_holdout(second_past, dictionary, catalog, vectors, protocol, second)
    one, two = verify_packet(first, protocol), verify_packet(second, protocol)
    assert one['search']['selected'] == two['search']['selected']
    for seed in protocol['seeds']:
        assert sha256(first / f'predictions_{seed}.csv') == sha256(second / f'predictions_{seed}.csv')
    assert not second_targets.order_count.equals(heldout.order_count)


def test_corrupt_predictions_cannot_be_scored(trained_packet):
    folder, _, heldout, dictionary, protocol, _ = trained_packet
    path = folder / 'predictions_7.csv'
    path.write_bytes(path.read_bytes() + b'corrupt')
    with pytest.raises(ValueError, match='сумм'):
        evaluate_holdout(folder, heldout, dictionary, protocol)


def test_target_dates_and_empty_targets_are_checked(trained_packet):
    folder, _, heldout, dictionary, protocol, _ = trained_packet
    with pytest.raises(ValueError, match='целев'):
        evaluate_holdout(folder, heldout.assign(date=heldout.date + pd.Timedelta(days=28)), dictionary, protocol)
    empty = heldout.assign(order_count=0)
    result = evaluate_holdout(folder, empty, dictionary, protocol)
    assert result['status'] == 'no_positive_targets'
    assert result['interval']['difference'] is None
    assert result['effect_confirmed_on_window'] is False


def test_subcatalog_scoring_matches_full_encoded_scores(small_holdout, trained_packet):
    from dormant_learning import ranking_metrics
    from graph_learning import load_graph
    from notebook_research import profile_scorer
    from semantic_adapter import apply_adapter, load_adapter

    facts, dictionary, catalog, vectors, protocol = small_holdout
    folder, past, _, _, _, _ = trained_packet
    study = TemporalHoldoutStudy(past, dictionary, catalog, vectors, protocol)
    manifest = verify_packet(folder, protocol)
    part = study.parts['validation']
    for kind in ['adapter', 'graph']:
        relative = manifest['selected_models']['7'][kind]
        model = load_adapter(folder / relative) if kind == 'adapter' else load_graph(folder / relative)
        encoded = apply_adapter(model, vectors) if kind == 'adapter' else model.query_vectors(catalog)[0]
        expected = profile_scorer(part['base'], part['history'], catalog, weighted=kind == 'adapter')(encoded)
        actual = study.score(kind, model, part['base'], part['history'])
        np.testing.assert_allclose(actual, expected, atol=2e-6)
        epoch_part = study.parts['epoch_validation']
        epoch_scores = study.score(kind, model, epoch_part['base'], epoch_part['history'])
        restored = ranking_metrics(epoch_part['base'][['sku', 'normquery']].assign(score=epoch_scores),
                                   epoch_part['targets'])['ndcg@10']
        selected_audit = next(row['audit'] for row in manifest['training_runs']
                              if row['seed'] == 7 and row['model'] == kind
                              and row['parameters'] == manifest['search']['selected'][kind])
        assert restored == pytest.approx(selected_audit['best_validation_score'])


def test_existing_packet_is_never_overwritten(trained_packet, small_holdout):
    folder, past, _, dictionary, protocol, _ = trained_packet
    _, _, catalog, vectors, _ = small_holdout
    digest = sha256(folder / 'manifest.json')
    with pytest.raises(ValueError, match='существует'):
        fit_predict_holdout(past, dictionary, catalog, vectors, protocol, folder)
    assert sha256(folder / 'manifest.json') == digest