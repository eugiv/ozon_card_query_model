import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from independent_check import (evaluate_predictions, export_check_facts, freeze_predictions,
                               register_check, run_independent_check)


@pytest.fixture
def study():
    protocol = {
        'decision': '2026-10-21', 'frozen_at': '2026-10-20T14:19:17Z',
        'window_days': 14, 'history_days': 14, 'train_end': '2026-09-04',
        'selection_end': '2026-09-18', 'dictionary_as_of': '2026-06-26T23:59:59.999999999Z',
        'catalog_clusters': 2, 'per_source': 5, 'max_views': 5, 'max_seeds': 32,
        'bootstrap_samples': 100, 'bootstrap_seed': 7,
        'weights': {'Полные четыре': {name: .25 for name in
            ['pretrained_embeddings', 'co_sales', 'semantic_adapter', 'lightgcn']},
            'База: E5 + co-sales': {'pretrained_embeddings': .5, 'co_sales': .5}},
    }
    columns = ['date', 'sku', 'query', 'order_count', 'unique_view_users', 'unique_search_users']
    history = pd.DataFrame([[day, 'a', 'seed', 1, 10, 1]
                            for day in pd.date_range('2026-10-08', '2026-10-21')], columns=columns)
    targets = pd.DataFrame([[day, 'a', 'seed', 1, 10, 1]
                            for day in pd.date_range('2026-10-22', '2026-11-04')]
                           + [[pd.Timestamp('2026-10-22'), 'a', query, 1, 1, 1]
                              for query in ['new', 'outside']], columns=columns)
    dictionary = pd.DataFrame({'query': ['seed', 'new', 'outside'],
        'normquery': ['seed', 'new', 'outside'], 'loaded_at': ['2026-06-26T00:00:00Z'] * 3,
        'status_code': [200] * 3, 'error': [''] * 3})
    vectors = np.array([[.8, .6], [1., 0.]], dtype=np.float32)
    frozen = {'catalog': np.array(['new', 'seed']), 'e5': vectors,
              'adapter': vectors.copy(), 'graph': vectors.copy()}
    return protocol, history, targets, dictionary, frozen


def prepare(study, folder):
    protocol, history, _, dictionary, frozen = study
    register_check(protocol, folder, now='2026-10-20T16:00:00Z')
    return freeze_predictions(history, dictionary, frozen, protocol, folder,
                              now='2026-10-22T01:00:00Z')


def test_registration_is_early_idempotent_and_does_not_claim_a_result(study, tmp_path):
    protocol = study[0]
    first = register_check(protocol, tmp_path, now='2026-10-20T16:00:00Z')
    second = register_check(protocol, tmp_path, now='2026-11-06T00:00:00Z')
    assert first == second
    assert first['status'] == 'registered_not_evaluated'
    assert first['registered_at'] == '2026-10-20T16:00:00+00:00'
    assert first['target_start'] == '2026-10-22'
    assert first['target_end'] == '2026-11-04'
    assert 'models' not in first
    assert not (tmp_path / 'predictions').exists()


@pytest.mark.parametrize('now', ['2026-10-22T00:00:00Z', '2026-11-05T00:00:00Z',
                                 '2026-10-20T10:00:00Z', '2026-10-20T16:00:00'])
def test_registration_rejects_late_backdated_or_naive_time(study, tmp_path, now):
    with pytest.raises(ValueError):
        register_check(study[0], tmp_path, now=now)
    assert not (tmp_path / 'registration.json').exists()


def test_registration_rejects_changed_protocol_or_execution_code(study, tmp_path):
    protocol = study[0]
    register_check(protocol, tmp_path, now='2026-10-20T16:00:00Z')
    with pytest.raises(ValueError, match='измен'):
        register_check({**protocol, 'per_source': 12}, tmp_path, now='2026-10-20T17:00:00Z')
    path = tmp_path / 'registration.json'
    registration = json.loads(path.read_text())
    registration['execution_sha256'] = '0' * 64
    path.write_text(json.dumps(registration))
    with pytest.raises(ValueError, match='измен'):
        register_check(protocol, tmp_path, now='2026-10-20T17:00:00Z')


def test_predictions_use_history_only_and_are_never_overwritten(study, tmp_path):
    protocol, history, _, dictionary, frozen = study
    original = history.copy(deep=True)
    manifest = prepare(study, tmp_path)
    assert manifest['status'] == 'predictions_frozen'
    assert manifest['training_performed'] is False
    assert manifest['hyperparameter_selection_performed'] is False
    assert manifest['history_end'] == protocol['decision']
    assert manifest['catalog_clusters'] == 2
    assert not (tmp_path / 'evaluation').exists()
    predictions = pd.read_csv(tmp_path / 'predictions/predictions.csv')
    assert predictions[['sku', 'normquery']].values.tolist() == [['a', 'new']]
    assert {'baseline_score', 'system_score'} <= set(predictions)
    assert 'orders' not in predictions
    pd.testing.assert_frame_equal(history, original)
    with pytest.raises(ValueError, match='перезапись'):
        freeze_predictions(history, dictionary, frozen, protocol, tmp_path,
                           now='2026-10-23T00:00:00Z')


def test_predictions_require_registration_before_any_inference(study, tmp_path):
    protocol, history, _, dictionary, frozen = study
    with pytest.raises(ValueError, match='фиксац'):
        freeze_predictions(history, dictionary, frozen, protocol, tmp_path,
                           now='2026-10-22T01:00:00Z')
    assert not (tmp_path / 'predictions').exists()


@pytest.mark.parametrize('scenario', ['future_rows', 'missing_history', 'open_history_day',
                                      'duplicate_rows', 'negative_orders', 'wrong_catalog'])
def test_predictions_reject_invalid_inputs(study, tmp_path, scenario):
    protocol, history, targets, dictionary, frozen = study
    register_check(protocol, tmp_path, now='2026-10-20T16:00:00Z')
    now = '2026-10-22T01:00:00Z'
    if scenario == 'future_rows':
        history = pd.concat([history, targets.iloc[:1]], ignore_index=True)
    elif scenario == 'missing_history':
        history = history.iloc[1:]
    elif scenario == 'open_history_day':
        now = '2026-10-21T23:59:59Z'
    elif scenario == 'duplicate_rows':
        history = pd.concat([history, history.iloc[:1]], ignore_index=True)
    elif scenario == 'negative_orders':
        history.loc[0, 'order_count'] = -1
    else:
        frozen = {**frozen, 'catalog': frozen['catalog'][::-1]}
    with pytest.raises(ValueError):
        freeze_predictions(history, dictionary, frozen, protocol, tmp_path, now=now)
    assert not (tmp_path / 'predictions').exists()


def test_evaluation_scores_saved_pool_keeps_missing_targets_and_writes_result(study, tmp_path):
    protocol, _, targets, dictionary, _ = study
    prepare(study, tmp_path)
    before = (tmp_path / 'predictions/predictions.csv').read_bytes()
    result = evaluate_predictions(targets, dictionary, protocol, tmp_path,
                                  now='2026-11-05T00:00:00Z')
    assert result['status'] == 'evaluated_frozen_predictions'
    assert result['effect_confirmed'] is False
    assert result['training_performed'] is False
    assert result['hyperparameter_selection_performed'] is False
    assert result['interval']['difference'] == 0
    assert result['interval']['lower'] == result['interval']['upper'] == 0
    assert result['targets_not_retrieved'] == 1
    assert result['normalization']['order_coverage'] == 1
    for metrics in result['models'].values():
        assert metrics['recall@10'] == .5
        assert metrics['positive_pairs'] == 2
        assert metrics['evaluated_skus'] == 1
    assert before == (tmp_path / 'predictions/predictions.csv').read_bytes()
    assert json.loads((tmp_path / 'evaluation/result.json').read_text()) == result
    assert evaluate_predictions(targets, dictionary, protocol, tmp_path,
                                now='2026-11-06T00:00:00Z') == result


@pytest.mark.parametrize('scenario', ['open_last_day', 'missing_day', 'history_rows',
                                      'duplicate_rows', 'nan_orders', 'changed_dictionary'])
def test_evaluation_rejects_incomplete_or_changed_inputs(study, tmp_path, scenario):
    protocol, history, targets, dictionary, _ = study
    prepare(study, tmp_path)
    now = '2026-11-05T00:00:00Z'
    if scenario == 'open_last_day':
        now = '2026-11-04T23:59:59Z'
    elif scenario == 'missing_day':
        targets = targets.loc[targets.date.ne(pd.Timestamp('2026-10-23'))]
    elif scenario == 'history_rows':
        targets = pd.concat([targets, history.iloc[:1]], ignore_index=True)
    elif scenario == 'duplicate_rows':
        targets = pd.concat([targets, targets.iloc[:1]], ignore_index=True)
    elif scenario == 'nan_orders':
        targets['order_count'] = targets.order_count.astype(float)
        targets.loc[0, 'order_count'] = np.nan
    else:
        dictionary.loc[dictionary['query'].eq('new'), 'normquery'] = 'outside'
    with pytest.raises(ValueError):
        evaluate_predictions(targets, dictionary, protocol, tmp_path, now=now)
    assert not (tmp_path / 'evaluation').exists()


def test_evaluation_requires_saved_predictions_and_detects_tampering(study, tmp_path):
    protocol, _, targets, dictionary, _ = study
    register_check(protocol, tmp_path, now='2026-10-20T16:00:00Z')
    with pytest.raises(ValueError, match='рекомендац'):
        evaluate_predictions(targets, dictionary, protocol, tmp_path, now='2026-11-05T00:00:00Z')
    prepare(study, tmp_path)
    path = tmp_path / 'predictions/predictions.csv'
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='измен'):
        evaluate_predictions(targets, dictionary, protocol, tmp_path, now='2026-11-05T00:00:00Z')


def test_evaluation_does_not_replace_result_with_another_sample(study, tmp_path):
    protocol, _, targets, dictionary, _ = study
    prepare(study, tmp_path)
    evaluate_predictions(targets, dictionary, protocol, tmp_path, now='2026-11-05T00:00:00Z')
    targets.loc[targets['query'].eq('new'), 'order_count'] = 0
    with pytest.raises(ValueError, match='результат'):
        evaluate_predictions(targets, dictionary, protocol, tmp_path, now='2026-11-06T00:00:00Z')


def test_no_positive_targets_is_a_result_not_a_fake_zero_quality(study, tmp_path):
    protocol, _, targets, dictionary, _ = study
    prepare(study, tmp_path)
    targets = targets.loc[targets['query'].eq('seed')]
    result = evaluate_predictions(targets, dictionary, protocol, tmp_path, now='2026-11-05T00:00:00Z')
    assert result['status'] == 'no_positive_targets'
    assert result['interval'] is None
    assert result['effect_confirmed'] is False
    assert all(metric['recall@10'] is None for metric in result['models'].values())


def test_staging_preserves_the_existing_frozen_model_metrics(study, tmp_path):
    from notebook_research import evaluate_prospective

    protocol, history, targets, dictionary, frozen = study
    expected = evaluate_prospective(pd.concat([history, targets], ignore_index=True),
                                    dictionary, frozen, protocol)
    prepare(study, tmp_path)
    actual = evaluate_predictions(targets, dictionary, protocol, tmp_path,
                                  now='2026-11-05T00:00:00Z')
    assert actual['models'] == expected['models']
    assert actual['interval']['difference'] == expected['interval']['difference']


def test_interval_resamples_whole_skus_including_unretrieved_targets(study, tmp_path):
    protocol, history, targets, dictionary, _ = study
    queries = [f'q{i:02d}' for i in range(13)]
    catalog = queries + ['seed']
    angles = np.linspace(.05, 1.4, 13)
    vectors = np.vstack([np.column_stack([np.cos(angles), np.sin(angles)]), [1., 0.]]).astype('float32')
    reversed_vectors = np.vstack([vectors[:-1][::-1], [1., 0.]]).astype('float32')
    frozen = {'catalog': np.asarray(catalog), 'e5': vectors,
              'adapter': reversed_vectors, 'graph': reversed_vectors.copy()}
    dictionary = pd.concat([dictionary, pd.DataFrame({
        'query': queries, 'normquery': queries, 'loaded_at': ['2026-06-26T00:00:00Z'] * 13,
        'status_code': [200] * 13, 'error': [''] * 13})], ignore_index=True)
    protocol = {**protocol, 'catalog_clusters': len(catalog), 'per_source': 13}
    history = pd.concat([history.assign(sku=sku) for sku in ['a', 'b', 'c']], ignore_index=True)
    targets = targets.loc[targets['query'].eq('seed')]
    targets = pd.concat([targets, pd.DataFrame([
        {'date': pd.Timestamp('2026-10-22'), 'sku': sku, 'query': query,
         'order_count': 1, 'unique_view_users': 1, 'unique_search_users': 1}
        for sku, query in [('a', 'q12'), ('b', 'outside'), ('c', 'q00')]])], ignore_index=True)
    register_check(protocol, tmp_path, now='2026-10-20T16:00:00Z')
    freeze_predictions(history, dictionary, frozen, protocol, tmp_path, now='2026-10-22T01:00:00Z')
    result = evaluate_predictions(targets, dictionary, protocol, tmp_path, now='2026-11-05T00:00:00Z')
    assert result['models']['Полные четыре']['recall@10'] == pytest.approx(1 / 3)
    assert result['models']['База: E5 + co-sales']['recall@10'] == pytest.approx(1 / 3)
    assert result['interval']['n_skus'] == 3
    differences = np.array([1., 0., -1.])
    indices = np.random.default_rng(7).integers(0, 3, size=(100, 3))
    expected = np.quantile(differences[indices].mean(axis=1), [.025, .975])
    np.testing.assert_allclose([result['interval']['lower'], result['interval']['upper']], expected)


def test_public_runner_refuses_missing_fixed_protocol_before_creating_anything(tmp_path):
    with pytest.raises((ValueError, FileNotFoundError)):
        run_independent_check(tmp_path, tmp_path / 'history.csv', tmp_path / 'targets.csv',
                              tmp_path / 'check', now='2026-10-20T16:00:00Z')
    assert not (tmp_path / 'check').exists()


def test_public_registration_pins_real_protocol_and_execution_without_private_data():
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / 'reports/notebook_full_protocol.json').read_text())
    path = root / 'reports/notebook_independent_registration.json'
    registration = json.loads(path.read_text())
    canonical = json.dumps(protocol, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    assert registration['protocol_sha256'] == hashlib.sha256(canonical).hexdigest()
    assert registration['execution_sha256'] == hashlib.sha256((root / 'src/independent_check.py').read_bytes()).hexdigest()
    assert registration['status'] == 'registered_not_evaluated'
    assert pd.Timestamp(protocol['frozen_at']) <= pd.Timestamp(registration['registered_at'])
    assert pd.Timestamp(registration['registered_at']) < pd.Timestamp('2026-10-22T00:00:00Z')
    assert 'models' not in registration
    assert '/Users/' not in path.read_text()
    private = root / 'data/notebook_independent_check/registration.json'
    if private.is_file():
        assert json.loads(private.read_text()) == registration


@pytest.mark.parametrize('phase', ['history', 'targets'])
def test_export_uses_only_one_closed_window_and_never_reloads_dictionary(study, tmp_path, phase):
    protocol, history, targets, _, _ = study
    calls = []

    class Client:
        def query_df(self, query, parameters):
            calls.append((query, parameters))
            return history if phase == 'history' else targets

    path = tmp_path / 'facts.csv'
    exported = export_check_facts(Client(), protocol, phase, path, now='2026-11-05T00:00:00Z')
    assert len(calls) == 1
    query, parameters = calls[0]
    assert 'WHERE date BETWEEN {start:Date} AND {end:Date}' in query
    assert 'wb_normquery' not in query
    assert str(parameters['start']) == ('2026-10-08' if phase == 'history' else '2026-10-22')
    assert str(parameters['end']) == ('2026-10-21' if phase == 'history' else '2026-11-04')
    assert exported['rows'] == len(pd.read_csv(path))
    with pytest.raises(ValueError, match='перезапись'):
        export_check_facts(Client(), protocol, phase, path, now='2026-11-05T00:00:00Z')
    assert len(calls) == 1


@pytest.mark.parametrize('phase,now', [('history', '2026-10-21T23:59:59Z'),
                                      ('targets', '2026-11-04T23:59:59Z'),
                                      ('wrong', '2026-11-05T00:00:00Z')])
def test_export_does_not_contact_source_for_open_or_invalid_window(study, tmp_path, phase, now):
    with pytest.raises(ValueError):
        export_check_facts(None, study[0], phase, tmp_path / 'facts.csv', now=now)
    assert not (tmp_path / 'facts.csv').exists()


def test_export_does_not_publish_missing_dates_or_expose_connection_secrets(study, tmp_path):
    class Client:
        def query_df(self, query, parameters):
            return study[1].iloc[1:]

    path = tmp_path / 'facts.csv'
    with pytest.raises(ValueError, match='неполный'):
        export_check_facts(Client(), study[0], 'history', path, now='2026-11-05T00:00:00Z')
    assert not path.exists()

    class FailedClient:
        def query_df(self, query, parameters):
            raise OSError('private-password-and-host')

    with pytest.raises(RuntimeError) as error:
        export_check_facts(FailedClient(), study[0], 'history', path, now='2026-11-05T00:00:00Z')
    assert 'private-password-and-host' not in str(error.value)
    assert error.value.__suppress_context__
    assert not path.exists()