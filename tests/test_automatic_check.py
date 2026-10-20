import json
import logging
import sys

import pandas as pd
import pytest

import automatic_check
from cluster_data import sha256


@pytest.fixture
def project(tmp_path, monkeypatch):
    reports = tmp_path / 'reports'
    reports.mkdir()
    data = tmp_path / 'data'
    data.mkdir()
    for filename in ['facts.csv', 'dictionary.csv', 'vectors.npz']:
        (data / filename).write_text('fixed input')
    protocol = {'decision': '2026-10-21', 'window_days': 14,
        'inputs': {'facts': {'path': 'data/facts.csv'},
                   'dictionary': {'path': 'data/dictionary.csv'}},
        'artifacts': {'e5': {'path': 'data/vectors.npz'}}}
    protocol_path = reports / 'notebook_full_protocol.json'
    protocol_path.write_text(json.dumps(protocol))
    (reports / 'notebook_review.json').write_text(json.dumps(
        {'full_protocol_sha256': sha256(protocol_path)}))
    events = []

    def verify(root, saved):
        assert root == tmp_path and saved == protocol
        events.append('verify')

    def run(root, history, targets, folder, *, now, check_only=False):
        assert root == tmp_path
        events.append('preflight' if check_only else 'run')
        packet = folder / 'predictions'
        if history.is_file() and not check_only:
            packet.mkdir(parents=True, exist_ok=True)
            events.append('predictions_saved')
        if packet.is_dir() and targets.is_file():
            return {'status': 'ready_to_evaluate'} if check_only else {
                'status': 'evaluated_frozen_predictions', 'models': {},
                'training_performed': False, 'hyperparameter_selection_performed': False}
        return {'status': 'awaiting_target_data' if packet.is_dir() else 'awaiting_history_data',
                'prediction_packet_created': packet.is_dir(),
                'missing_history_days': [] if history.is_file() else ['2026-10-21'],
                'missing_target_days': ['2026-11-04']}

    def export(client, saved, phase, output, *, now):
        assert saved == protocol and not output.exists()
        if phase == 'targets':
            assert (data / 'notebook_independent_check/predictions').is_dir()
        events.append('export_' + phase)
        output.write_text('complete validated export')
        return {'phase': phase, 'rows': 14, 'sha256': sha256(output)}

    monkeypatch.setattr('automatic_check.verify_full_protocol', verify)
    monkeypatch.setattr('automatic_check.run_independent_check', run)
    monkeypatch.setattr('automatic_check.export_check_facts', export)
    return tmp_path, events


class Client:
    def __init__(self, events):
        self.events = events

    def query_df(self, query, parameters):
        self.events.append('history_calendar')
        assert 'order_count' not in query and 'unique_view_users' not in query
        assert parameters['end'] <= pd.Timestamp('2026-10-21').date()
        return pd.DataFrame({'date': pd.date_range(parameters['start'], parameters['end']),
                             'rows': 100})

    def close(self):
        self.events.append('close')


def test_checks_source_automatically_but_never_exports_an_open_window(project):
    root, events = project
    result = automatic_check.run_automatic_check(root, now='2026-10-20T16:00:00Z',
                                                client_factory=lambda: Client(events))
    assert result['source']['status'] == 'available'
    assert result['evaluation']['status'] == 'awaiting_history_data'
    assert 'history_calendar' in events and events[-1] == 'close'
    assert not any(event.startswith('export_') for event in events)
    assert 'models' not in result['evaluation']
    assert json.loads((root / 'reports/notebook_independent_status.json').read_text()) == result


def test_downloads_history_freezes_predictions_then_downloads_targets_in_one_run(project):
    root, events = project
    result = automatic_check.run_automatic_check(root, now='2026-11-05T00:00:00Z',
                                                client_factory=lambda: Client(events))
    assert result['evaluation']['status'] == 'evaluated_frozen_predictions'
    assert events.index('verify') < events.index('history_calendar')
    assert events.index('export_history') < events.index('predictions_saved') < events.index('export_targets')
    assert not result['evaluation']['training_performed']
    assert not result['evaluation']['hyperparameter_selection_performed']


def test_missing_private_inputs_never_connect_to_source(project):
    root, _ = project
    (root / 'data/facts.csv').unlink()
    result = automatic_check.run_automatic_check(root, now='2026-10-20T16:00:00Z',
        client_factory=lambda: pytest.fail('Public notebook must not access the source'))
    assert result['evaluation']['status'] == 'private_inputs_unavailable'
    assert result['source']['status'] == 'not_required'


def test_protocol_change_is_rejected_before_source_access(project):
    root, _ = project
    path = root / 'reports/notebook_full_protocol.json'
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='протокол изменён'):
        automatic_check.run_automatic_check(root, now='2026-10-20T16:00:00Z',
            client_factory=lambda: pytest.fail('Changed protocol must not access the source'))


def test_integrity_errors_are_not_disguised_as_missing_data(project, monkeypatch):
    root, _ = project

    def changed(*args):
        raise ValueError('Зафиксированная модель изменена')

    monkeypatch.setattr('automatic_check.verify_full_protocol', changed)
    with pytest.raises(ValueError, match='модель изменена'):
        automatic_check.run_automatic_check(root, now='2026-10-20T16:00:00Z',
            client_factory=lambda: pytest.fail('Changed model must not access the source'))


def test_local_files_work_even_when_source_is_unavailable_and_secrets_are_hidden(project, capsys):
    root, events = project
    (root / 'data/notebook_independent_history.csv').write_text('local export')

    def unavailable():
        print('secret-host-and-password', file=sys.stderr)
        logging.error('secret-host-and-password')
        raise RuntimeError('secret-host-and-password')

    original_logging = logging.root.manager.disable
    result = automatic_check.run_automatic_check(root, now='2026-10-22T00:00:00Z',
                                                client_factory=unavailable)
    assert result['source']['status'] == 'unavailable'
    assert result['evaluation']['prediction_packet_created']
    assert 'predictions_saved' in events
    assert 'secret' not in json.dumps(result)
    captured = capsys.readouterr()
    assert 'secret' not in captured.out + captured.err
    assert logging.root.manager.disable == original_logging


def test_certificate_failure_is_explicit_and_never_disables_verification(project):
    root, _ = project

    def unavailable():
        raise RuntimeError('CERTIFICATE_VERIFY_FAILED: secret-host-and-password')

    result = automatic_check.run_automatic_check(root, now='2026-10-20T16:00:00Z',
                                                client_factory=unavailable)
    assert result['source']['reason'] == 'certificate_verification_failed'
    assert result['evaluation']['status'] == 'awaiting_history_data'
    assert 'secret' not in json.dumps(result)


def test_existing_result_needs_no_source_or_repeated_exports(project):
    root, events = project
    data = root / 'data'
    (data / 'notebook_independent_history.csv').write_text('local history')
    (data / 'notebook_independent_targets.csv').write_text('local targets')
    (data / 'notebook_independent_check/predictions').mkdir(parents=True)
    result = automatic_check.run_automatic_check(root, now='2026-11-05T00:00:00Z',
        client_factory=lambda: pytest.fail('Saved result needs no source'))
    assert result['evaluation']['status'] == 'evaluated_frozen_predictions'
    assert result['source']['status'] == 'not_required'
    assert not any(event.startswith('export_') for event in events)


def test_incomplete_export_is_reported_not_published_or_called_a_result(project, monkeypatch):
    root, events = project

    def incomplete(*args, **kwargs):
        raise ValueError('Источник вернул неполный период')

    monkeypatch.setattr('automatic_check.export_check_facts', incomplete)
    result = automatic_check.run_automatic_check(root, now='2026-11-05T00:00:00Z',
                                                client_factory=lambda: Client(events))
    assert result['source']['exports'] == [{'phase': 'history', 'status': 'incomplete_or_invalid_data'}]
    assert result['evaluation']['status'] == 'awaiting_history_data'
    assert 'models' not in result['evaluation']
    assert not (root / 'data/notebook_independent_history.csv').exists()


@pytest.mark.parametrize('now', ['2026-10-20', 'NaT'])
def test_automatic_check_requires_a_real_timezone_aware_timestamp(project, now):
    root, _ = project
    with pytest.raises(ValueError, match='временной зон'):
        automatic_check.run_automatic_check(root, now=now,
            client_factory=lambda: pytest.fail('Invalid time must not access the source'))


@pytest.fixture
def cached_study(tmp_path, monkeypatch):
    (tmp_path / 'src').mkdir()
    (tmp_path / 'reports').mkdir()
    (tmp_path / 'data').mkdir()
    (tmp_path / 'src/model.py').write_text('SEED = 7\n')
    (tmp_path / 'requirements.txt').write_text('numpy\n')
    inputs = {key: 'data/' + key for key in ['facts', 'dictionary', 'embeddings']}
    (tmp_path / 'config.json').write_text(json.dumps({'inputs': inputs}))
    for relative in inputs.values():
        (tmp_path / relative).write_text('fixed input')
    for filename in ['notebook_research.json', 'notebook_review.json']:
        (tmp_path / 'reports' / filename).write_text('{"real_computation": true}')
    (tmp_path / 'notebook.json').write_text(json.dumps({'cells': [
        {'cell_type': 'code', 'source': ['seed = 7\n'], 'outputs': []},
        {'cell_type': 'markdown', 'source': ['Explanation']},
    ]}))
    monkeypatch.setattr('automatic_check._notebook_document',
                        lambda root: json.loads((root / 'notebook.json').read_text()))
    return tmp_path


def test_study_cache_requires_a_recorded_complete_computation(cached_study):
    root = cached_study
    assert not automatic_check.study_cache_valid(root)
    automatic_check.record_study_cache(root)
    assert automatic_check.study_cache_valid(root)


@pytest.mark.parametrize('changed', ['facts', 'dictionary', 'embeddings', 'code', 'notebook', 'report', 'requirements'])
def test_any_changed_input_or_training_code_invalidates_automatic_cache(cached_study, changed):
    root = cached_study
    automatic_check.record_study_cache(root)
    if changed in {'facts', 'dictionary', 'embeddings'}:
        path = root / 'data' / changed
    elif changed == 'code':
        path = root / 'src/model.py'
    elif changed == 'report':
        path = root / 'reports/notebook_research.json'
    elif changed == 'requirements':
        path = root / 'requirements.txt'
    else:
        path = root / 'notebook.json'
        notebook = json.loads(path.read_text())
        notebook['cells'][0]['source'] = ['seed = 17\n']
        path.write_text(json.dumps(notebook))
        assert not automatic_check.study_cache_valid(root)
        return
    path.write_text(path.read_text() + '\n')
    assert not automatic_check.study_cache_valid(root)


def test_rendered_outputs_do_not_invalidate_training_cache(cached_study):
    root = cached_study
    automatic_check.record_study_cache(root)
    path = root / 'notebook.json'
    notebook = json.loads(path.read_text())
    notebook['cells'][0]['outputs'] = [{'text': 'new rendered output'}]
    notebook['cells'][1]['source'] = ['Shorter explanation']
    path.write_text(json.dumps(notebook))
    assert automatic_check.study_cache_valid(root)


def test_partial_or_missing_study_files_never_activate_cache(cached_study):
    root = cached_study
    automatic_check.record_study_cache(root)
    (root / 'data/embeddings').unlink()
    assert not automatic_check.study_cache_valid(root)