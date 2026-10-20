"""Автоматический запуск зафиксированной проверки без изменения её протокола."""

import hashlib
import io
import json
import logging
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cluster_data import sha256, source_client, write_json
from independent_check import export_check_facts, run_independent_check
from notebook_review import verify_full_protocol


HISTORY_CALENDAR_SQL = """
SELECT date, count() AS rows
FROM default_staging.stg_ozon_product_queries_details
WHERE date BETWEEN {start:Date} AND {end:Date}
GROUP BY date
ORDER BY date
"""


@contextmanager
def _private_source_output():
    disabled = logging.root.manager.disable
    try:
        logging.disable(logging.CRITICAL)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            yield
    finally:
        logging.disable(disabled)


def run_automatic_check(root, *, now=None, client_factory=None):
    """Проверяет источник, выгружает завершённые этапы и оценивает сохранённую выдачу."""
    root = Path(root)
    stamp = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    if not isinstance(stamp, pd.Timestamp) or stamp.tzinfo is None:
        raise ValueError('Нужна действительная дата с явной временной зоной')
    stamp = stamp.tz_convert('UTC')
    protocol_path = root / 'reports/notebook_full_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    review = json.loads((root / 'reports/notebook_review.json').read_text())
    if sha256(protocol_path) != review['full_protocol_sha256']:
        raise ValueError('Зафиксированный протокол изменён')
    history_path = root / 'data/notebook_independent_history.csv'
    target_path = root / 'data/notebook_independent_targets.csv'
    folder = root / 'data/notebook_independent_check'
    source: dict[str, Any] = {'status': 'not_required', 'exports': []}

    def save(evaluation):
        result = {'schema_version': 1, 'checked_at': stamp.isoformat(),
                  'protocol_sha256': sha256(protocol_path),
                  'source': source, 'evaluation': evaluation}
        write_json(root / 'reports/notebook_independent_status.json', result)
        return result

    records = [*protocol['inputs'].values(), *protocol['artifacts'].values()]
    missing = [record['path'] for record in records if not (root / record['path']).is_file()]
    if missing:
        return save({'status': 'private_inputs_unavailable', 'missing_files': missing,
                     'prediction_packet_created': False})
    verify_full_protocol(root, protocol)
    preflight = run_independent_check(root, history_path, target_path, folder, now=stamp, check_only=True)
    if preflight['status'] == 'ready_to_evaluate':
        return save(run_independent_check(root, history_path, target_path, folder, now=stamp))

    decision = pd.Timestamp(protocol['decision'])
    history_start = decision - pd.Timedelta(days=13)
    last_closed = stamp.normalize().tz_localize(None) - pd.Timedelta(days=1)
    client: Any = None
    evaluation = preflight
    try:
        try:
            with _private_source_output():
                client = (source_client if client_factory is None else client_factory)()
            source['status'] = 'available'
        except Exception as failure:
            reason = ('certificate_verification_failed' if 'certificate_verify_failed' in str(failure).lower()
                      else 'connection_failed')
            source.update(status='unavailable', stage='connection', reason=reason)
        if client is not None and last_closed >= history_start:
            end = min(last_closed, decision)
            try:
                with _private_source_output():
                    calendar = client.query_df(HISTORY_CALENDAR_SQL,
                        parameters={'start': history_start.date(), 'end': end.date()})
                dates = pd.to_datetime(calendar['date'], errors='raise')
                counts = pd.to_numeric(calendar['rows'], errors='raise')
                if (dates.isna().any() or dates.duplicated().any()
                        or not dates.between(history_start, end).all()
                        or not np.isfinite(counts).all() or counts.le(0).any()
                        or counts.mod(1).ne(0).any()):
                    raise ValueError('Некорректный календарь источника')
                source['history_available_days'] = dates.dt.strftime('%Y-%m-%d').tolist()
                source['missing_closed_history_days'] = [day.strftime('%Y-%m-%d') for day in
                    pd.date_range(history_start, end).difference(pd.DatetimeIndex(dates))]
            except Exception:
                source.update(status='unavailable', stage='history_calendar')

        def download(phase, path, end):
            if path.is_file() or source['status'] != 'available' or last_closed < end:
                return
            try:
                with _private_source_output():
                    exported = export_check_facts(client, protocol, phase, path, now=stamp)
                source['exports'].append({**exported, 'status': 'downloaded'})
            except ValueError:
                source['exports'].append({'phase': phase, 'status': 'incomplete_or_invalid_data'})
            except Exception:
                source['exports'].append({'phase': phase, 'status': 'source_unavailable'})

        if not (folder / 'predictions').is_dir():
            download('history', history_path, decision)
        evaluation = run_independent_check(root, history_path, target_path, folder, now=stamp)
        if evaluation.get('prediction_packet_created'):
            download('targets', target_path, decision + pd.Timedelta(days=protocol['window_days']))
            if target_path.is_file():
                evaluation = run_independent_check(root, history_path, target_path, folder, now=stamp)
    finally:
        if client is not None:
            try:
                with _private_source_output():
                    client.close()
            except Exception:
                source['close_failed'] = True
    return save(evaluation)


def _notebook_document(root):
    return json.loads((root / 'recommender_demo.ipynb').read_text())


def _study_signature(root):
    config = json.loads((root / 'config.json').read_text())
    cells = _notebook_document(root)['cells']
    sources = [''.join(cell['source']) for cell in cells if cell['cell_type'] == 'code']
    return {'inputs': {name: sha256(root / config['inputs'][name])
                       for name in ['facts', 'dictionary', 'embeddings']},
        'config': sha256(root / 'config.json'), 'requirements': sha256(root / 'requirements.txt'),
        'code': {path.name: sha256(path) for path in sorted((root / 'src').iterdir())
                 if path.suffix == '.py'},
        'notebook_code': hashlib.sha256(json.dumps(sources, ensure_ascii=False).encode()).hexdigest(),
        'python': sys.version,
        'libraries': {name: version(name) for name in
                      ['numpy', 'pandas', 'torch', 'scipy', 'scikit-learn', 'catboost', 'sentence-transformers']},
        'reports': {name: sha256(root / 'reports' / name) for name in
                    ['notebook_research.json', 'notebook_review.json']},
        'vectors': sha256(root / 'data/notebook_frozen_vectors.npz')
                   if (root / 'data/notebook_frozen_vectors.npz').is_file() else None}


def study_cache_valid(root):
    root = Path(root)
    path = root / 'reports/notebook_pipeline_cache.json'
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text())
        return record['schema_version'] == 1 and record['signature'] == _study_signature(root)
    except (OSError, ValueError, KeyError):
        return False


def record_study_cache(root):
    root = Path(root)
    record = {'schema_version': 1, 'verified_at': pd.Timestamp.now(tz='UTC').isoformat(),
              'signature': _study_signature(root)}
    write_json(root / 'reports/notebook_pipeline_cache.json', record)
    return record