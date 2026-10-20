"""Независимый holdout: фиксация расчёта → рекомендации → открытие целевых фактов."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from cluster_data import (FACT_COLUMNS, FACT_SQL, aggregate_clusters, dictionary_snapshot,
                          future_positives, period_pairs, publish_snapshot,
                          require_columns, sha256, write_json)
from cluster_recommender import HybridRetriever, lexical_index
from dormant_learning import fusion_scores, paired_metric_interval, ranking_metrics
from full_history_recommender import DormantConfig, dormant_positives, enrich_candidates
from notebook_research import prospective_status

SYSTEM = 'Полные четыре'
BASELINE = 'База: E5 + co-sales'


def _utc(now):
    stamp = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError('Время запуска должно иметь явную временную зону')
    return stamp.tz_convert('UTC')


def _digest(payload):
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False).encode()).hexdigest()


def _frame_digest(frame):
    return hashlib.sha256(frame.to_csv(index=False, float_format='%.17g').encode()).hexdigest()


def _schedule(protocol):
    prospective_status(protocol, [], model_training_end=protocol['train_end'],
                       selection_end=protocol['selection_end'])
    if protocol.get('history_days', 14) != 14 or protocol['window_days'] != 14:
        raise ValueError('Для этой проверки нужны полные 14-дневные окна')
    decision = pd.Timestamp(protocol['decision'])
    history = pd.date_range(decision - pd.Timedelta(days=13), decision)
    target = pd.date_range(decision + pd.Timedelta(days=1), periods=14)
    return history, target


def _registration(protocol, folder):
    path = Path(folder) / 'registration.json'
    if not path.is_file():
        raise ValueError('Сначала нужна предварительная фиксация расчёта')
    record = json.loads(path.read_text(encoding='utf-8'))
    if (record['protocol_sha256'] != _digest(protocol)
            or record['execution_sha256'] != sha256(Path(__file__))):
        raise ValueError('Протокол или код независимой проверки изменён после фиксации')
    _, target = _schedule(protocol)
    registered = _utc(record['registered_at'])
    if not _utc(protocol['frozen_at']) <= registered < target.min().tz_localize('UTC'):
        raise ValueError('Предварительная фиксация сделана вне допустимого срока')
    return record


def register_check(protocol, folder, *, now=None):
    """Локально закрепляет и модель, и код расчёта до начала целевого периода."""
    folder = Path(folder)
    if (folder / 'registration.json').exists():
        return _registration(protocol, folder)
    history, target = _schedule(protocol)
    stamp = _utc(now)
    if not _utc(protocol['frozen_at']) <= stamp < target.min().tz_localize('UTC'):
        raise ValueError('Нельзя создавать предварительную фиксацию задним числом')
    record = {'schema_version': 1, 'status': 'registered_not_evaluated',
              'registered_at': stamp.isoformat(), 'protocol_sha256': _digest(protocol),
              'execution_sha256': sha256(Path(__file__)),
              'history_start': history.min().date().isoformat(),
              'history_end': history.max().date().isoformat(),
              'target_start': target.min().date().isoformat(),
              'target_end': target.max().date().isoformat(),
              'registration': 'local only; no external timestamp or preregistration'}
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=folder, prefix='.registration-') as temporary:
        path = Path(temporary) / 'registration.json'
        write_json(path, record)
        try:
            os.link(path, folder / 'registration.json')
        except FileExistsError:
            return _registration(protocol, folder)
    return record


def _facts(frame):
    require_columns(frame, FACT_COLUMNS)
    data = frame[FACT_COLUMNS].copy()
    data['date'] = pd.to_datetime(data['date'], errors='raise')
    if data.date.isna().any() or data.date.dt.tz is not None:
        raise ValueError('Нужны календарные даты источника без пропусков и timezone')
    if not data.date.eq(data.date.dt.normalize()).all():
        raise ValueError('Ожидается дата источника, не время внутри дня')
    for column in ['sku', 'query']:
        data[column] = data[column].astype('string')
        if data[column].isna().any() or data[column].str.strip().eq('').any():
            raise ValueError('SKU и query должны быть заполнены')
    for column in ['order_count', 'unique_view_users', 'unique_search_users']:
        values = pd.to_numeric(data[column], errors='raise').astype(float)
        if not np.isfinite(values).all() or (values < 0).any() or (values % 1 != 0).any():
            raise ValueError('Счётчики должны быть конечными неотрицательными целыми')
        data[column] = values
    if data.duplicated(['date', 'sku', 'query']).any():
        raise ValueError('Дубли date + sku + query недопустимы')
    return data.sort_values(['date', 'sku', 'query']).reset_index(drop=True)


def _snapshot(dictionary, protocol):
    snapshot, _ = dictionary_snapshot(dictionary, pd.Timestamp(protocol['dictionary_as_of']))
    fingerprint = _frame_digest(snapshot[['clean_query', 'normquery']].sort_values(
        'clean_query').reset_index(drop=True))
    return snapshot, fingerprint


def _publish(output, frames, payload, name='manifest.json'):
    output = Path(output)
    if output.exists():
        raise ValueError('Результат уже существует; перезапись запрещена')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix='.independent-') as temporary:
        folder = Path(temporary) / 'snapshot'
        folder.mkdir()
        for filename, frame in frames.items():
            frame.to_csv(folder / filename, index=False, float_format='%.17g')
        payload = {**payload, 'files': {filename: sha256(folder / filename) for filename in frames}}
        write_json(folder / name, payload)
        publish_snapshot(folder, output)
    return payload


def freeze_predictions(history_facts, dictionary, frozen, protocol, folder, *, now=None):
    """Не принимает целевые факты: сохраняет общий пул и оба фиксированных ранжирования."""
    folder = Path(folder)
    registration = _registration(protocol, folder)
    output = folder / 'predictions'
    if output.exists():
        raise ValueError('Рекомендации уже сохранены; перезапись запрещена')
    history_days, target_days = _schedule(protocol)
    stamp = _utc(now)
    if stamp < target_days.min().tz_localize('UTC'):
        raise ValueError('День решения ещё не завершён: история может быть неполной')
    facts = _facts(history_facts)
    if facts.empty or facts.date.gt(history_days.max()).any():
        raise ValueError('В историю нельзя передавать будущие целевые строки')
    if not history_days.difference(pd.DatetimeIndex(facts.date.unique())).empty:
        raise ValueError('Недавняя история доступна не полностью')
    snapshot, dictionary_digest = _snapshot(dictionary, protocol)
    catalog = frozen['catalog'].astype(str).tolist()
    if (len(catalog) != protocol['catalog_clusters'] or catalog != sorted(set(catalog))
            or not set(catalog).issubset(set(snapshot.normquery))):
        raise ValueError('Нужен фиксированный отсортированный каталог без повторов')
    for name in ['e5', 'adapter', 'graph']:
        vectors = np.asarray(frozen[name])
        if vectors.ndim != 2 or len(vectors) != len(catalog) or not np.isfinite(vectors).all():
            raise ValueError('Векторы не соответствуют фиксированному каталогу')
    daily, _ = aggregate_clusters(facts, snapshot)
    if daily.empty:
        raise ValueError('Нет истории, сопоставленной фиксированному словарю')
    decision = history_days.max().date()
    full_history = period_pairs(daily, daily.date.min().date(), decision)
    full_recent = period_pairs(daily, history_days.min().date(), decision)
    history = full_history.loc[full_history.normquery.isin(catalog)].copy()
    recent = full_recent.loc[full_recent.normquery.isin(catalog)].copy()
    active = set(full_recent.sku)
    part = daily.loc[daily.date.between(history_days.min(), history_days.max())]
    demand = part.drop_duplicates(['date', 'normquery']).groupby('normquery').cluster_searches.sum()
    selection = pd.Timestamp(protocol['selection_end']).date()
    config = DormantConfig(pd.Timestamp(protocol['dictionary_as_of']).date(),
        pd.Timestamp(protocol['dictionary_as_of']).date(),
        (pd.Timestamp(selection) - pd.Timedelta(days=14)).date(), selection,
        per_source=protocol['per_source'], max_views=protocol['max_views'],
        max_seeds=protocol['max_seeds'])
    text = lexical_index(catalog)
    base = HybridRetriever(catalog, history, demand, frozen['e5'], text).generate(
        config.per_source, max_history_views=config.max_views, traffic_history=recent,
        target_skus=active, max_seeds=config.max_seeds)
    pool = enrich_candidates(base, history, recent, demand, catalog, frozen['e5'],
                             frozen['adapter'], frozen['graph'], text, config, active)
    predictions = pool[['sku', 'normquery']].assign(
        baseline_score=fusion_scores(pool, protocol['weights'][BASELINE]),
        system_score=fusion_scores(pool, protocol['weights'][SYSTEM]))
    predictions = predictions.sort_values(['sku', 'normquery']).reset_index(drop=True)
    manifest = {'status': 'predictions_frozen', 'created_at': stamp.isoformat(),
                'registration_sha256': _digest(registration), 'protocol_sha256': _digest(protocol),
                'dictionary_sha256': dictionary_digest, 'history_input_sha256': _frame_digest(facts),
                'history_rows': len(facts), 'history_end': decision.isoformat(),
                'catalog_clusters': len(catalog), 'active_skus': len(active),
                'candidate_pairs': len(predictions), 'training_performed': False,
                'hyperparameter_selection_performed': False}
    return _publish(output, {'predictions.csv': predictions, 'history.csv': full_history,
                             'recent.csv': full_recent}, manifest)


def _packet(protocol, folder):
    registration = _registration(protocol, folder)
    path = Path(folder) / 'predictions/manifest.json'
    if not path.is_file():
        raise ValueError('Нет сохранённых рекомендаций; сначала нужен отдельный этап построения')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if (manifest['registration_sha256'] != _digest(registration)
            or manifest['protocol_sha256'] != _digest(protocol)
            or set(manifest['files']) != {'predictions.csv', 'history.csv', 'recent.csv'}):
        raise ValueError('Пакет рекомендаций изменён или относится к другому протоколу')
    for filename, digest in manifest['files'].items():
        if sha256(path.parent / filename) != digest:
            raise ValueError('Сохранённые рекомендации или история изменены')
    return manifest


def _read_table(path):
    return pd.read_csv(path, dtype={'sku': 'string', 'normquery': 'string'},
                       keep_default_na=False, float_precision='round_trip')


def evaluate_predictions(target_facts, dictionary, protocol, folder, *, now=None):
    """Оценивает только записанные scores; не загружает модели и не строит кандидатов."""
    folder = Path(folder)
    manifest = _packet(protocol, folder)
    _, target_days = _schedule(protocol)
    stamp = _utc(now)
    if stamp < (target_days.max() + pd.Timedelta(days=1)).tz_localize('UTC'):
        raise ValueError('Последний целевой день ещё не завершён')
    facts = _facts(target_facts)
    if not facts.date.between(target_days.min(), target_days.max()).all():
        raise ValueError('Файл целевых фактов должен содержать только независимое окно')
    if not target_days.difference(pd.DatetimeIndex(facts.date.unique())).empty:
        raise ValueError('Целевое окно доступно не полностью')
    snapshot, dictionary_digest = _snapshot(dictionary, protocol)
    if dictionary_digest != manifest['dictionary_sha256']:
        raise ValueError('Соответствия фиксированного словаря изменены')
    target_digest = _frame_digest(facts)
    output = folder / 'evaluation'
    if output.exists():
        saved = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
        if (saved['target_input_sha256'] != target_digest
                or saved['predictions_sha256'] != _digest(manifest)
                or sha256(output / 'result.json') != saved['result_sha256']):
            raise ValueError('Итоговый результат уже зафиксирован; замена выгрузки запрещена')
        return json.loads((output / 'result.json').read_text(encoding='utf-8'))
    daily, audit = aggregate_clusters(facts, snapshot)
    history = _read_table(folder / 'predictions/history.csv')
    recent = _read_table(folder / 'predictions/recent.csv')
    predictions = _read_table(folder / 'predictions/predictions.csv')
    future = period_pairs(daily, target_days.min().date(), target_days.max().date())
    targets = dormant_positives(future_positives(history, future), recent,
                                protocol['max_views'], set(recent.sku))
    scored = {name: predictions[['sku', 'normquery']].assign(score=predictions[column])
              for name, column in [(BASELINE, 'baseline_score'), (SYSTEM, 'system_score')]}
    metrics = {name: ranking_metrics(frame, targets) for name, frame in scored.items()}
    interval = None
    if not targets.empty:
        paired = paired_metric_interval(scored[SYSTEM], scored[BASELINE], targets,
            samples=protocol['bootstrap_samples'], seed=protocol['bootstrap_seed'])
        interval = {'difference': paired['difference'], 'lower': paired['95_interval'][0],
                    'upper': paired['95_interval'][1], 'n_skus': paired['skus'],
                    'repeats': paired['samples'], 'seed': paired['seed'], 'method': paired['method']}
    found = pd.MultiIndex.from_frame(targets).isin(pd.MultiIndex.from_frame(predictions[['sku', 'normquery']]))
    result = {'status': 'evaluated_frozen_predictions' if len(targets) else 'no_positive_targets',
              'evaluated_at': stamp.isoformat(), 'target_start': target_days.min().date().isoformat(),
              'target_end': target_days.max().date().isoformat(), 'models': metrics,
              'interval': interval, 'effect_confirmed': bool(interval and interval['lower'] > 0
                                                            and interval['difference'] > 0),
              'targets_not_retrieved': int((~found).sum()),
              'normalization': {key: audit[key] for key in
                                ['input_rows', 'orders', 'covered_orders', 'order_coverage']},
              'protocol_sha256': _digest(protocol), 'predictions_sha256': _digest(manifest),
              'target_input_sha256': target_digest, 'training_performed': False,
              'hyperparameter_selection_performed': False,
              'limitation': 'single frozen model; offline ranking, not sales uplift; local registration only'}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix='.evaluation-') as temporary:
        bundle = Path(temporary) / 'result'
        bundle.mkdir()
        write_json(bundle / 'result.json', result)
        write_json(bundle / 'manifest.json', {'target_input_sha256': target_digest,
            'predictions_sha256': _digest(manifest), 'result_sha256': sha256(bundle / 'result.json')})
        publish_snapshot(bundle, output)
    return result


def export_check_facts(client, protocol, phase, output, *, now=None):
    """Только SELECT нужного периода; словарь остаётся фиксированным, неполный CSV не публикуется."""
    history_days, target_days = _schedule(protocol)
    if phase not in {'history', 'targets'}:
        raise ValueError('Этап выгрузки: history или targets')
    days = history_days if phase == 'history' else target_days
    if _utc(now) < (days.max() + pd.Timedelta(days=1)).tz_localize('UTC'):
        raise ValueError('Запрошенный период ещё не завершён')
    output = Path(output)
    if output.exists():
        raise ValueError('Выгрузка уже существует; перезапись запрещена')
    try:
        facts = client.query_df(FACT_SQL, parameters={'start': days.min().date(),
                                                     'end': days.max().date()})
    except Exception:
        raise RuntimeError('Источник недоступен; подробности соединения не выводятся') from None
    facts = _facts(facts)
    if (not facts.date.between(days.min(), days.max()).all()
            or not days.difference(pd.DatetimeIndex(facts.date.unique())).empty):
        raise ValueError('Источник вернул неполный период или посторонние даты')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix='.facts-') as temporary:
        path = Path(temporary) / 'facts.csv'
        facts.to_csv(path, index=False, float_format='%.17g')
        os.link(path, output)
    return {'phase': phase, 'rows': len(facts), 'sha256': sha256(output),
            'start': days.min().date().isoformat(), 'end': days.max().date().isoformat(),
            'completeness': 'closed calendar window and rows on every day; source delivery must be complete'}


def run_independent_check(root, history_path, target_path, folder, *, now=None, check_only=False):
    """Вход из ноутбука: проверяет подлинность полной модели перед любыми действиями."""
    from notebook_review import load_full_frozen, verify_full_protocol

    root = Path(root)
    protocol_path = root / 'reports/notebook_full_protocol.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    review = json.loads((root / 'reports/notebook_review.json').read_text(encoding='utf-8'))
    if sha256(protocol_path) != review['full_protocol_sha256']:
        raise ValueError('Зафиксированный протокол изменён')
    verify_full_protocol(root, protocol)
    registration = register_check(protocol, folder, now=now)
    history_days, target_days = _schedule(protocol)
    stamp = _utc(now)
    state = {'status': 'awaiting_history_data', 'registered_at': registration['registered_at'],
             'target_start': registration['target_start'], 'target_end': registration['target_end'],
             'prediction_packet_created': False, 'missing_history_days': [],
             'missing_target_days': target_days.strftime('%Y-%m-%d').tolist()}
    folder = Path(folder)
    if not (folder / 'predictions').exists():
        original_path = root / protocol['inputs']['facts']['path']
        original_days = pd.to_datetime(pd.read_csv(original_path, usecols=['date']).date)
        available_days = original_days
        if Path(history_path).is_file():
            extra_days = pd.to_datetime(pd.read_csv(history_path, usecols=['date']).date)
            if extra_days.le(original_days.max()).any():
                raise ValueError('Дополнение истории повторно включает использованные даты')
            available_days = pd.concat([original_days, extra_days], ignore_index=True)
        if available_days.isna().any() or available_days.gt(history_days.max()).any():
            raise ValueError('В дополнении истории присутствуют будущие целевые факты')
        state['history_available_through'] = available_days.max().date().isoformat()
        state['missing_history_days'] = history_days.difference(
            pd.DatetimeIndex(available_days.unique())).strftime('%Y-%m-%d').tolist()
        if state['missing_history_days']:
            return state
        if stamp < target_days.min().tz_localize('UTC'):
            return {**state, 'status': 'awaiting_history_close'}
        if check_only:
            return {**state, 'status': 'ready_to_freeze'}
        original = _facts(pd.read_csv(original_path, dtype={'sku': 'string', 'query': 'string'},
                                     keep_default_na=False))
        history = original
        if Path(history_path).is_file():
            extra = _facts(pd.read_csv(history_path, dtype={'sku': 'string', 'query': 'string'},
                                      keep_default_na=False))
            history = pd.concat([original, extra], ignore_index=True)
        dictionary = pd.read_csv(root / protocol['inputs']['dictionary']['path'], keep_default_na=False)
        frozen = load_full_frozen(root, dictionary, protocol)
        freeze_predictions(history, dictionary, frozen, protocol, folder, now=stamp)
        del frozen, history, original, dictionary
    _packet(protocol, folder)
    state.update(status='awaiting_target_data', prediction_packet_created=True)
    if stamp < (target_days.max() + pd.Timedelta(days=1)).tz_localize('UTC'):
        return {**state, 'status': 'awaiting_target_close'}
    if not Path(target_path).is_file():
        return state
    if check_only:
        return {**state, 'status': 'ready_to_evaluate'}
    targets = _facts(pd.read_csv(target_path, dtype={'sku': 'string', 'query': 'string'},
                                keep_default_na=False))
    if not targets.date.between(target_days.min(), target_days.max()).all():
        raise ValueError('Файл целевых фактов содержит даты вне независимого окна')
    state['missing_target_days'] = target_days.difference(
        pd.DatetimeIndex(targets.date.unique())).strftime('%Y-%m-%d').tolist()
    if state['missing_target_days']:
        return state
    dictionary = pd.read_csv(root / protocol['inputs']['dictionary']['path'], keep_default_na=False)
    return evaluate_predictions(targets, dictionary, protocol, folder, now=stamp)