"""Полные хронологические окна: применение одной фиксированной модели без подбора."""

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from cluster_data import (aggregate_clusters, dictionary_snapshot, future_positives,
                          period_pairs, sha256, write_json)
from cluster_recommender import HybridRetriever, lexical_index
from dormant_learning import fusion_scores, ranking_metrics
from full_history_recommender import DormantConfig, dormant_positives, enrich_candidates
from independent_check import BASELINE, SYSTEM, _facts
from notebook_research import data_overview
from notebook_review import load_full_frozen, verify_full_protocol
from research_diagnostics import paired_recall_intervals


def evaluate_closed_window(facts, dictionary, frozen, protocol, *, previously_observed_through,
                           cutoff=None):
    started = perf_counter()
    facts = _facts(facts)
    if facts.empty:
        raise ValueError('Нет фактов для оценки')
    end = facts.date.max() if cutoff is None else pd.Timestamp(cutoff)
    if pd.isna(end) or end.tzinfo is not None or facts.date.gt(end).any():
        raise ValueError('В снимке присутствуют даты после выбранного cutoff')
    decision = end - pd.Timedelta(days=14)
    history_start = decision - pd.Timedelta(days=13)
    required = pd.date_range(history_start, end)
    if not required.difference(pd.DatetimeIndex(facts.date.unique())).empty:
        raise ValueError('Нужны полные 14 дней истории и полные 14 целевых дней')
    if not pd.Timestamp(protocol['train_end']) <= pd.Timestamp(protocol['selection_end']) <= decision:
        raise ValueError('Обучение и выбор настроек должны завершиться до даты решения')
    expected_weights = {channel: .25 for channel in
        ['pretrained_embeddings', 'co_sales', 'semantic_adapter', 'lightgcn']}
    if (protocol['weights'][SYSTEM] != expected_weights
            or protocol['weights'][BASELINE] != {'pretrained_embeddings': .5, 'co_sales': .5}):
        raise ValueError('Нужна исходная полная модель и фиксированный контроль')
    snapshot, _ = dictionary_snapshot(dictionary, pd.Timestamp(protocol['dictionary_as_of']))
    catalog = frozen['catalog'].astype(str).tolist()
    if (catalog != sorted(set(catalog)) or len(catalog) != protocol['catalog_clusters']
            or not set(catalog).issubset(set(snapshot.normquery))):
        raise ValueError('Каталог не соответствует зафиксированному словарю')
    for name in ['e5', 'adapter', 'graph']:
        vectors = np.asarray(frozen[name])
        if vectors.ndim != 2 or len(vectors) != len(catalog) or not np.isfinite(vectors).all():
            raise ValueError('Векторы не соответствуют каталогу')
    daily, mapping = aggregate_clusters(facts, snapshot)
    full_history = period_pairs(daily, facts.date.min().date(), decision.date())
    full_recent = period_pairs(daily, history_start.date(), decision.date())
    history = full_history.loc[full_history.normquery.isin(catalog)].copy()
    recent = full_recent.loc[full_recent.normquery.isin(catalog)].copy()
    active = set(full_recent.sku)
    part = daily.loc[daily.date.between(history_start, decision)]
    demand = part.drop_duplicates(['date', 'normquery']).groupby('normquery').cluster_searches.sum()
    selection = pd.Timestamp(protocol['selection_end'])
    config = DormantConfig(pd.Timestamp(protocol['dictionary_as_of']).date(),
        pd.Timestamp(protocol['dictionary_as_of']).date(),
        (selection - pd.Timedelta(days=14)).date(), selection.date(),
        per_source=protocol['per_source'], max_views=protocol['max_views'], max_seeds=protocol['max_seeds'])
    text = lexical_index(catalog)
    base = HybridRetriever(catalog, history, demand, frozen['e5'], text).generate(
        config.per_source, max_history_views=config.max_views, traffic_history=recent,
        target_skus=active, max_seeds=config.max_seeds)
    pool = enrich_candidates(base, history, recent, demand, catalog, frozen['e5'],
                             frozen['adapter'], frozen['graph'], text, config, active)
    weights = {BASELINE: protocol['weights'][BASELINE], SYSTEM: protocol['weights'][SYSTEM],
        'Только E5': {'pretrained_embeddings': 1.}, 'Только совместные продажи': {'co_sales': 1.},
        'Только адаптер': {'semantic_adapter': 1.}, 'Только LightGCN': {'lightgcn': 1.},
        'База + адаптер': {name: 1 / 3 for name in ['pretrained_embeddings', 'co_sales', 'semantic_adapter']},
        'База + LightGCN': {name: 1 / 3 for name in ['pretrained_embeddings', 'co_sales', 'lightgcn']},
        'Четыре без обучения': {name: .25 for name in ['pretrained_embeddings', 'co_sales', 'lexical', 'popularity']}}
    scores = {name: fusion_scores(pool, chosen) for name, chosen in weights.items()}
    predictions = pool[['sku', 'normquery']].assign(
        baseline_score=scores[BASELINE], system_score=scores[SYSTEM]).sort_values(['sku', 'normquery'])
    prediction_digest = hashlib.sha256(predictions.to_csv(
        index=False, float_format='%.17g').encode()).hexdigest()
    future = period_pairs(daily, (decision + pd.Timedelta(days=1)).date(), end.date())
    targets = dormant_positives(future_positives(full_history, future), full_recent,
                                protocol['max_views'], active)
    metrics = {name: ranking_metrics(pool[['sku', 'normquery']].assign(score=values), targets)
               for name, values in scores.items()}
    options = {'repeats': protocol['bootstrap_samples'], 'seed': protocol['bootstrap_seed']}
    interval = paired_recall_intervals(pool, targets, scores[SYSTEM], scores[BASELINE], **options)
    channel_count_interval = paired_recall_intervals(
        pool, targets, scores[SYSTEM], scores['Четыре без обучения'], **options)
    sold = full_history.loc[full_history.orders.gt(0)].groupby('sku').size()
    sold = sold.reindex(targets.sku.unique(), fill_value=0)
    slices = []
    for label, lower, upper in [('0 кластеров', 0, 0), ('1–5 кластеров', 1, 5),
                                ('6–20 кластеров', 6, 20), ('21+ кластеров', 21, np.inf)]:
        group = set(sold.loc[sold.between(lower, upper)].index)
        selected = targets.loc[targets.sku.isin(group)]
        slices.append({'history': label, **paired_recall_intervals(
            pool, selected, scores[SYSTEM], scores[BASELINE], **options)})
    assert sum(row['n_targets'] for row in slices) == len(targets)
    assert sum(row['n_skus'] for row in slices) == targets.sku.nunique()
    found = pd.MultiIndex.from_frame(targets).isin(pd.MultiIndex.from_frame(pool[['sku', 'normquery']]))
    target_days = pd.date_range(decision + pd.Timedelta(days=1), end)
    observed = pd.Timestamp(previously_observed_through)
    if pd.isna(observed) or observed.tzinfo is not None:
        raise ValueError('Нужна календарная дата границы доступных ранее фактов')
    return {'schema_version': 1, 'status': 'evaluated_closed_window' if len(targets) else 'no_positive_targets',
        'evaluation_kind': 'completed_14_day_chronological_check', 'independent_test': False,
        'history_start': history_start.date().isoformat(), 'decision': decision.date().isoformat(),
        'target_start': target_days.min().date().isoformat(), 'target_end': end.date().isoformat(),
        'target_days': 14, 'new_source_target_days': int((target_days > observed).sum()),
        'previously_observed_through': observed.date().isoformat(),
        'catalog_clusters': len(catalog), 'candidate_pairs': len(pool), 'active_skus': len(active),
        'models': metrics, 'weights': weights, 'interval': interval, 'slices': slices,
        'channel_count_interval': channel_count_interval,
        'prediction_sha256': prediction_digest, 'targets_not_retrieved': int((~found).sum()),
        'normalization': {key: mapping[key] for key in ['input_rows', 'orders', 'covered_orders', 'order_coverage']},
        'training_performed': False, 'hyperparameter_selection_performed': False,
        'effect_confirmed_on_window': bool(len(targets) and interval['lower'] > 0),
        'seconds': perf_counter() - started}


def run_closed_validation(root, source_record):
    root = Path(root)
    protocol_path = root / 'reports/notebook_full_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    review = json.loads((root / 'reports/notebook_review.json').read_text())
    if sha256(protocol_path) != review['full_protocol_sha256']:
        raise ValueError('Зафиксированный протокол изменён')
    report_path = root / 'reports/notebook_current_validation.json'
    if source_record['status'] == 'saved_public_report':
        return json.loads(report_path.read_text())
    verify_full_protocol(root, protocol)
    source_path = root / source_record['snapshot_path']
    manifest_path = source_path / 'source_manifest.json'
    if sha256(manifest_path) != source_record['manifest_sha256']:
        raise ValueError('Снимок источника изменён')
    manifest = json.loads(manifest_path.read_text())
    for name, digest in manifest['files'].items():
        if sha256(source_path / name) != digest:
            raise ValueError('Файл снимка источника изменён')
    signature = {'source_manifest_sha256': sha256(manifest_path),
        'protocol_sha256': sha256(protocol_path), 'evaluation_code_sha256': sha256(Path(__file__))}
    if report_path.is_file():
        saved = json.loads(report_path.read_text())
        if saved.get('provenance') == signature:
            print('Расчёт полного окна проверен по снимку, модели и коду; используются сохранённые метрики.')
            return saved
    print('Полная модель: проверка артефактов и построение выдачи по истории до даты решения...', flush=True)
    facts = pd.read_csv(source_path / 'facts.csv', dtype={'sku': 'string', 'query': 'string'}, keep_default_na=False)
    dictionary = pd.read_csv(root / protocol['inputs']['dictionary']['path'], keep_default_na=False)
    frozen = load_full_frozen(root, dictionary, protocol)
    observed = json.loads((root / 'reports/experiment_summary.json').read_text())['source']['last_day']
    result = evaluate_closed_window(facts, dictionary, frozen, protocol,
                                    previously_observed_through=observed, cutoff=source_record['last_day'])
    snapshot, _ = dictionary_snapshot(dictionary, pd.Timestamp(protocol['dictionary_as_of']))
    result['data_overview'] = data_overview(facts, snapshot)
    result['provenance'] = signature
    result['evaluated_at'] = pd.Timestamp.now(tz='UTC').isoformat()
    write_json(report_path, result)
    return result