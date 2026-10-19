"""Воспроизводимые проверки данных и временного протокола ноутбука."""

from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from cluster_data import clean_text
from dormant_learning import reciprocal_ranks, ranking_metrics


def data_overview(facts, snapshot):
    dates = pd.to_datetime(facts['date']).dt.normalize()
    calendar = pd.date_range(dates.min(), dates.max())
    present = pd.DatetimeIndex(dates.unique())
    fields = []
    for column in facts:
        fields.append({'field': column, 'missing': int(facts[column].isna().sum()),
                       'blank': int(facts[column].astype(str).str.strip().eq('').sum())})
    distributions = []
    histograms = {}
    for column in ['order_count', 'unique_view_users', 'unique_search_users']:
        values = pd.to_numeric(facts[column], errors='raise').to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError('Некорректные значения воронки')
        q1, q3 = np.quantile(values, [.25, .75])
        fence = q3 + 1.5 * (q3 - q1)
        distributions.append({'field': column, 'zeros_pct': float(100 * (values == 0).mean()),
                              'median': float(np.median(values)), 'p95': float(np.quantile(values, .95)),
                              'p99': float(np.quantile(values, .99)), 'max': float(values.max()),
                              'iqr_fence': float(fence), 'above_fence': int((values > fence).sum())})
        bins = np.array([0, 1, 2, 5, 10, 25, 100, 1000, np.inf])
        histograms[column] = np.histogram(values, bins=bins)[0].tolist()
    cleaned = clean_text(facts['query'])
    mapped = cleaned.isin(set(snapshot.clean_query))
    daily = facts.assign(date=dates).groupby('date').agg(
        rows=('sku', 'size'), orders=('order_count', 'sum'), views=('unique_view_users', 'sum'))
    daily = daily.reindex(calendar)
    daily.index.name = 'date'
    daily = daily.reset_index()
    daily['date'] = daily['date'].dt.strftime('%Y-%m-%d')
    daily = daily.astype(object).where(daily.notna(), None)
    return {'rows': len(facts), 'skus': int(facts.sku.nunique()),
            'missing_days': calendar.difference(present).strftime('%Y-%m-%d').tolist(),
            'fields': fields, 'distributions': distributions, 'histograms': histograms,
            'daily': daily.to_dict('records'),
            'duplicate_keys': int(facts.duplicated(['date', 'sku', 'query']).sum()),
            'row_mapping_pct': float(100 * mapped.mean()),
            'query_mapping_pct': float(100 * cleaned.drop_duplicates().isin(set(snapshot.clean_query)).mean()),
            'orders_without_views_rows': int(((facts.order_count > 0) & (facts.unique_view_users == 0)).sum())}


def semantic_review_sample(facts, snapshot, size=100, seed=7):
    observed = set(clean_text(facts['query']))
    available = snapshot.loc[snapshot.clean_query.isin(observed)].sort_values('clean_query')
    sample = available.sample(n=min(size, len(available)), random_state=seed).copy()
    sample['correct'] = ''
    sample['comment'] = ''
    return sample.reset_index(drop=True)


def semantic_review_summary(review, expected):
    keys = ['clean_query', 'normquery']
    if not review[keys].astype(str).reset_index(drop=True).equals(expected[keys].astype(str).reset_index(drop=True)):
        raise ValueError('Изменена выборка проверки нормализации')
    labels = review['correct'].fillna('').astype(str).str.strip()
    if not labels.isin(['', '0', '1', '?']).all():
        raise ValueError('correct: 1 = верно, 0 = ошибка, ? = неоднозначно, пусто = не проверено')
    decided = labels.isin(['0', '1'])
    count = int(decided.sum())
    accuracy = float(labels.loc[decided].eq('1').mean()) if count else None
    interval = None
    if count:
        z = 1.96
        center = (accuracy + z * z / (2 * count)) / (1 + z * z / count)
        half = z * np.sqrt(accuracy * (1 - accuracy) / count + z * z / (4 * count * count)) / (1 + z * z / count)
        interval = [float(center - half), float(center + half)]
    return {'sample_size': len(review), 'decided': count, 'uncertain': int(labels.eq('?').sum()),
            'unreviewed': int(labels.eq('').sum()), 'accuracy_on_decided': accuracy,
            'wilson_95_on_decided': interval, 'complete': bool(labels.ne('').all()),
            'sampling_unit': 'uniform observed mapped query, not orders-weighted'}


def profile_scorer(candidates, history, catalog, *, weighted):
    """Те же профили продающих seeds, что у enrich_candidates; без future labels."""
    lookup = {query: i for i, query in enumerate(catalog)}
    skus = sorted(candidates.sku.unique())
    sku_lookup = {sku: i for i, sku in enumerate(skus)}
    rows, columns, values = [], [], []
    sold = history.loc[history.orders.gt(0) & history.sku.isin(skus)]
    for sku, own in sold.groupby('sku', sort=True):
        if weighted:
            own = own.sort_values(['orders', 'normquery'], ascending=[False, True], kind='stable').head(32)
        weights = own.orders.to_numpy(dtype=float, copy=True) if weighted else np.ones(len(own))
        weights /= weights.sum()
        rows.extend([sku_lookup[sku]] * len(own))
        columns.extend(lookup[q] for q in own.normquery)
        values.extend(weights)
    profiles = csr_matrix((values, (rows, columns)), shape=(len(skus), len(catalog)))
    sku_positions = candidates.sku.map(sku_lookup).to_numpy(dtype=int)
    query_positions = candidates.normquery.map(lookup).to_numpy(dtype=int)

    def score(vectors):
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) != len(catalog) or not np.isfinite(vectors).all():
            raise ValueError('Векторы не соответствуют каталогу')
        profiles_values = np.asarray(profiles @ vectors, dtype=np.float32)
        if not weighted:
            profiles_values /= np.maximum(np.linalg.norm(profiles_values, axis=1, keepdims=True), 1e-12)
        return np.einsum('ij,ij->i', vectors[query_positions], profiles_values[sku_positions])

    return score


def tune_fusion(candidates, positives, channels, units=4):
    """Сетка симплекса, максимизация только переданных validation NDCG@10."""
    if type(units) is not int or units < 1:
        raise ValueError('Число делений сетки должно быть положительным целым')
    if not channels or len(set(channels)) != len(channels):
        raise ValueError('Нужны различные каналы')
    ranks = reciprocal_ranks(candidates, channels)

    def compositions(total, count):
        if count == 1:
            yield (total,)
        else:
            for value in range(total + 1):
                for tail in compositions(total - value, count - 1):
                    yield (value, *tail)

    trials = []
    for numbers in compositions(units, len(channels)):
        weights = dict(zip(channels, [n / units for n in numbers], strict=True))
        prediction = candidates[['sku', 'normquery']].assign(
            score=sum(weights[name] * ranks[name] for name in channels))
        metric = ranking_metrics(prediction, positives)['ndcg@10']
        if metric is None:
            raise ValueError('Нет validation targets')
        trials.append({'weights': weights, 'ndcg@10': metric})
    best = max(trials, key=lambda trial: trial['ndcg@10'])
    return {**best, 'trials': len(trials), 'grid_step': 1 / units, 'offset': 60}


def prospective_status(protocol, source_days, *, model_training_end, selection_end):
    """Не позволяет объявить использованный или неполный период чистым тестом."""
    decision = pd.Timestamp(protocol['decision']).normalize()
    frozen = pd.Timestamp(protocol['frozen_at'])
    if frozen.tzinfo is None:
        raise ValueError('Нужна явная временная зона фиксации')
    if decision <= frozen.tz_convert('UTC').tz_localize(None).normalize():
        raise ValueError('Решение должно быть после фиксации протокола')
    if max(pd.Timestamp(model_training_end), pd.Timestamp(selection_end)) > decision:
        raise ValueError('Обучение или подбор использует будущий тест')
    horizon = protocol['window_days']
    if type(horizon) is not int or horizon < 1:
        raise ValueError('Некорректный горизонт')
    required = pd.date_range(decision + timedelta(days=1), periods=horizon)
    available = pd.DatetimeIndex(pd.to_datetime(source_days)).normalize()
    missing = required.difference(available)
    history_days = pd.date_range(decision - timedelta(days=13), decision)
    missing_history = history_days.difference(available)
    return {'status': 'ready' if missing.empty and missing_history.empty else 'awaiting_unseen_data',
            'target_start': required.min().strftime('%Y-%m-%d'),
            'target_end': required.max().strftime('%Y-%m-%d'),
            'missing_target_days': missing.strftime('%Y-%m-%d').tolist(),
            'missing_history_days': missing_history.strftime('%Y-%m-%d').tolist()}


def evaluate_prospective(facts, dictionary, frozen, protocol):
    """Только применение фиксированных векторов; никаких fit и поиска настроек."""
    from datetime import date

    from cluster_data import aggregate_clusters, dictionary_snapshot, future_positives, period_pairs
    from cluster_recommender import HybridRetriever, lexical_index
    from dormant_learning import fusion_scores
    from full_history_recommender import DormantConfig, bootstrap_difference, dormant_positives, enrich_candidates

    days = pd.DatetimeIndex(pd.to_datetime(facts.date).unique())
    status = prospective_status(protocol, days, model_training_end=protocol['train_end'],
                                selection_end=protocol['selection_end'])
    if status['status'] != 'ready':
        raise ValueError('Независимое окно или недавняя история ещё не полностью доступны')
    snapshot, _ = dictionary_snapshot(dictionary, pd.Timestamp(protocol['dictionary_as_of']))
    catalog = frozen['catalog'].astype(str).tolist()
    if not set(catalog).issubset(set(snapshot.normquery)):
        raise ValueError('Каталог содержит запросы вне фиксированного словаря')
    daily, _ = aggregate_clusters(facts, snapshot)
    decision = date.fromisoformat(protocol['decision'])
    start = (pd.Timestamp(decision) - timedelta(days=13)).date()
    full_history = period_pairs(daily, daily.date.min().date(), decision)
    full_recent = period_pairs(daily, start, decision)
    history = full_history.loc[full_history.normquery.isin(catalog)].copy()
    recent = full_recent.loc[full_recent.normquery.isin(catalog)].copy()
    active = set(full_recent.sku)
    part = daily.loc[daily.date.between(pd.Timestamp(start), pd.Timestamp(decision))]
    demand = part.drop_duplicates(['date', 'normquery']).groupby('normquery').cluster_searches.sum()
    config = DormantConfig(date(2026, 6, 26), date(2026, 6, 26), date(2026, 9, 4),
                           date(2026, 9, 18), per_source=protocol['per_source'], max_views=protocol['max_views'])
    vectors = frozen['e5']
    text = lexical_index(catalog)
    base = HybridRetriever(catalog, history, demand, vectors, text).generate(
        config.per_source, max_history_views=config.max_views, traffic_history=recent,
        target_skus=active, max_seeds=32)
    pool = enrich_candidates(base, history, recent, demand, catalog, vectors,
                             frozen['adapter'], frozen['graph'], text, config, active)
    future = period_pairs(daily, decision + timedelta(days=1), decision + timedelta(days=protocol['window_days']))
    targets = dormant_positives(future_positives(full_history, future), full_recent, config.max_views, active)
    if targets.empty:
        raise ValueError('Нет положительных целей для независимой оценки')
    scores = {name: fusion_scores(pool, weights) for name, weights in protocol['weights'].items()}
    metrics = {name: ranking_metrics(pool[['sku', 'normquery']].assign(score=score), targets)
               for name, score in scores.items()}
    interval = bootstrap_difference(pool, targets, scores['Полные четыре'], scores['База: E5 + co-sales'])
    return {'status': 'evaluated_frozen_protocol', 'models': metrics, 'interval': interval,
            'effect_confirmed': interval['lower'] > 0 and interval['difference'] > 0,
            'training_performed': False, 'hyperparameter_selection_performed': False}
