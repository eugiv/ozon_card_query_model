import json
from math import prod

import numpy as np
import pandas as pd

from dormant_learning import ranking_metrics


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name}: нужно целое число >= {minimum}')
    return int(value)


def perturb_history(history, scenario, seed=7, fraction=.1):
    """Меняет support orders > 0, используя только узлы обучающей истории."""
    seed = _integer(seed, 'seed')
    if (isinstance(fraction, (bool, np.bool_)) or not np.isfinite(fraction)
            or not 0 < fraction <= 1):
        raise ValueError('fraction должна быть в (0, 1]')
    if scenario not in {'drop_edges', 'false_edges', 'rewire_edges'}:
        raise ValueError('Неизвестный сценарий')
    if not {'sku', 'normquery', 'orders'}.issubset(history.columns):
        raise ValueError('Нужны sku, normquery, orders')
    if history[['sku', 'normquery']].isna().any().any() or history.duplicated(['sku', 'normquery']).any():
        raise ValueError('Нужны уникальные непустые пары')
    if any(history[key].astype(str).str.strip().eq('').any() for key in ['sku', 'normquery']):
        raise ValueError('Пустой идентификатор')
    orders = pd.to_numeric(history.orders, errors='raise').to_numpy(dtype=float)
    if not np.isfinite(orders).all() or (orders < 0).any():
        raise ValueError('orders должны быть конечными неотрицательными числами')
    damaged = history.copy(deep=True).reset_index(drop=True)
    damaged['orders'] = orders
    pairs = list(zip(history.sku, history.normquery, strict=True))
    positive_positions = np.flatnonzero(orders > 0)
    positive = {pairs[index] for index in positive_positions}
    if not positive:
        raise ValueError('Нет положительных рёбер')
    count = max(1, int(np.floor(fraction * len(positive))))
    rng = np.random.default_rng(seed)
    if scenario in {'drop_edges', 'rewire_edges'}:
        removed = rng.choice(positive_positions, size=count, replace=False)
        damaged.loc[removed, 'orders'] = 0
    if scenario in {'false_edges', 'rewire_edges'}:
        skus = sorted(history.sku.unique())
        queries = sorted(history.normquery.unique())
        if len(skus) * len(queries) - len(positive) < count:
            raise ValueError('Недостаточно отсутствующих обучающих связей')
        added = set()
        for _ in range(count * 40 + 100):
            pair = skus[int(rng.integers(len(skus)))], queries[int(rng.integers(len(queries)))]
            if pair not in positive:
                added.add(pair)
            if len(added) == count:
                break
        if len(added) < count:
            for sku_index in rng.permutation(len(skus)):
                for query_index in rng.permutation(len(queries)):
                    pair = skus[sku_index], queries[query_index]
                    if pair not in positive:
                        added.add(pair)
                    if len(added) == count:
                        break
                if len(added) == count:
                    break
        existing = {pair: index for index, pair in enumerate(pairs)}
        rows = []
        for sku, query in sorted(added):
            if (sku, query) in existing:
                damaged.loc[existing[sku, query], 'orders'] = 1
            else:
                row = {key: 0 if pd.api.types.is_numeric_dtype(dtype) else None
                       for key, dtype in history.dtypes.items()}
                rows.append({**row, 'sku': sku, 'normquery': query, 'orders': 1})
        if rows:
            damaged = pd.concat([damaged, pd.DataFrame(rows, columns=history.columns)], ignore_index=True)
    damaged.attrs = history.attrs.copy()
    after = set(map(tuple, damaged.loc[damaged.orders.gt(0), ['sku', 'normquery']].to_numpy()))
    removed_count, added_count = len(positive - after), len(after - positive)
    return damaged, {'scenario': scenario, 'seed': seed, 'fraction': float(fraction),
        'edges_before': len(positive), 'edges_after': len(after), 'removed_edges': removed_count,
        'added_edges': added_count, 'changed_fraction': (removed_count + added_count) / len(positive)}


def paired_recall_intervals(candidates, positives, system_scores, baseline_scores, *, repeats=2000, seed=7):
    """Парные SKU-bootstrap интервалы двух macro Recall@10 и их разницы."""
    repeats = _integer(repeats, 'repeats', 1)
    seed = _integer(seed, 'seed')
    scores = []
    for values in [baseline_scores, system_scores]:
        array = np.asarray(values, dtype=float)
        if array.ndim != 1 or len(array) != len(candidates) or not np.isfinite(array).all():
            raise ValueError('Нужен конечный score для каждого кандидата')
        ranking_metrics(candidates[['sku', 'normquery']].assign(score=array), positives)
        scores.append(array)
    values = []
    for sku in sorted(positives.sku.unique()):
        selected = candidates.sku.eq(sku).to_numpy()
        targets = positives.loc[positives.sku.eq(sku)]
        frame = candidates.loc[selected, ['sku', 'normquery']]
        values.append([ranking_metrics(frame.assign(score=array[selected]), targets)['recall@10']
                       for array in scores])
    result = {'n_skus': len(values), 'n_targets': len(positives), 'repeats': repeats, 'seed': seed}
    names = ['baseline', 'system', 'difference', 'baseline_lower', 'baseline_upper',
             'system_lower', 'system_upper', 'lower', 'upper']
    if not values:
        return {**result, **dict.fromkeys(names)}
    matrix = np.asarray(values)
    samples = matrix[np.random.default_rng(seed).integers(0, len(matrix), size=(repeats, len(matrix)))].mean(axis=1)
    result.update(baseline=float(matrix[:, 0].mean()), system=float(matrix[:, 1].mean()),
                  difference=float((matrix[:, 1] - matrix[:, 0]).mean()))
    for index, prefix in [(0, 'baseline_'), (1, 'system_')]:
        result[prefix + 'lower'], result[prefix + 'upper'] = map(float, np.quantile(samples[:, index], [.025, .975]))
    result['lower'], result['upper'] = map(float, np.quantile(samples[:, 1] - samples[:, 0], [.025, .975]))
    return result


def random_configurations(space, budget, seed=20261020):
    budget = _integer(budget, 'budget', 1)
    seed = _integer(seed, 'seed')
    if not isinstance(space, dict) or not space:
        raise ValueError('Нужно непустое пространство поиска')
    keys = list(space)
    for key, values in space.items():
        if not isinstance(key, str) or not key or not isinstance(values, (list, tuple)) or not values:
            raise ValueError('Для каждого параметра нужен непустой список значений')
        if len(set(values)) != len(values):
            raise ValueError('Значения параметра повторяются')
    total = prod(len(values) for values in space.values())
    if budget > total:
        raise ValueError('Бюджет превышает пространство поиска')
    configurations = []
    for position in np.random.default_rng(seed).choice(total, size=budget, replace=False):
        indices = {}
        for key in reversed(keys):
            position, index = divmod(int(position), len(space[key]))
            indices[key] = index
        configurations.append({key: space[key][indices[key]] for key in keys})
    return configurations


def summarize_search(rows):
    if not rows:
        raise ValueError('Нет запусков')
    grouped = {}
    for row in rows:
        if row['model'] not in {'adapter', 'graph'} or not isinstance(row['parameters'], dict):
            raise ValueError('Неизвестная модель или параметры')
        _integer(row['seed'], 'seed')
        for key in ['validation_ndcg', 'validation_recall', 'seconds']:
            value = row[key]
            if not np.isfinite(value) or value < 0 or (key != 'seconds' and value > 1):
                raise ValueError('Некорректная метрика или время')
        if row['seconds'] <= 0:
            raise ValueError('Время должно быть положительным')
        key = row['model'], json.dumps(row['parameters'], sort_keys=True, allow_nan=False)
        grouped.setdefault(key, []).append(row)
    configurations, expected_seeds = [], None
    for (kind, parameters), runs in sorted(grouped.items()):
        seeds = sorted(row['seed'] for row in runs)
        if len(seeds) < 3 or len(set(seeds)) != len(seeds):
            raise ValueError('Нужны минимум три разных seed для каждой конфигурации')
        if expected_seeds is None:
            expected_seeds = seeds
        if seeds != expected_seeds:
            raise ValueError('Наборы seed должны совпадать')
        ndcg = np.asarray([row['validation_ndcg'] for row in runs])
        recall = np.asarray([row['validation_recall'] for row in runs])
        configurations.append({'model': kind, 'parameters': json.loads(parameters), 'seeds': seeds,
            'mean_ndcg': float(ndcg.mean()), 'std_ndcg': float(ndcg.std(ddof=1)),
            'mean_recall': float(recall.mean()), 'std_recall': float(recall.std(ddof=1)),
            'mean_seconds': float(np.mean([row['seconds'] for row in runs])), 'selected': False})
    selected = {}
    for kind in sorted({row['model'] for row in configurations}):
        own = [row for row in configurations if row['model'] == kind]
        best = max(own, key=lambda row: row['mean_ndcg'])
        for row in own:
            row['equivalent_to_best'] = best['mean_ndcg'] - row['mean_ndcg'] <= max(best['std_ndcg'], row['std_ndcg'])

        def complexity(row):
            parameters = row['parameters']
            if kind == 'adapter':
                return parameters['hidden_dim'], -parameters.get('batch_size', 128), json.dumps(parameters, sort_keys=True)
            return (parameters['dimensions'] * (parameters['layers'] + 1),
                    parameters['dimensions'], parameters['layers'])

        chosen = min([row for row in own if row['equivalent_to_best']], key=complexity)
        chosen['selected'] = True
        selected[kind] = chosen['parameters'].copy()
    return {'configurations': configurations, 'selected': selected}