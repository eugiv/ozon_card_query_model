"""Переобучение с исключённым временным периодом и отдельным чтением его целей."""

import hashlib
import json
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from cluster_data import (aggregate_clusters, dictionary_snapshot, future_positives,
                          period_pairs, sha256, write_json)
from cluster_recommender import HybridRetriever, lexical_index, load_embeddings
from dormant_learning import fusion_scores, ranking_metrics
from full_history_recommender import DormantConfig, dormant_positives, enrich_candidates
from graph_learning import fit_graph, load_graph, save_graph
from independent_check import BASELINE, SYSTEM, _facts
from notebook_research import profile_scorer
from research_diagnostics import random_configurations, summarize_search
from semantic_adapter import apply_adapter, fit_adapter, load_adapter, save_adapter


CONTROL = 'Четыре без обучения'
WEIGHTS = {
    BASELINE: {'pretrained_embeddings': .5, 'co_sales': .5},
    SYSTEM: {name: .25 for name in ['pretrained_embeddings', 'co_sales', 'semantic_adapter', 'lightgcn']},
    CONTROL: {name: .25 for name in ['pretrained_embeddings', 'co_sales', 'lexical', 'popularity']},
    'База + адаптер': {name: 1 / 3 for name in ['pretrained_embeddings', 'co_sales', 'semantic_adapter']},
    'База + LightGCN': {name: 1 / 3 for name in ['pretrained_embeddings', 'co_sales', 'lightgcn']},
    'Только адаптер': {'semantic_adapter': 1.},
    'Только LightGCN': {'lightgcn': 1.},
}
SPACE = {'adapter': {'hidden_dim': [64, 128], 'batch_size': [64, 128]},
         'graph': {'dimensions': [16, 32, 64], 'layers': [1, 2]}}


def default_protocol():
    return {
        'schema_version': 1, 'evaluation_kind': 'retrained_temporal_holdout',
        'dictionary_as_of': '2026-06-26T23:59:59.999999999Z',
        'train_end': '2026-08-07', 'epoch_validation_end': '2026-08-21',
        'decision': '2026-09-04', 'target_start': '2026-09-05', 'target_end': '2026-09-18',
        'window_days': 14, 'seeds': [7, 17, 27, 37, 47], 'search_seed': 20261020,
        'search_budget': 3, 'search_repeats': 3, 'epochs': 40, 'patience': 5,
        'min_delta': .0001, 'per_source': 150, 'max_seeds': 32, 'max_views': 5,
        'bootstrap_samples': 2000, 'bootstrap_seed': 7,
        'primary_metric': 'macro Recall@10 averaged over every predefined seed',
        'criterion': 'paired SKU bootstrap lower > 0 and gain > 0 in every predefined seed',
        'weights': {name: values.copy() for name, values in WEIGHTS.items()},
        'catalog_policy': 'entire dictionary snapshot available before gradient training',
        'selection_metric': 'validation NDCG@10 on fixed E5 candidate pool',
        'old_trained_weights_used': False, 'independent_of_research_history': False,
        'period_selection': 'fixed calendar dates, not selected by observed holdout quality',
    }


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def _validate_protocol(protocol):
    names = ['train_end', 'epoch_validation_end', 'decision', 'target_start', 'target_end']
    days = [pd.Timestamp(protocol[name]) for name in names]
    if any(pd.isna(day) or day.tzinfo is not None or day != day.normalize() for day in days):
        raise ValueError('Нужны календарные даты протокола')
    train, epoch, decision, start, end = days
    if (epoch - train != pd.Timedelta(days=14) or decision - epoch != pd.Timedelta(days=14)
            or start - decision != pd.Timedelta(days=1) or end - decision != pd.Timedelta(days=14)
            or protocol['window_days'] != 14):
        raise ValueError('Обучение, настройка и отложенное окно должны быть разделены полными 14 днями')
    dictionary_day = pd.Timestamp(protocol['dictionary_as_of'])
    if (pd.isna(dictionary_day) or dictionary_day.tzinfo is None
            or dictionary_day.tz_convert('UTC').tz_localize(None).normalize() > train):
        raise ValueError('Словарь должен быть известен до обучения')
    for name in ['search_seed', 'bootstrap_seed', 'max_views']:
        if type(protocol[name]) is not int or protocol[name] < 0:
            raise ValueError('Нужны неотрицательные целые параметры')
    for name in ['search_budget', 'search_repeats', 'epochs', 'patience', 'per_source',
                 'max_seeds', 'bootstrap_samples']:
        if type(protocol[name]) is not int or protocol[name] < 1:
            raise ValueError('Нужны положительные целые параметры')
    seeds = protocol['seeds']
    if (not isinstance(seeds, list) or len(seeds) < 3 or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
            or protocol['search_repeats'] < 3 or protocol['search_repeats'] > len(seeds)
            or protocol['search_budget'] > 4):
        raise ValueError('Для выбора нужны минимум три различных seed и допустимый бюджет')
    if (protocol['weights'] != WEIGHTS or protocol['old_trained_weights_used'] is not False
            or protocol['independent_of_research_history'] is not False
            or not np.isfinite(protocol['min_delta']) or protocol['min_delta'] < 0):
        raise ValueError('Нельзя менять зафиксированное сравнение или выдавать его за слепой тест')


def _complete_days(facts, start, end):
    required = pd.date_range(start, end)
    if not required.difference(pd.DatetimeIndex(facts.date.unique())).empty:
        raise ValueError('Нужны полные 14 дней истории, проверки эпох, настройки и целей')


def split_holdout(facts, protocol):
    _validate_protocol(protocol)
    facts = _facts(facts)
    _complete_days(facts, pd.Timestamp(protocol['train_end']) - pd.Timedelta(days=13),
                   protocol['target_end'])
    past = facts.loc[facts.date.le(protocol['decision'])].copy().reset_index(drop=True)
    heldout = facts.loc[facts.date.between(protocol['target_start'], protocol['target_end'])].copy()
    training = past.loc[past.date.le(protocol['train_end'])]
    return past, heldout.reset_index(drop=True), {
        'past_rows': len(past), 'heldout_rows': len(heldout), 'heldout_days': int(heldout.date.nunique()),
        'gradient_training_rows': len(training),
        'training_last_day': training.date.max().date().isoformat(),
        'last_selection_day': past.date.max().date().isoformat(),
        'ignored_later_rows': int(facts.date.gt(protocol['target_end']).sum()),
        'overlapping_dates': 0,
    }


class TemporalHoldoutStudy:
    """Получает только прошлое; целевые факты не входят в API обучения и выдачи."""

    def __init__(self, past, dictionary, catalog, vectors, protocol):
        _validate_protocol(protocol)
        past = _facts(past)
        if past.empty or past.date.gt(protocol['decision']).any():
            raise ValueError('В обучение переданы факты из отложенного периода')
        _complete_days(past, pd.Timestamp(protocol['train_end']) - pd.Timedelta(days=13),
                       protocol['decision'])
        snapshot, _ = dictionary_snapshot(dictionary, pd.Timestamp(protocol['dictionary_as_of']))
        if list(catalog) != sorted(snapshot.normquery.unique()):
            raise ValueError('Нужен полный каталог исторического словаря, без будущих запросов')
        vectors = np.asarray(vectors, dtype=np.float32)
        if (vectors.shape != (len(catalog), 384) or not np.isfinite(vectors).all()
                or (np.linalg.norm(vectors, axis=1) <= 0).any()):
            raise ValueError('Нужны E5-векторы в точном порядке каталога')
        self.protocol = protocol
        self.catalog = list(catalog)
        self.index = pd.Index(catalog)
        self.vectors = vectors
        self.text = lexical_index(catalog)
        self.daily, self.mapping = aggregate_clusters(past, snapshot)
        self.config = DormantConfig(pd.Timestamp(protocol['dictionary_as_of']).date(),
            pd.Timestamp(protocol['train_end']).date(),
            pd.Timestamp(protocol['epoch_validation_end']).date(),
            pd.Timestamp(protocol['decision']).date(), per_source=protocol['per_source'],
            max_views=protocol['max_views'], max_seeds=protocol['max_seeds'])
        self.parts = {name: self._part(protocol[field], labels=name != 'test') for name, field in [
            ('epoch_validation', 'train_end'), ('validation', 'epoch_validation_end'), ('test', 'decision')]}
        self.history = self.parts['epoch_validation']['history']
        self.score_cache = {}
        self.training_audit = {
            'last_day': protocol['train_end'], 'rows': len(self.history),
            'positive_edges': int(self.history.orders.gt(0).sum()),
            'skus': int(self.history.sku.nunique()),
            'queries': int(self.history.normquery.nunique()),
        }

    def _part(self, decision, *, labels):
        cutoff = pd.Timestamp(decision)
        start = cutoff - pd.Timedelta(days=13)
        history = period_pairs(self.daily, self.daily.date.min().date(), cutoff.date())
        recent = period_pairs(self.daily, start.date(), cutoff.date())
        active = set(recent.sku)
        daily = self.daily.loc[self.daily.date.between(start, cutoff)]
        demand = daily.drop_duplicates(['date', 'normquery']).groupby('normquery').cluster_searches.sum()
        base = HybridRetriever(self.catalog, history, demand, self.vectors, self.text).generate(
            self.config.per_source, max_history_views=self.config.max_views,
            traffic_history=recent, target_skus=active, max_seeds=self.config.max_seeds)
        targets = None
        if labels:
            future = period_pairs(self.daily, (cutoff + pd.Timedelta(days=1)).date(),
                                  (cutoff + pd.Timedelta(days=14)).date())
            targets = dormant_positives(future_positives(history, future), recent,
                                        self.config.max_views, active)
            if targets.empty:
                raise ValueError('Нет положительных целей для выбора эпох и настроек')
        return {'base': base, 'history': history, 'recent': recent, 'active': active,
                'demand': demand, 'targets': targets, 'decision': decision}

    def score(self, kind, model, pool, history):
        key = kind, id(pool), id(history)
        if key not in self.score_cache:
            names = sorted(set(pool.normquery) | set(history.loc[history.orders.gt(0), 'normquery']))
            positions = self.index.get_indexer(names)
            if (positions < 0).any():
                raise ValueError('Запрос вне зафиксированного каталога')
            scorer = profile_scorer(pool, history, names, weighted=kind == 'adapter')
            self.score_cache[key] = names, positions, scorer
        names, positions, scorer = self.score_cache[key]
        encoded = (apply_adapter(model, self.vectors[positions]) if kind == 'adapter'
                   else model.query_vectors(names)[0])
        return scorer(encoded)

    def train(self, kind, parameters, seed, output):
        part = self.parts['epoch_validation']

        def validation(model):
            values = self.score(kind, model, part['base'], part['history'])
            return ranking_metrics(part['base'][['sku', 'normquery']].assign(score=values),
                                   part['targets'])['ndcg@10']

        options = dict(epochs=self.protocol['epochs'], seed=seed,
            train_end=self.protocol['train_end'], validation_callback=validation,
            patience=self.protocol['patience'], min_delta=self.protocol['min_delta'],
            selection_metric='NDCG@10: 08.08–21.08.2026')
        started = perf_counter()
        if kind == 'adapter':
            model, audit = fit_adapter(self.history, self.catalog, self.vectors,
                                       device='cpu', **parameters, **options)
            save_adapter(model, output)
        else:
            model, audit = fit_graph(self.history, self.catalog, **parameters, **options)
            save_graph(model, output)
        part = self.parts['validation']
        score = self.score(kind, model, part['base'], part['history'])
        metrics = ranking_metrics(part['base'][['sku', 'normquery']].assign(score=score), part['targets'])
        return {'model': kind, 'parameters': parameters, 'seed': seed,
                'validation_ndcg': metrics['ndcg@10'], 'validation_recall': metrics['recall@10'],
                'seconds': perf_counter() - started, 'audit': audit}


def _load_model(kind, path):
    return load_adapter(path) if kind == 'adapter' else load_graph(path)


def fit_predict_holdout(past, dictionary, catalog, vectors, protocol, output):
    """Обучает с нуля и фиксирует все выдачи; параметра с целевыми заказами здесь нет."""
    output = Path(output)
    if output.exists():
        raise ValueError('Пакет уже существует; перезаписывать модели и выдачу нельзя')
    study = TemporalHoldoutStudy(past, dictionary, catalog, vectors, protocol)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix='.holdout-') as temporary:
        folder = Path(temporary) / 'packet'
        folder.mkdir()
        write_json(folder / 'protocol.json', protocol)
        runs, paths = [], {}
        search_seeds = protocol['seeds'][:protocol['search_repeats']]
        for kind in ['adapter', 'graph']:
            configurations = random_configurations(SPACE[kind], protocol['search_budget'], protocol['search_seed'])
            for index, parameters in enumerate(configurations):
                for seed in search_seeds:
                    relative = f'{kind}_{index}_{seed}' + ('.pt' if kind == 'adapter' else '')
                    print(f'Переобучение {kind}, конфигурация {index + 1}/{len(configurations)}, seed {seed}', flush=True)
                    row = study.train(kind, parameters, seed, folder / relative)
                    runs.append(row)
                    paths[kind, _digest(parameters), seed] = relative
        search = summarize_search(runs)
        selected_models = {}
        for seed in protocol['seeds']:
            selected_models[str(seed)] = {}
            for kind in ['adapter', 'graph']:
                parameters = search['selected'][kind]
                key = kind, _digest(parameters), seed
                if key not in paths:
                    relative = f'{kind}_selected_{seed}' + ('.pt' if kind == 'adapter' else '')
                    print(f'Переобучение выбранного {kind}, seed {seed}', flush=True)
                    runs.append(study.train(kind, parameters, seed, folder / relative))
                    paths[key] = relative
                selected_models[str(seed)][kind] = paths[key]
        write_json(folder / 'selection.json', {'search': search, 'selected_models': selected_models,
            'last_selection_day': protocol['decision'], 'targets_available_to_selection': False})
        part = study.parts['test']
        frames = []
        columns = ['sku', 'normquery', 'embedding_mean', 'graph_score', 'lexical_mean', 'log_cluster_orders']
        for seed in protocol['seeds']:
            print(f'Полный каталог: построение кандидатов, seed {seed}', flush=True)
            adapter = _load_model('adapter', folder / selected_models[str(seed)]['adapter'])
            graph = _load_model('graph', folder / selected_models[str(seed)]['graph'])
            adapted = apply_adapter(adapter, study.vectors)
            graph_encoded = graph.query_vectors(study.catalog)[0]
            pool = enrich_candidates(part['base'], part['history'], part['recent'], part['demand'],
                study.catalog, study.vectors, adapted, graph_encoded, study.text, study.config, part['active'])
            frames.append(pool[columns])
            del adapted, graph_encoded, pool, adapter, graph
        common = pd.concat(frames, ignore_index=True).drop_duplicates(['sku', 'normquery']).sort_values(
            ['sku', 'normquery']).reset_index(drop=True)
        del frames
        for seed in protocol['seeds']:
            print(f'Фиксация общей выдачи, seed {seed}', flush=True)
            pool = common.copy()
            for kind, column in [('adapter', 'adapter_score'), ('graph', 'lightgcn_score')]:
                model = _load_model(kind, folder / selected_models[str(seed)][kind])
                pool[column] = study.score(kind, model, common, part['history'])
            pool.to_csv(folder / f'predictions_{seed}.csv', index=False, float_format='%.17g')
        part['history'].to_csv(folder / 'history.csv', index=False)
        part['recent'].to_csv(folder / 'recent.csv', index=False)
        write_json(folder / 'context.json', {'active': sorted(part['active'])})
        manifest = {'schema_version': 1, 'protocol_sha256': _digest(protocol),
            'training_performed': True, 'old_trained_weights_used': False,
            'targets_read_during_training': False, 'seeds': protocol['seeds'],
            'catalog_clusters': len(catalog), 'candidate_pairs': len(common),
            'active_skus': len(part['active']), 'training': study.training_audit,
            'selection': {'last_day': protocol['decision']}, 'search': search,
            'selected_models': selected_models, 'training_runs': runs,
            'predictions_frozen_at': pd.Timestamp.now(tz='UTC').isoformat(),
            'files': {str(path.relative_to(folder)): sha256(path)
                      for path in sorted(folder.rglob('*')) if path.is_file()}}
        write_json(folder / 'manifest.json', manifest)
        folder.replace(output)
    return manifest


def verify_packet(folder, protocol):
    _validate_protocol(protocol)
    folder = Path(folder).resolve()
    manifest = json.loads((folder / 'manifest.json').read_text())
    if manifest['protocol_sha256'] != _digest(protocol):
        raise ValueError('Протокол выдачи изменён')
    for name, digest in manifest['files'].items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder) or sha256(path) != digest:
            raise ValueError('Контрольная сумма модели или выдачи не совпала')
    if manifest['seeds'] != protocol['seeds']:
        raise ValueError('Нельзя отбирать seed по результату теста')
    return manifest


def _read_pairs(path):
    return pd.read_csv(path, dtype={'sku': 'string', 'normquery': 'string'},
                       keep_default_na=False, float_precision='round_trip')


def _sku_recalls(pool, targets, scores):
    counts = targets.groupby('sku').size().sort_index()
    ordered = pool[['sku', 'normquery']].assign(score=scores).sort_values(
        ['sku', 'score', 'normquery'], ascending=[True, False, True], kind='stable').groupby('sku').head(10)
    hits = ordered.merge(targets, on=['sku', 'normquery'], how='inner').groupby('sku').size()
    return (hits.reindex(counts.index, fill_value=0) / counts).to_numpy(dtype=float)


def _interval(baseline, system, n_targets, protocol):
    result = {'n_skus': len(baseline), 'n_targets': int(n_targets),
              'repeats': protocol['bootstrap_samples'], 'seed': protocol['bootstrap_seed'],
              'method': 'paired SKU bootstrap of mean recall over every fixed seed'}
    names = ['baseline', 'system', 'difference', 'baseline_lower', 'baseline_upper',
             'system_lower', 'system_upper', 'lower', 'upper']
    if not len(baseline):
        return {**result, **dict.fromkeys(names)}
    matrix = np.column_stack([baseline, system])
    indices = np.random.default_rng(protocol['bootstrap_seed']).integers(
        0, len(matrix), size=(protocol['bootstrap_samples'], len(matrix)))
    samples = matrix[indices].mean(axis=1)
    result.update(baseline=float(matrix[:, 0].mean()), system=float(matrix[:, 1].mean()),
                  difference=float((matrix[:, 1] - matrix[:, 0]).mean()))
    for index, prefix in [(0, 'baseline_'), (1, 'system_')]:
        result[prefix + 'lower'], result[prefix + 'upper'] = map(float, np.quantile(samples[:, index], [.025, .975]))
    result['lower'], result['upper'] = map(float, np.quantile(samples[:, 1] - samples[:, 0], [.025, .975]))
    return result


def evaluate_holdout(folder, heldout, dictionary, protocol):
    folder = Path(folder)
    manifest = verify_packet(folder, protocol)
    heldout = _facts(heldout)
    if heldout.empty or not heldout.date.between(protocol['target_start'], protocol['target_end']).all():
        raise ValueError('Факты выходят за целевое отложенное окно')
    _complete_days(heldout, protocol['target_start'], protocol['target_end'])
    snapshot, _ = dictionary_snapshot(dictionary, pd.Timestamp(protocol['dictionary_as_of']))
    daily, mapping = aggregate_clusters(heldout, snapshot)
    future = period_pairs(daily, pd.Timestamp(protocol['target_start']).date(), pd.Timestamp(protocol['target_end']).date())
    history, recent = [_read_pairs(folder / f'{name}.csv') for name in ['history', 'recent']]
    active = set(json.loads((folder / 'context.json').read_text())['active'])
    targets = dormant_positives(future_positives(history, future), recent, protocol['max_views'], active)
    seed_rows, ablations, recalls = [], [], {name: [] for name in WEIGHTS}
    for seed in protocol['seeds']:
        pool = _read_pairs(folder / f'predictions_{seed}.csv')
        metrics = {}
        for name, weights in WEIGHTS.items():
            scores = fusion_scores(pool, weights)
            metrics[name] = ranking_metrics(pool[['sku', 'normquery']].assign(score=scores), targets)
            recalls[name].append(_sku_recalls(pool, targets, scores))
        ablations.append(metrics)
        baseline, system = metrics[BASELINE]['recall@10'], metrics[SYSTEM]['recall@10']
        seed_rows.append({'seed': seed, 'baseline': baseline, 'system': system,
                          'difference': system - baseline if len(targets) else None})
    mean_recalls = {name: np.mean(values, axis=0) for name, values in recalls.items()}
    interval = _interval(mean_recalls[BASELINE], mean_recalls[SYSTEM], len(targets), protocol)
    control_interval = _interval(mean_recalls[CONTROL], mean_recalls[SYSTEM], len(targets), protocol)
    skus = targets.groupby('sku').size().sort_index().index
    sold = history.loc[history.orders.gt(0)].groupby('sku').size().reindex(skus, fill_value=0)
    slices = []
    for label, lower, upper in [('0 кластеров', 0, 0), ('1–5 кластеров', 1, 5),
                                ('6–20 кластеров', 6, 20), ('21+ кластеров', 21, np.inf)]:
        selected = sold.between(lower, upper).to_numpy()
        count = int(targets.sku.isin(skus[selected]).sum())
        slices.append({'history': label, **_interval(mean_recalls[BASELINE][selected],
                                                    mean_recalls[SYSTEM][selected], count, protocol)})
    assert sum(row['n_targets'] for row in slices) == len(targets)
    means = {name: {key: (float(np.mean([row[name][key] for row in ablations]))
                         if row_value is not None else None)
                   for key, row_value in ablations[0][name].items()} for name in WEIGHTS}
    gains = np.asarray([row['difference'] for row in seed_rows], dtype=float) if len(targets) else np.array([])
    not_found = ~pd.MultiIndex.from_frame(targets).isin(pd.MultiIndex.from_frame(pool[['sku', 'normquery']]))
    return {'schema_version': 1,
        'status': 'evaluated_retrospective_holdout' if len(targets) else 'no_positive_targets',
        'evaluation_kind': protocol['evaluation_kind'], 'independent_of_research_history': False,
        'test_excluded_from_training': True, 'old_trained_weights_used': False,
        'training_performed': True, 'hyperparameter_selection_performed': True,
        'train_end': protocol['train_end'], 'epoch_validation_end': protocol['epoch_validation_end'],
        'decision': protocol['decision'], 'target_start': protocol['target_start'],
        'target_end': protocol['target_end'], 'target_days': 14,
        'catalog_clusters': manifest['catalog_clusters'], 'candidate_pairs': manifest['candidate_pairs'],
        'active_skus': manifest['active_skus'], 'training': manifest['training'],
        'search': manifest['search'], 'seeds': seed_rows,
        'seed_summary': {'count': len(seed_rows), 'mean_gain': float(gains.mean()) if len(gains) else None,
            'std_gain': float(gains.std(ddof=1)) if len(gains) else None,
            'wins': int((gains > 0).sum()), 'minimum_gain': float(gains.min()) if len(gains) else None,
            'maximum_gain': float(gains.max()) if len(gains) else None},
        'interval': interval, 'channel_count_interval': control_interval, 'ablation': means,
        'slices': slices, 'targets_not_retrieved': int(not_found.sum()),
        'normalization': {key: mapping[key] for key in ['input_rows', 'orders', 'covered_orders', 'order_coverage']},
        'training_runs': [{key: row[key] for key in ['model', 'parameters', 'seed', 'validation_ndcg',
                                                   'validation_recall', 'seconds', 'audit']}
                          for row in manifest['training_runs']],
        'prediction_packet_sha256': sha256(folder / 'manifest.json'),
        'predictions_frozen_at': manifest['predictions_frozen_at'],
        'evaluated_at': pd.Timestamp.now(tz='UTC').isoformat(),
        'effect_confirmed_on_window': bool(len(gains) and interval['lower'] > 0 and (gains > 0).all())}


def run_retrospective_holdout(root, source_record):
    root = Path(root).resolve()
    report_path = root / 'reports/notebook_retrospective_holdout.json'
    if source_record['status'] == 'saved_public_report':
        if not report_path.is_file():
            raise RuntimeError('Для переобучения нужны приватные данные; сохранённого расчёта нет')
        print('Отложенная проверка: показан сохранённый публичный результат, обучение здесь не запускалось.')
        return json.loads(report_path.read_text())
    protocol = default_protocol()
    config = json.loads((root / 'config.json').read_text())
    source = (root / source_record['snapshot_path']).resolve()
    manifest_path = source / 'source_manifest.json'
    if not source.is_relative_to(root) or sha256(manifest_path) != source_record['manifest_sha256']:
        raise ValueError('Снимок источника изменён')
    source_manifest = json.loads(manifest_path.read_text())
    for name, digest in source_manifest['files'].items():
        path = (source / name).resolve()
        if not path.is_relative_to(source) or sha256(path) != digest:
            raise ValueError('Контрольная сумма входных фактов не совпала')
    inputs = {name: (root / config['inputs'][name]).resolve() for name in ['dictionary', 'embeddings']}
    if any(not path.is_relative_to(root) for path in inputs.values()):
        raise ValueError('Входы должны находиться в проекте')
    code = ['retrospective_holdout.py', 'cluster_data.py', 'cluster_recommender.py',
            'dormant_learning.py', 'full_history_recommender.py', 'graph_learning.py',
            'semantic_adapter.py', 'training_validation.py', 'notebook_research.py',
            'research_diagnostics.py', 'independent_check.py']
    signature = {'source_manifest_sha256': sha256(manifest_path), 'protocol_sha256': _digest(protocol),
        'inputs': {name: sha256(path) for name, path in inputs.items()},
        'code': {name: sha256(root / 'src' / name) for name in code},
        'requirements_sha256': sha256(root / 'requirements.txt')}
    folder = root / 'data' / f'retrospective_holdout_{_digest(signature)[:16]}'
    if report_path.is_file() and folder.is_dir() and (folder / 'evaluation_manifest.json').is_file():
        saved = json.loads(report_path.read_text())
        if saved.get('provenance') == signature:
            verify_packet(folder, protocol)
            evaluation = json.loads((folder / 'evaluation_manifest.json').read_text())
            if (sha256(report_path) != evaluation['result_sha256']
                    or sha256(folder / 'result.json') != evaluation['result_sha256']):
                raise ValueError('Сохранённый результат отложенной проверки изменён')
            print('Переобучение и отложенная проверка воспроизведены; проверены данные, код, веса и выдачи.')
            return saved
    facts = pd.read_csv(source / 'facts.csv', dtype={'sku': 'string', 'query': 'string'}, keep_default_na=False)
    past, heldout, audit = split_holdout(facts, protocol)
    del facts
    dictionary = pd.read_csv(inputs['dictionary'], keep_default_na=False)
    if not folder.exists():
        snapshot, _ = dictionary_snapshot(dictionary, pd.Timestamp(protocol['dictionary_as_of']))
        catalog = sorted(snapshot.normquery.unique())
        vectors, _ = load_embeddings(inputs['embeddings'], catalog)
        print('Период 05–18.09 исключён. Обучение с нуля; выбор настроек только до 04.09.', flush=True)
        fit_predict_holdout(past, dictionary, catalog, vectors, protocol, folder)
        del vectors
    verify_packet(folder, protocol)
    print('Все выдачи сохранены. Только теперь считаются метрики отложенного периода.', flush=True)
    result = evaluate_holdout(folder, heldout, dictionary, protocol)
    result.update(provenance=signature, split_audit=audit, packet_path=str(folder.relative_to(root)))
    write_json(folder / 'result.json', result)
    write_json(folder / 'evaluation_manifest.json', {'result_sha256': sha256(folder / 'result.json'),
        'packet_sha256': sha256(folder / 'manifest.json'), 'protocol_sha256': _digest(protocol),
        'source_manifest_sha256': sha256(manifest_path)})
    write_json(report_path, result)
    return result
