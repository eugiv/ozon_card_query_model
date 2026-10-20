import json
from datetime import date
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from cluster_data import aggregate_clusters, dictionary_snapshot, future_positives, period_pairs, sha256
from cluster_recommender import HybridRetriever, lexical_index, load_embeddings
from dormant_learning import ranking_metrics
from full_history_recommender import DormantConfig, dormant_positives, enrich_candidates, graph_vectors
from graph_learning import fit_graph, load_graph
from notebook_research import profile_scorer, prospective_status
from semantic_adapter import apply_adapter, fit_adapter


HISTORICAL_DIGESTS = {
    'e5': '35fda89e78b324c41035b2aa9e4d0f3fcfb0029b2c076561ca1a8077fb9225db',
    'adapter': '4c09604bfb7014d6656a0c45a0b3230d65ac6c1513c29dd1577ba0daeca10a7c',
    'adapter_weights': 'e5d3b11d6d75ec6d6f2b4b3309852e3386acd6cbbfd5d35684b8929fc5a1998b',
    'graph': '461d24e08d4ec1b290f3155783e5f99f9527d1250899469256aaef619a88f81d',
    'graph_catalog': '0a729b02e8e4f7c72e9efba7d4d80d7c6b9148fff2d104e22beee6f43d4e0a9f',
}


class ReviewStudy:
    """Те же полные временные окна, но отдельный аудит без изменения старого отчёта."""

    def __init__(self, daily, catalog, vectors, source_days):
        self.daily = daily
        self.catalog = catalog
        self.vectors = vectors
        self.source_days = pd.DatetimeIndex(source_days).normalize()
        self.text = lexical_index(catalog)
        self.config = DormantConfig(date(2026, 6, 26), date(2026, 6, 26),
                                    date(2026, 9, 4), date(2026, 9, 18), per_source=50)
        self.parts = {name: self.split(day) for name, day in [
            ('epoch_validation', '2026-08-21'), ('validation', '2026-09-04'), ('test', '2026-09-18')]}
        self.history = self.parts['epoch_validation']['history']
        self.scorers = {(name, kind): profile_scorer(part['base'], part['history'], catalog,
                                                   weighted=kind == 'adapter')
                        for name, part in self.parts.items() for kind in ['adapter', 'graph']}
        self.cache = {}
        self.provenance = {}
        self.data_audit = {}

    @classmethod
    def from_project(cls, root):
        root = Path(root)
        config = json.loads((root / 'config.json').read_text())
        facts = pd.read_csv(root / config['inputs']['facts'],
                            dtype={'sku': 'string', 'query': 'string'}, keep_default_na=False)
        dictionary = pd.read_csv(root / config['inputs']['dictionary'], keep_default_na=False)
        snapshot, _ = dictionary_snapshot(dictionary, pd.Timestamp('2026-06-26T23:59:59.999999999Z'))
        daily, _ = aggregate_clusters(facts, snapshot)
        full_catalog = sorted(snapshot.normquery.unique())
        catalog = sorted(daily.loc[daily.date.le('2026-08-21'), 'normquery'].unique())
        full_vectors, _ = load_embeddings(root / config['inputs']['embeddings'], full_catalog)
        vectors = full_vectors[pd.Index(full_catalog).get_indexer(catalog)].copy()
        del full_vectors
        result = cls(daily, catalog, vectors, pd.to_datetime(facts.date).unique())
        result.provenance = {key: sha256(root / config['inputs'][key])
                             for key in ['facts', 'dictionary', 'embeddings']}
        result.data_audit = {'rows': len(facts), 'skus': int(facts.sku.nunique()),
            'catalog_clusters': len(catalog), 'full_catalog_clusters': len(full_catalog),
            'first_day': min(result.source_days).strftime('%Y-%m-%d'),
            'last_day': max(result.source_days).strftime('%Y-%m-%d'),
            'nulls': {key: int(value) for key, value in facts.isna().sum().items()},
            'dtypes': {key: str(value) for key, value in facts.dtypes.items()},
            'splits': [{'name': name, 'decision': part['decision'], 'target_days': 14,
                'target_skus': int(part['targets'].sku.nunique()), 'targets': len(part['targets']),
                'targets_outside_catalog': int((~part['targets'].normquery.isin(catalog)).sum())}
                for name, part in result.parts.items()]}
        return result

    def split(self, decision):
        cutoff = pd.Timestamp(decision)
        start = cutoff - pd.Timedelta(days=13)
        required = pd.date_range(start, cutoff + pd.Timedelta(days=14))
        if not required.difference(self.source_days).empty:
            raise ValueError('Неполное 14-дневное окно не оценивается')
        frame = self.daily.loc[self.daily.normquery.isin(self.catalog)]
        history = period_pairs(frame, frame.date.min().date(), cutoff.date())
        recent = period_pairs(frame, start.date(), cutoff.date())
        full_history = period_pairs(self.daily, self.daily.date.min().date(), cutoff.date())
        full_recent = period_pairs(self.daily, start.date(), cutoff.date())
        active = set(full_recent.sku)
        part = frame.loc[frame.date.between(start, cutoff)]
        demand = part.drop_duplicates(['date', 'normquery']).groupby('normquery').cluster_searches.sum()
        base = HybridRetriever(self.catalog, history, demand, self.vectors, self.text).generate(
            self.config.per_source, max_history_views=5, traffic_history=recent,
            target_skus=active, max_seeds=32)
        future = period_pairs(self.daily, (cutoff + pd.Timedelta(days=1)).date(),
                              (cutoff + pd.Timedelta(days=14)).date())
        targets = dormant_positives(future_positives(full_history, future), full_recent, 5, active)
        return {'base': base, 'history': history, 'recent': recent, 'demand': demand,
                'active': active, 'targets': targets, 'decision': decision}

    def train(self, kind, parameters, seed, *, history=None, epochs=40):
        if kind not in {'adapter', 'graph'}:
            raise ValueError('Неизвестная модель')
        key = kind, tuple(sorted(parameters.items())), seed, epochs
        if history is None and key in self.cache:
            return self.cache[key]
        training = self.history if history is None else history
        part = self.parts['epoch_validation']

        def validation(model):
            encoded = (apply_adapter(model, self.vectors) if kind == 'adapter'
                       else model.query_vectors(self.catalog)[0])
            scores = self.scorers['epoch_validation', kind](encoded)
            return ranking_metrics(part['base'][['sku', 'normquery']].assign(score=scores),
                                   part['targets'])['ndcg@10']

        options = dict(epochs=epochs, seed=seed, train_end='2026-08-21', validation_callback=validation,
                       patience=5, min_delta=.0001, selection_metric='NDCG@10: 22.08–04.09')
        started = perf_counter()
        if kind == 'adapter':
            model, audit = fit_adapter(training, self.catalog, self.vectors, device='cpu',
                                       **parameters, **options)
            encoded = apply_adapter(model, self.vectors)
        else:
            model, audit = fit_graph(training, self.catalog, **parameters, **options)
            encoded = model.query_vectors(self.catalog)[0]
        audit['seconds'] = perf_counter() - started
        if not np.isfinite(encoded).all():
            raise ValueError('Неконечные векторы')
        if history is None:
            self.cache[key] = encoded, audit
        return encoded, audit

    def metric(self, kind, encoded, period='validation'):
        part = self.parts[period]
        scores = self.scorers[period, kind](encoded)
        return ranking_metrics(part['base'][['sku', 'normquery']].assign(score=scores), part['targets'])

    def pool(self, encoded, period):
        part = self.parts[period]
        return enrich_candidates(part['base'], part['history'], part['recent'], part['demand'],
            self.catalog, self.vectors, encoded['adapter'], encoded['graph'], self.text, self.config, part['active'])

    def rescore(self, pool, encoded, period='test'):
        part = self.parts[period]
        result = pool.copy()
        result['adapter_score'] = profile_scorer(pool, part['history'], self.catalog, weighted=True)(encoded['adapter'])
        result['lightgcn_score'] = profile_scorer(pool, part['history'], self.catalog, weighted=False)(encoded['graph'])
        return result


def make_full_protocol(root, control_weights, *, frozen_at):
    """Фиксирует исходную full-catalog evaluation, не финальное дообучение."""
    root = Path(root)
    frozen = pd.Timestamp(frozen_at)
    if not isinstance(frozen, pd.Timestamp) or frozen.tzinfo is None:
        raise ValueError('Нужна временная зона фиксации')
    if control_weights != {'pretrained_embeddings': .5, 'co_sales': .5}:
        raise ValueError('Веса базы не соответствуют историческому контролю')
    paths = {'e5': 'data/e5_small_expanded.npz',
        'adapter': 'reports/stage3_20261011_adapter/vectors.npz',
        'adapter_weights': 'reports/stage3_20261011_adapter/adapter.pt',
        'graph': 'reports/stage3_graph/graph.pt',
        'graph_catalog': 'reports/stage3_graph/catalog.json',
        'graph_manifest': 'reports/stage3_graph/manifest.json',
        'graph_metrics': 'reports/stage3_graph/metrics.json'}
    code = ['src/notebook_review.py', 'src/notebook_research.py', 'src/cluster_data.py',
            'src/cluster_recommender.py', 'src/full_history_recommender.py',
            'src/dormant_learning.py', 'src/graph_learning.py']
    config = json.loads((root / 'config.json').read_text())
    for name, digest in HISTORICAL_DIGESTS.items():
        if sha256(root / paths[name]) != digest:
            raise ValueError('Артефакт не совпадает с исходной исторической моделью')
    decision = (frozen.tz_convert('UTC').normalize() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    return {'schema_version': 2, 'frozen_at': frozen.isoformat(), 'decision': decision,
        'window_days': 14, 'history_days': 14, 'train_end': '2026-09-04', 'selection_end': '2026-09-18',
        'dictionary_as_of': '2026-06-26T23:59:59.999999999Z', 'catalog_clusters': 477579,
        'catalog_policy': 'entire fixed dictionary snapshot; identical to historical evaluation',
        'model_identity': 'historical_full_catalog_four_channels', 'seed': 7,
        'per_source': 150, 'max_views': 5, 'max_seeds': 32,
        'weights': {'Полные четыре': {channel: .25 for channel in [
            'pretrained_embeddings', 'co_sales', 'semantic_adapter', 'lightgcn']},
            'База: E5 + co-sales': control_weights},
        'artifacts': {name: {'path': path, 'sha256': sha256(root / path)} for name, path in paths.items()},
        'inputs': {key: {'path': config['inputs'][key], 'sha256': sha256(root / config['inputs'][key])}
                   for key in ['facts', 'dictionary']},
        'code_sha256': {path: sha256(root / path) for path in code},
        'primary_metric': 'macro recall@10', 'bootstrap_samples': 2000, 'bootstrap_seed': 7,
        'criterion': 'full minus baseline > 0 and paired SKU bootstrap 95% lower > 0',
        'no_tuning_on_future': True, 'supersedes': 'reports/notebook_research_protocol.json',
        'registration': 'local artifact only; no external preregistration',
        'limitation': 'one original seed; evaluates this frozen model, not stability of arbitrary retraining'}


def verify_full_protocol(root, protocol):
    root = Path(root).resolve()
    full_weights = {channel: .25 for channel in [
        'pretrained_embeddings', 'co_sales', 'semantic_adapter', 'lightgcn']}
    if (protocol.get('model_identity') != 'historical_full_catalog_four_channels'
            or protocol.get('catalog_clusters') != 477579
            or protocol['weights']['Полные четыре'] != full_weights
            or protocol['weights']['База: E5 + co-sales'] != {'pretrained_embeddings': .5, 'co_sales': .5}
            or protocol.get('seed') != 7 or protocol.get('per_source') != 150
            or protocol.get('max_views') != 5 or protocol.get('max_seeds') != 32
            or protocol.get('window_days') != 14 or protocol.get('no_tuning_on_future') is not True):
        raise ValueError('Протокол не соответствует исходной полной модели')
    for name, digest in HISTORICAL_DIGESTS.items():
        if protocol['artifacts'][name]['sha256'] != digest:
            raise ValueError('Артефакт не совпадает с исходной исторической моделью')
    for group in ['artifacts', 'inputs']:
        for record in protocol[group].values():
            path = root / record['path']
            if not path.resolve().is_relative_to(root) or sha256(path) != record['sha256']:
                raise ValueError('Зафиксированный вход или модель изменены')
    if any('final_' in record['path'] or 'final_graph' in record['path']
           for record in protocol['artifacts'].values()):
        raise ValueError('Финальное дообучение не является исторической моделью')
    for path, digest in protocol['code_sha256'].items():
        if not (root / path).resolve().is_relative_to(root) or sha256(root / path) != digest:
            raise ValueError('Код изменился после фиксации')
    prospective_status(protocol, [], model_training_end=protocol['train_end'], selection_end=protocol['selection_end'])


def load_full_frozen(root, dictionary, protocol):
    root = Path(root)
    verify_full_protocol(root, protocol)
    snapshot, _ = dictionary_snapshot(dictionary, pd.Timestamp(protocol['dictionary_as_of']))
    catalog = sorted(snapshot.normquery.unique())
    if len(catalog) != protocol['catalog_clusters']:
        raise ValueError('Сокращённый каталог не является исходной моделью')
    vectors, _ = load_embeddings(root / protocol['artifacts']['e5']['path'], catalog)
    adapted, metadata = load_embeddings(root / protocol['artifacts']['adapter']['path'], catalog)
    if pd.Timestamp(metadata['dictionary_as_of']) != pd.Timestamp(protocol['dictionary_as_of']):
        raise ValueError('Дата словаря адаптера не совпадает')
    graph_path = root / protocol['artifacts']['graph']['path']
    model = load_graph(graph_path.parent)
    if pd.Timestamp(model.train_end).date().isoformat() != protocol['train_end']:
        raise ValueError('Дата обучения графа не совпадает')
    return {'catalog': np.asarray(catalog), 'e5': vectors, 'adapter': adapted,
            'graph': graph_vectors(model, catalog)}