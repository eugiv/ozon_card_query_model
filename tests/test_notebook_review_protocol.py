import copy

import pytest

from notebook_review import HISTORICAL_DIGESTS, make_full_protocol, verify_full_protocol


@pytest.mark.parametrize('field,value', [('catalog_clusters', 18444), ('seed', 17),
    ('window_days', 4), ('per_source', 50), ('max_seeds', 12), ('no_tuning_on_future', False),
    ('model_identity', 'retrained_final_production')])
def test_full_protocol_rejects_substitution_before_reading_models(tmp_path, field, value):
    protocol = {'catalog_clusters': 477579, 'seed': 7, 'window_days': 14, 'per_source': 150,
        'max_seeds': 32, 'max_views': 5, 'no_tuning_on_future': True,
        'model_identity': 'historical_full_catalog_four_channels',
        'weights': {'Полные четыре': {'pretrained_embeddings': .25, 'co_sales': .25,
                                       'semantic_adapter': .25, 'lightgcn': .25},
                    'База: E5 + co-sales': {'pretrained_embeddings': .5, 'co_sales': .5}}}
    protocol[field] = value
    with pytest.raises(ValueError, match='исходной полной модели'):
        verify_full_protocol(tmp_path, protocol)


def test_full_protocol_rejects_zero_co_sales_and_changed_baseline(tmp_path):
    protocol = {'catalog_clusters': 477579, 'seed': 7, 'window_days': 14, 'per_source': 150,
        'max_seeds': 32, 'max_views': 5, 'no_tuning_on_future': True,
        'model_identity': 'historical_full_catalog_four_channels',
        'weights': {'Полные четыре': {'pretrained_embeddings': .25, 'co_sales': .25,
                                       'semantic_adapter': .25, 'lightgcn': .25},
                    'База: E5 + co-sales': {'pretrained_embeddings': .5, 'co_sales': .5}}}
    for name in protocol['weights']:
        altered = copy.deepcopy(protocol)
        altered['weights'][name]['co_sales'] = 0
        with pytest.raises(ValueError, match='исходной полной модели'):
            verify_full_protocol(tmp_path, altered)


@pytest.mark.parametrize('timestamp', ['2026-10-20', 'NaT'])
def test_protocol_creation_requires_real_timezone_aware_freeze(tmp_path, timestamp):
    with pytest.raises(ValueError, match='временная зона'):
        make_full_protocol(tmp_path, {'pretrained_embeddings': .5, 'co_sales': .5}, frozen_at=timestamp)


def test_original_model_digests_are_bound_not_just_filenames():
    assert set(HISTORICAL_DIGESTS) == {'e5', 'adapter', 'adapter_weights', 'graph', 'graph_catalog'}
    assert all(len(digest) == 64 for digest in HISTORICAL_DIGESTS.values())
    assert HISTORICAL_DIGESTS['adapter'] != HISTORICAL_DIGESTS['graph']