import hashlib
import json
from pathlib import Path

import nbformat
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_notebook_leads_with_unconfirmed_effect_not_historical_gain():
    notebook = nbformat.read(ROOT / 'recommender_demo.ipynb', as_version=4)
    introduction = notebook.cells[0].source
    assert 'устойчивое преимущество не подтверждено' in introduction.lower()
    assert '+3,79' not in introduction
    code = '\n'.join(cell.source for cell in notebook.cells if cell.cell_type == 'code')
    assert 'random_configurations(' in code
    assert 'summarize_search(' in code
    assert 'paired_recall_intervals(' in code
    assert 'perturb_history(' in code


def test_four_days_are_not_reported_as_fourteen_day_effect():
    notebook = nbformat.read(ROOT / 'recommender_demo.ipynb', as_version=4)
    sources = '\n'.join(cell.source for cell in notebook.cells)
    assert "['test', 'new_days']" not in sources
    assert "('new_days', '4 новых дня')" not in sources
    assert 'Неполные целевые окна не оцениваются' in sources
    assert "heldout_report['target_days'] == 14" in sources
    assert "current_validation['target_days'] == 14" in sources


@pytest.fixture
def review_report():
    return json.loads((ROOT / 'reports/notebook_review.json').read_text())


def test_real_random_search_reports_all_seed_runs_and_selection_rule(review_report):
    search = review_report['random_search']
    assert search['seeds'] == [7, 17, 27]
    assert search['selection_period'] == '2026-09-05/2026-09-18'
    assert search['diagnostic_period_used_for_selection'] is False
    assert search['selection_rule'] == 'within_one_std_choose_simpler'
    assert search['search_seed'] == 20261020
    for kind in ['adapter', 'graph']:
        trials = [row for row in search['configurations'] if row['model'] == kind]
        assert len(trials) >= 4
        assert sum(row['selected'] for row in trials) == 1
        for row in trials:
            assert row['seeds'] == [7, 17, 27]
            assert row['std_ndcg'] >= 0
            assert row['std_recall'] >= 0
            assert row['mean_seconds'] > 0
    assert len(search['runs']) == 3 * len(search['configurations'])


def test_saved_search_summary_is_recomputed_from_every_run(review_report):
    from research_diagnostics import summarize_search

    search = review_report['random_search']
    recomputed = summarize_search(search['runs'])
    assert recomputed['configurations'] == search['configurations']
    assert recomputed['selected'] == search['selected']


def test_structural_noise_has_matched_clean_seeds_and_actual_edge_changes(review_report):
    rows = review_report['structural_noise']
    assert {row['scenario'] for row in rows} == {'drop_edges', 'false_edges', 'rewire_edges'}
    assert len(rows) == 9
    for row in rows:
        assert row['seed'] in [7, 17, 27]
        assert row['removed_edges'] + row['added_edges'] > 0
        assert row['graph_sold_edges'] == row['edges_after']
        assert row['changed_fraction'] > 0
        assert row['difference_from_clean_pp'] == pytest.approx(
            row['recall_pp'] - row['clean_recall_pp'])
        assert row['targets_changed'] is False


def test_slice_intervals_keep_every_target_and_report_negative_slices(review_report):
    rows = review_report['slices']
    assert sum(row['n_skus'] for row in rows) == 121
    assert sum(row['n_targets'] for row in rows) == 260
    nonempty = [row for row in rows if row['n_skus']]
    assert len(nonempty) == 4
    assert sum(row['difference'] < 0 for row in nonempty) == 2
    cold = next(row for row in rows if row['history'] == '0 кластеров')
    assert cold['n_skus'] == 12 and cold['n_targets'] == 25
    assert cold['baseline'] == cold['system'] == 0
    for row in nonempty:
        assert row['lower'] <= row['upper']
        assert row['baseline_lower'] <= row['baseline_upper']
        assert row['system_lower'] <= row['system_upper']
        assert row['repeats'] == 2000


def test_weight_search_cannot_use_diagnostic_test_to_choose_weights(review_report):
    study = review_report['weight_search']
    assert study['policy_for_next_test'] == 'fixed_equal_weights_no_search'
    assert study['diagnostic_test_used_for_selection'] is False
    rows = study['diagnostics']
    assert {row['trials'] for row in rows} >= {1, 35, 286}
    for row in rows:
        assert row['validation_test_gap'] == pytest.approx(row['validation_ndcg'] - row['test_ndcg'])


def test_replacement_protocol_checks_original_full_model_without_backdating(review_report):
    protocol = json.loads((ROOT / 'reports/notebook_full_protocol.json').read_text())
    assert protocol['catalog_clusters'] == 477579
    assert protocol['model_identity'] == 'historical_full_catalog_four_channels'
    assert protocol['weights']['Полные четыре'] == {
        'pretrained_embeddings': .25, 'co_sales': .25, 'semantic_adapter': .25, 'lightgcn': .25}
    assert protocol['window_days'] == 14
    assert protocol['decision'] > protocol['frozen_at'][:10]
    assert protocol['no_tuning_on_future'] is True
    assert protocol['supersedes'] == 'reports/notebook_research_protocol.json'
    assert 'final_adapter' not in protocol['artifacts']['adapter']['path']
    assert 'final_graph' not in protocol['artifacts']['graph']['path']
    assert review_report['prospective']['status'] == 'awaiting_unseen_data'


def test_new_artifact_identity_and_code_hashes_are_anchored(review_report):
    from notebook_review import HISTORICAL_DIGESTS

    path = ROOT / 'reports/notebook_full_protocol.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == review_report['full_protocol_sha256']
    protocol = json.loads(path.read_text())
    for name, digest in HISTORICAL_DIGESTS.items():
        assert protocol['artifacts'][name]['sha256'] == digest
    for path, digest in (review_report['code_sha256'] | protocol['code_sha256']).items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    checked = review_report['frozen_model_check']
    assert checked['original_artifacts_verified'] is True
    assert checked['e5_shape'] == checked['adapter_shape'] == [477579, 384]
    assert checked['graph_shape'] == [477579, 32]
    assert checked['training_performed'] is False and checked['future_evaluation_performed'] is False
    for path in [ROOT / 'reports/notebook_review.json', ROOT / 'reports/notebook_full_protocol.json']:
        assert '/Users/' not in path.read_text() and 'DBT_CLICKHOUSE' not in path.read_text()


def test_public_review_artifacts_are_versionable_without_exposing_private_models():
    rules = (ROOT / '.gitignore').read_text().splitlines()
    assert '!/reports/notebook_review.json' in rules
    assert '!/reports/notebook_full_protocol.json' in rules
    assert '/reports/*' in rules and 'data/' in rules
    assert '*.npz' in rules and '*.pt' in rules