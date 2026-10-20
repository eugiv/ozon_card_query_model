import ast
import json
from math import isclose
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_reader_text_has_no_development_chatter():
    notebook = json.loads((ROOT / 'recommender_demo.ipynb').read_text())
    text = '\n'.join(''.join(cell['source']) for cell in notebook['cells'])
    for fragment in ['не к четырём октябрьским дням', 'без ручных переключателей',
                     'не переименовывается', 'Экспертная разметка не является условием запуска',
                     'Историческое наблюдение', 'Для повторного исследования начните']:
        assert fragment.casefold() not in text.casefold(), fragment


def test_snapshot_loading_and_evaluation_are_not_duplicated():
    notebook = json.loads((ROOT / 'recommender_demo.ipynb').read_text())
    code = '\n'.join(''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code')
    calls = [node.func.id for node in ast.walk(ast.parse(code))
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    assert calls.count('refresh_source_snapshot') == 1
    assert calls.count('run_closed_validation') == 1
    assert 'reload' not in calls


def test_holdout_presentation_keeps_estimates_and_uncertainty():
    notebook = json.loads((ROOT / 'recommender_demo.ipynb').read_text())
    report = json.loads((ROOT / 'reports/notebook_retrospective_holdout.json').read_text())
    cell = next(cell for cell in notebook['cells'] if cell['id'] == '7c6ce2ca083c290e')
    rendered = []
    for output in cell['outputs']:
        value = output.get('data', {}).get('text/markdown', '')
        rendered.append(value if isinstance(value, str) else ''.join(value))
    text = '\n'.join(rendered)
    for key in ['baseline', 'system']:
        assert f"{100 * report['interval'][key]:.2f}%" in text
    for key in ['difference', 'lower', 'upper']:
        assert f"{100 * report['interval'][key]:+.2f}" in text
    assert '95%-й интервал' in text
    assert f"{report['seed_summary']['wins']} из {report['seed_summary']['count']}" in text
    markdown = '\n'.join(''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'markdown')
    assert 'ретроспективная отложенная проверка' in markdown
    assert 'устойчивое преимущество не подтверждено' in markdown


def test_notebook_runs_complete_holdout_automatically_and_explains_temporal_split():
    notebook = json.loads((ROOT / 'recommender_demo.ipynb').read_text())
    code = '\n'.join(''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code')
    markdown = '\n'.join(''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'markdown')
    calls = [node for node in ast.walk(ast.parse(code)) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == 'run_retrospective_holdout']
    assert len(calls) == 1
    cell = next(''.join(cell['source']) for cell in notebook['cells']
                if cell['cell_type'] == 'code' and 'heldout_report = run_retrospective_holdout' in ''.join(cell['source']))
    for fragment in ['test_excluded_from_training', 'old_trained_weights_used', 'overlapping_dates',
                     'channel_count_interval', "['slices']", "['seeds']", 'validation_per_epoch']:
        assert fragment in cell
    for fragment in ['05–18.09.2026', 'До 07.08', '08–21.08', '22.08–04.09',
                     'random search', 'ретроспективная отложенная проверка']:
        assert fragment in markdown or fragment in markdown.lower()
    assert '#holdout-check' in ''.join(notebook['cells'][0]['source'])
    assert 'RUN_HOLDOUT' not in code and 'RETRAIN_HOLDOUT' not in code


def test_research_setup_imports_retriever_before_recomputing_without_cache():
    notebook = json.loads((ROOT / 'recommender_demo.ipynb').read_text())
    setup = next(''.join(cell['source']) for cell in notebook['cells']
                 if cell['id'] == '6e85e91b44cda9e0')
    imports = {alias.name for node in ast.walk(ast.parse(setup))
               if isinstance(node, ast.ImportFrom) and node.module == 'cluster_recommender'
               for alias in node.names}
    assert 'HybridRetriever' in imports
    for module, required in [
        ('semantic_adapter', {'fit_adapter', 'apply_adapter'}),
        ('graph_learning', {'fit_graph'}),
        ('full_history_recommender', {'enrich_candidates'}),
    ]:
        available = {alias.name for node in ast.walk(ast.parse(setup))
                     if isinstance(node, ast.ImportFrom) and node.module == module
                     for alias in node.names}
        assert required.issubset(available)
    assert setup.index('from cluster_recommender import') < setup.index('if RECOMPUTE_RESEARCH:')


def test_saved_holdout_report_does_not_select_a_seed_or_hide_missed_targets():
    report = json.loads((ROOT / 'reports/notebook_retrospective_holdout.json').read_text())
    assert report['status'] == 'evaluated_retrospective_holdout'
    assert report['target_start'] == '2026-09-05' and report['target_end'] == '2026-09-18'
    assert report['test_excluded_from_training'] and report['training_performed']
    assert report['old_trained_weights_used'] is False
    assert report['independent_of_research_history'] is False
    assert report['catalog_clusters'] == 477579
    assert [row['seed'] for row in report['seeds']] == [7, 17, 27, 37, 47]
    assert report['seed_summary']['count'] == 5
    assert report['interval']['n_targets'] > 0
    assert sum(row['n_targets'] for row in report['slices']) == report['interval']['n_targets']
    assert sum(row['n_skus'] for row in report['slices']) == report['interval']['n_skus']
    assert report['channel_count_interval']['n_targets'] == report['interval']['n_targets']
    assert isclose(report['interval']['difference'], report['seed_summary']['mean_gain'], abs_tol=1e-14)
    assert all(row['seeds'] == [7, 17, 27] for row in report['search']['configurations'])
    assert '!/reports/notebook_retrospective_holdout.json' in (ROOT / '.gitignore').read_text()

    def aggregates_only(value):
        if isinstance(value, dict):
            assert not {'sku', 'query', 'normquery', 'password', 'host', 'unmapped_queries'} & value.keys()
            for item in value.values():
                aggregates_only(item)
        elif isinstance(value, list):
            for item in value:
                aggregates_only(item)
        elif isinstance(value, str):
            assert '/Users/' not in value

    aggregates_only(report)
