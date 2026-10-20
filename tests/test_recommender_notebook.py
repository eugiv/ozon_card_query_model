import ast
import copy
import json
import re
from pathlib import Path
from typing import Callable, cast

import nbformat
import numpy as np
import pandas as pd
import pytest
from IPython.display import display

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def notebook():
    return nbformat.read(ROOT / "recommender_demo.ipynb", as_version=4)


def test_notebook_is_valid_and_code_compiles(notebook):
    nbformat.validate(notebook)
    assert notebook.cells[0].cell_type == "markdown"
    assert "синтетические" in notebook.cells[0].source
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, "recommender_demo.ipynb", "exec")


def test_all_code_cells_have_saved_static_outputs(notebook):
    cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert len(cells) >= 10
    counts = []
    tables = 0
    for cell in cells:
        assert isinstance(cell.execution_count, int)
        counts.append(cell.execution_count)
        assert cell.outputs, cell.id
        for output in cell.outputs:
            assert output.output_type != "error"
            data = output.get("data", {})
            assert "application/vnd.jupyter.widget-view+json" not in data
            tables += "text/html" in data
    assert counts == list(range(1, len(cells) + 1))
    assert tables >= 10


def test_models_run_directly_in_kernel_and_private_replay_is_guarded(notebook):
    sources = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    calls = set()
    for source in sources:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                calls.add(node.func.id)
        assert "subprocess" not in source
        assert "os.system" not in source and "os.popen" not in source
        assert not any(isinstance(node, ast.Constant) and isinstance(node.value, str)
                       and Path(node.value).name == ".env" for node in ast.walk(tree))
    assert "load_dotenv" not in calls
    assert {"fit_adapter", "fit_graph", "fit_ranker", "fit_fusion", "ranking_metrics",
            "enrich_candidates", "annotate_recommendations", "recommend_from_release"} <= calls
    private_source = next(source for source in sources if "recommend_from_release(" in source)
    guard = ast.parse(private_source).body[0]
    assert isinstance(guard, ast.If)
    assert ast.unparse(guard.test) == "not RUN_PRIVATE_CHECK"
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id == "recommend_from_release"
               for statement in guard.orelse for node in ast.walk(statement))
    assert "del expected, replayed" in private_source


def test_saved_outputs_do_not_expose_local_paths(notebook):
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            rendered = str(dict(output))
            assert "/Users/" not in rendered
            assert "DBT_CLICKHOUSE" not in rendered
            assert "Traceback" not in rendered


def test_real_comparison_comes_first_with_saved_chart_and_both_outcomes(notebook):
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert "snapshot_metrics(" in code_cells[0].source
    assert "notebook_ranking_pairs.csv" in code_cells[0].source
    assert "run_retrospective_holdout(" in code_cells[1].source
    assert any("image/png" in output.get("data", {}) for output in code_cells[1].outputs)
    assert any("image/png" in output.get("data", {}) for output in code_cells[2].outputs)
    examples = code_cells[3]
    rendered = str(examples.outputs)
    assert "Выигрыш системы" in rendered
    assert "Потеря системы" in rendered
    assert 'полно' in rendered.lower() and '14' in str(examples.source)
    assert 'new_days' not in examples.source
    assert "gained.head(3)" in examples.source and "lost.head(3)" in examples.source


def test_research_covers_review_topics_without_claiming_independent_success(notebook):
    code = '\n'.join(cell.source for cell in notebook.cells if cell.cell_type == 'code')
    text = '\n'.join(cell.source for cell in notebook.cells if cell.cell_type == 'markdown')
    for required in ['data_overview(', 'semantic_review_summary(', 'train_network(',
                     'validation_callback=', 'patience=5', 'best_epoch',
                     'Четыре без обучения', 'training_noise', 'sensitivity',
                     'prospective_status(', 'run_independent_check(', 'full_protocol_sha256',
                     'verify_full_protocol(', 'random_configurations(', 'paired_recall_intervals(']:
        assert required in code, required
    assert 'устойчивое преимущество не подтверждено' in notebook.cells[0].source.lower()
    assert 'Категорийных меток нет' in text
    assert 'семантическая точность' in text.lower() or 'точность нормализации' in text.lower()
    assert 'pd.testing.assert_frame_equal(past_before_future_event, past_after_future_event)' in code
    assert '(lexical_index(rechecked_catalog) != text).nnz == 0' in code
    assert 'dictionary_snapshot(demo_dictionary' in code


def presentation_summary_function(notebook) -> Callable[[dict], dict]:
    source = next(cell.source for cell in notebook.cells
                  if cell.cell_type == 'code' and 'def repeat_result_summary(' in cell.source)
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'repeat_result_summary')
    namespace: dict[str, object] = {'np': np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<notebook summary>', 'exec'), namespace)
    return cast(Callable[[dict], dict], namespace['repeat_result_summary'])


def test_repeat_summary_uses_every_seed_and_reports_losses(notebook):
    summarize = presentation_summary_function(notebook)
    report = json.loads((ROOT / 'reports/notebook_research.json').read_text())
    result = summarize(report)
    scores = [row['Recall@10, %'] for row in report['seeds']]
    assert result['count'] == 5
    assert result['wins'] == 3 and result['losses'] == 2 and result['ties'] == 0
    assert result['mean'] == pytest.approx(np.mean(scores))
    assert result['std'] == pytest.approx(np.std(scores, ddof=1))
    assert result['gain'] == pytest.approx(result['mean'] - result['baseline'])
    assert result['minimum'] == min(scores) and result['maximum'] == max(scores)
    reordered = copy.deepcopy(report)
    reordered['seeds'].reverse()
    assert summarize(reordered) == pytest.approx(result)


def test_repeat_summary_does_not_turn_negative_results_into_improvement(notebook):
    summarize = presentation_summary_function(notebook)
    report = {'ablation': [{'Вариант': 'База: E5 + co-sales', 'Recall@10, %': 20}],
              'seeds': [{'seed': seed, 'Recall@10, %': value}
                        for seed, value in [(7, 10), (17, 20), (27, 15)]]}
    result = summarize(report)
    assert result['mean'] == 15 and result['gain'] == -5
    assert result['wins'] == 0 and result['losses'] == 2 and result['ties'] == 1
    report['ablation'].insert(0, {'Вариант': 'Другой вариант', 'Recall@10, %': 99})
    assert summarize(report) == result


@pytest.mark.parametrize('scores', [[10], [10, float('nan')], [10, float('inf')], [-1, 10], [10, 101]])
def test_repeat_summary_rejects_incomplete_or_invalid_metrics(notebook, scores: list[float]):
    summarize = presentation_summary_function(notebook)
    report = {'ablation': [{'Вариант': 'База: E5 + co-sales', 'Recall@10, %': 20}],
              'seeds': [{'seed': index, 'Recall@10, %': value} for index, value in enumerate(scores)]}
    with pytest.raises(ValueError):
        summarize(report)


def test_first_chart_includes_all_reported_methods_and_recomputed_control(notebook):
    code = [cell for cell in notebook.cells if cell.cell_type == 'code'][2]
    assert "summary_report['test']['models'].items()" in code.source
    assert "real_metrics['baseline']" in code.source
    assert "real_metrics['system']" in code.source
    assert "set(full_comparison.index)" in code.source
    rendered = str(code.outputs)
    for name in ['TF-IDF', 'E5', 'Совместные продажи', 'Адаптер', 'LightGCN', 'Популярность', 'CatBoost',
                 'База: E5 + совместные продажи', 'Полная система']:
        assert name in rendered
    assert 'Продажи не измерялись' in rendered


def test_saved_repeat_presentation_is_explicit_about_scope_and_all_runs(notebook):
    cell = next(cell for cell in notebook.cells if cell.cell_type == 'code'
                and 'Разброс по пяти seed' in cell.source)
    assert 'repeat_result_summary(research)' in cell.source
    assert any('image/png' in output.get('data', {}) for output in cell.outputs)
    rendered = str(cell.outputs)
    assert 'Среднее всех' in rendered and 'не ансамбль пяти моделей' in rendered
    assert 'Диагностика' in rendered and 'сокращённый каталог' in rendered
    assert '19.09–02.10' in rendered
    text = '\n'.join(cell.source for cell in notebook.cells if cell.cell_type == 'markdown')
    assert 'docs/answers_to_feedback' not in text
    assert 'точность нормализации не измерена' in text


def test_research_recomputation_is_opt_in_without_private_data_requirement(notebook, tmp_path):
    cell = next(cell for cell in notebook.cells if cell.cell_type == 'code'
                and 'RECOMPUTE_RESEARCH =' in cell.source)
    tree = ast.parse(cell.source)
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == 'RECOMPUTE_RESEARCH'
                              for target in node.targets))
    assert isinstance(assignment, ast.Assign)
    assert isinstance(assignment.value, ast.BoolOp) and isinstance(assignment.value.op, ast.And)
    assert isinstance(assignment.value.values[0], ast.Call)
    assert ast.unparse(assignment.value.values[0].func) == 'all'
    assert ast.unparse(assignment.value.values[1]) == 'not study_cache_valid(ROOT)'
    namespace = {'ROOT': tmp_path, 'research_config': {
        'inputs': {key: key + '.csv' for key in ['facts', 'dictionary', 'embeddings']}},
        'study_cache_valid': lambda root: False}
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), '<automatic input detection>', 'exec'), namespace)
    assert namespace['RECOMPUTE_RESEARCH'] is False
    for key in ['facts', 'dictionary', 'embeddings']:
        (tmp_path / (key + '.csv')).write_text('private input')
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), '<automatic input detection>', 'exec'), namespace)
    assert namespace['RECOMPUTE_RESEARCH'] is True
    namespace['study_cache_valid'] = lambda root: True
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), '<automatic input detection>', 'exec'), namespace)
    assert namespace['RECOMPUTE_RESEARCH'] is False
    (tmp_path / 'embeddings.csv').unlink()
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), '<automatic input detection>', 'exec'), namespace)
    assert namespace['RECOMPUTE_RESEARCH'] is False
    assert "json.loads(research_path.read_text())" in cell.source
    review = next(cell for cell in notebook.cells if cell.id == 'bf7a6206db8d4194')
    assert 'RECOMPUTE_REVIEW = RECOMPUTE_RESEARCH' in review.source


def test_feedback_note_covers_seven_points_and_links_to_existing_evidence():
    path = ROOT / 'docs/answers_to_feedback.md'
    text = path.read_text()
    assert re.findall(r'^## (\d+)\.', text, re.MULTILINE) == list('1234567')
    for fragment in ['Явно вымышленный пример', 'не 9 заказов', 'Смысловая точность не измерена',
                     'не является условием', 'сокращённый каталог', 'не ансамбль пяти',
                     '999', 'patience', '954', 'Итоговые статусы', '45 секунд']:
        assert fragment in text
    links = re.findall(r']\(([^)]+)\)', text)
    assert links
    for link in links:
        assert (path.parent / link).is_file(), link
    for fragment in ['19,06%', '22,85%', '17,7931%', '17,5354%', 'три запуска выше базы, два ниже']:
        assert fragment in text


def test_feedback_answers_are_self_contained_in_notebook(notebook):
    answers = {cell.id: cell.source for cell in notebook.cells if cell.cell_type == 'markdown'}
    for cell_id in ['b270d4ee824108da', '824f111c3eb039c8', '89a8e0f607946752',
                    '12e02b2d7f8ad026', '1198fc7954b097f1', '7e102ed155dec53c']:
        source = answers[cell_id]
        assert len(source.split()) <= 200
        assert '**Простыми словами.**' not in source
    assert all(value in answers['b270d4ee824108da'] for value in ['0,0203', '0,0588', '0,0494'])
    assert 'без нового поиска весов' in answers['b270d4ee824108da']
    assert '919 из 9 197' in answers['824f111c3eb039c8'] and '1,83' in answers['824f111c3eb039c8']
    assert '33 обучения' in answers['89a8e0f607946752'] and 'random search' in answers['89a8e0f607946752']
    assert 'полные 14 дней истории и 14 целевых дней' in answers['12e02b2d7f8ad026']
    assert 'среднее и std по трём seed' in answers['89a8e0f607946752']
    assert 'не формальная проверка значимости' in answers['89a8e0f607946752']
    assert '[−25,01; −1,25]' in answers['824f111c3eb039c8']
    assert '[−3,26; +6,50]' in answers['824f111c3eb039c8']
    assert all(value in answers['7e102ed155dec53c'] for value in ['+0,26', '−1,82', 'не подтверждено'])
    assert '477 579' in answers['1198fc7954b097f1'] and '22.10–04.11' in answers['1198fc7954b097f1']
    assert 'статус доступности, не результат качества' in answers['1198fc7954b097f1']
    assert 'очередью для ручного решения' in answers['7e102ed155dec53c']
    introduction = notebook.cells[0].source
    assert '#rezultaty' in introduction
    assert any('id="rezultaty"' in cell.source for cell in notebook.cells
               if cell.cell_type == 'markdown')


def test_notebook_presents_one_study_without_revision_or_feedback_narrative(notebook):
    text = '\n'.join(
        [cell.source for cell in notebook.cells]
        + [output.get('text', '') for cell in notebook.cells for output in cell.get('outputs', [])]
        + [value for cell in notebook.cells for output in cell.get('outputs', [])
           for key, value in output.get('data', {}).items() if key.startswith('text/')]
    ).lower()
    for fragment in ['замечание', 'otvety-na-zamechaniya', 'answers_to_feedback',
                     'прежн', 'старый протокол', 'старые сетки', 'старые агрегаты',
                     'новый протокол', 'новые проверки', 'новые параметры', 'новые таблицы',
                     'архив', 'больше не участвуют', 'формулировка «64', 'руководител']:
        assert re.search(re.escape(fragment), text) is None, fragment
    text = '\n'.join(cell.source for cell in notebook.cells if cell.cell_type == 'markdown')
    assert 'устойчивое преимущество не подтверждено' in text.lower()
    assert 'Рост продаж не измерялся' in text
    assert 'адаптера и LightGCN' in text


def test_prospective_check_can_run_without_recomputing_research(notebook, tmp_path, capsys):
    cell = next(cell for cell in notebook.cells if cell.cell_type == 'code'
                and 'automatic_status = run_automatic_check(ROOT)' in cell.source)
    assert 'RUN_PROSPECTIVE_CHECK' not in cell.source
    assert 'DOWNLOAD_PROSPECTIVE_DATA' not in cell.source
    assert 'RECOMPUTE_RESEARCH' not in cell.source
    assert 'train_network(' not in cell.source
    assert 'fit_fusion(' not in cell.source
    assert "protocol = json.loads(protocol_path.read_text())" in cell.source
    assert "sha256(protocol_path) == review['full_protocol_sha256']" in cell.source
    assert 'verify_full_protocol(ROOT, protocol)' in cell.source
    assert 'evaluate_prospective(' not in cell.source
    assert 'notebook_independent_status.json' in cell.source
    from cluster_data import sha256

    reports = tmp_path / 'reports'
    reports.mkdir()
    protocol = {'decision': '2026-10-21', 'window_days': 14,
                'inputs': {'facts': {'path': 'data/missing-facts.csv'}}, 'artifacts': {}}
    path = reports / 'notebook_full_protocol.json'
    path.write_text(json.dumps(protocol))
    (reports / 'notebook_review.json').write_text(json.dumps({'full_protocol_sha256': sha256(path)}))
    namespace = {'ROOT': tmp_path, 'json': json, 'sha256': sha256, 'pd': pd, 'display': display}
    exec(compile(cell.source, '<automatic prospective check>', 'exec'), namespace)
    assert 'Статус проверки: доступен публичный отчёт, приватных входов нет' in capsys.readouterr().out
    assert namespace['prospective_result']['status'] == 'private_inputs_unavailable'
    assert namespace['automatic_status']['source']['status'] == 'not_required'


@pytest.mark.parametrize('scenario', ['small_catalog', 'zero_co_sales', 'changed_protocol', 'final_model'])
def test_prospective_notebook_cell_uses_frozen_inputs(notebook, tmp_path, capsys, scenario):
    from cluster_data import sha256

    source = next(cell.source for cell in notebook.cells if cell.cell_type == 'code'
                  and 'automatic_status = run_automatic_check(ROOT)' in cell.source)
    tree = ast.parse(source)
    data = tmp_path / 'data'
    reports = tmp_path / 'reports'
    data.mkdir()
    reports.mkdir()
    (data / 'notebook_independent_targets.csv').write_text('date,sku,query\n2026-10-22,a,q\n')
    protocol = {'model_identity': 'historical_full_catalog_four_channels', 'catalog_clusters': 477579,
        'seed': 7, 'window_days': 14, 'max_views': 5, 'max_seeds': 32, 'per_source': 150,
        'no_tuning_on_future': True,
        'weights': {'Полные четыре': {name: .25 for name in
            ['pretrained_embeddings', 'co_sales', 'semantic_adapter', 'lightgcn']},
            'База: E5 + co-sales': {'pretrained_embeddings': .5, 'co_sales': .5}}}
    if scenario == 'small_catalog':
        protocol['catalog_clusters'] = 18444
    elif scenario == 'zero_co_sales':
        protocol['weights']['Полные четыре']['co_sales'] = 0
    else:
        protocol['model_identity'] = 'final_production'
    protocol_path = reports / 'notebook_full_protocol.json'
    protocol_path.write_text(json.dumps(protocol))
    (reports / 'notebook_review.json').write_text(json.dumps({'full_protocol_sha256': sha256(protocol_path)}))
    if scenario == 'changed_protocol':
        protocol_path.write_text(json.dumps({**protocol, 'per_source': 50}))
    namespace = {'ROOT': tmp_path, 'json': json, 'sha256': sha256}
    if scenario == 'changed_protocol':
        with pytest.raises(AssertionError, match='протокол изменён'):
            exec(compile(tree, '<frozen notebook evaluation>', 'exec'), namespace)
    else:
        with pytest.raises(ValueError, match='исходной полной модели'):
            exec(compile(tree, '<frozen notebook evaluation>', 'exec'), namespace)
    assert 'prospective_result' not in namespace


def test_notebook_separates_registration_predictions_and_target_evaluation(notebook):
    text = next(cell.source for cell in notebook.cells if cell.id == '1198fc7954b097f1')
    assert 'целевые заказы не читаются' in text.lower()
    assert 'рекомендации не пересчитываются' in text
    assert 'Ненайденные цели учитываются' in text
    preflight = next(cell.source for cell in notebook.cells if cell.id == 'dc7a159350b8a64a')
    assert 'check_only=True' in preflight
    assert "if all((ROOT / item['path']).is_file() for item in private_records):" in preflight
    source = next(cell.source for cell in notebook.cells if 'automatic_status = run_automatic_check(ROOT)' in cell.source)
    assert source.index('verify_full_protocol(ROOT, protocol)') < source.index('automatic_status = run_automatic_check(ROOT)')
    assert 'prospective_result = automatic_status[\'evaluation\']' in source
    code = '\n'.join(cell.source for cell in notebook.cells if cell.cell_type == 'code')
    assert 'RUN_PROSPECTIVE_CHECK' not in code and 'DOWNLOAD_PROSPECTIVE_DATA' not in code
    assert 'RECOMPUTE_RESEARCH = False' not in code and 'RECOMPUTE_REVIEW = False' not in code


def test_current_source_and_full_window_are_automatic_and_visible_in_notebook(notebook):
    code = '\n'.join(cell.source for cell in notebook.cells if cell.cell_type == 'code')
    text = '\n'.join(cell.source for cell in notebook.cells if cell.cell_type == 'markdown')
    assert 'current_source = refresh_source_snapshot(ROOT)' in code
    assert 'current_validation = run_closed_validation(ROOT, current_source)' in code
    assert "current_validation['target_days'] == 14" in code
    assert "not current_validation['training_performed']" in code
    assert "not current_validation['hyperparameter_selection_performed']" in code
    assert 'current-check' in text and 'последней доступной датой' in text
    assert 'хронологическая оценка' in text.lower()
    current_section = next(cell.source for cell in notebook.cells if cell.id == 'a95469096cafe0f2')
    assert 'диагностическая хронологическая оценка' in current_section
    assert 'не независимый тест' in current_section


def test_public_current_report_keeps_full_model_complete_window_and_all_targets():
    report = json.loads((ROOT / 'reports/notebook_current_validation.json').read_text())
    source = json.loads((ROOT / 'reports/notebook_current_data.json').read_text())
    assert source['source_rows'] == source['source_fingerprint']['rows'] > 0
    assert source['last_day'] == report['target_end']
    assert report['catalog_clusters'] == 477579
    assert report['target_days'] == 14
    import pandas as pd

    target_days = pd.date_range(report['target_start'], report['target_end'])
    history_days = pd.date_range(report['history_start'], report['decision'])
    assert len(target_days) == len(history_days) == 14
    assert (target_days.min() - history_days.max()).days == 1
    assert set(target_days.strftime('%Y-%m-%d')).isdisjoint(source['source_completeness_audit']['missing_dates'])
    assert set(history_days.strftime('%Y-%m-%d')).isdisjoint(source['source_completeness_audit']['missing_dates'])
    assert not report['independent_test']
    assert not report['training_performed'] and not report['hyperparameter_selection_performed']
    models = report['models']
    assert len(models) == 9
    assert len({row['positive_pairs'] for row in models.values()}) == 1
    assert len({row['evaluated_skus'] for row in models.values()}) == 1
    interval = report['interval']
    assert interval['n_targets'] == models['Полные четыре']['positive_pairs']
    assert interval['n_skus'] == models['Полные четыре']['evaluated_skus']
    assert sum(row['n_targets'] for row in report['slices']) == interval['n_targets']
    assert sum(row['n_skus'] for row in report['slices']) == interval['n_skus']
    assert report['targets_not_retrieved'] <= interval['n_targets']
    assert report['effect_confirmed_on_window'] == (interval['n_targets'] > 0 and interval['lower'] > 0)
    assert report['weights']['Полные четыре'] == {key: .25 for key in
        ['pretrained_embeddings', 'co_sales', 'semantic_adapter', 'lightgcn']}
    assert report['weights']['База: E5 + co-sales'] == {'pretrained_embeddings': .5, 'co_sales': .5}
    assert report['channel_count_interval']['n_targets'] == interval['n_targets']
    assert np.isclose(report['channel_count_interval']['difference'],
                      models['Полные четыре']['recall@10'] - models['Четыре без обучения']['recall@10'])
