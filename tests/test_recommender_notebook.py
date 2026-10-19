import ast
import copy
import json
import re
from pathlib import Path
from typing import Callable, cast

import nbformat
import numpy as np
import pytest

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
    assert any("image/png" in output.get("data", {}) for output in code_cells[1].outputs)
    examples = code_cells[2]
    rendered = str(examples.outputs)
    assert "Выигрыш системы" in rendered
    assert "Потеря системы" in rendered
    assert "Включает ноль" in rendered
    assert "gained.head(3)" in examples.source and "lost.head(3)" in examples.source


def test_research_covers_review_topics_without_claiming_independent_success(notebook):
    code = '\n'.join(cell.source for cell in notebook.cells if cell.cell_type == 'code')
    text = '\n'.join(cell.source for cell in notebook.cells if cell.cell_type == 'markdown')
    for required in ['data_overview(', 'semantic_review_summary(', 'train_network(',
                     'validation_callback=', 'patience=5', 'best_epoch',
                     'Четыре без обучения', 'training_noise', 'sensitivity',
                     'prospective_status(', 'evaluate_prospective(', 'frozen_vectors_sha256']:
        assert required in code, required
    assert 'не доказано' in notebook.cells[0].source
    assert 'не выдумываются' in text
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
    code = [cell for cell in notebook.cells if cell.cell_type == 'code'][1]
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
    assert 'Не независимый тест' in rendered
    text = '\n'.join(cell.source for cell in notebook.cells if cell.cell_type == 'markdown')
    assert 'docs/answers_to_feedback.md' in text
    assert 'Экспертная разметка не является условием запуска' in text


def test_research_recomputation_is_opt_in_without_private_data_requirement(notebook):
    cell = next(cell for cell in notebook.cells if cell.cell_type == 'code'
                and 'RECOMPUTE_RESEARCH =' in cell.source)
    tree = ast.parse(cell.source)
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == 'RECOMPUTE_RESEARCH'
                              for target in node.targets))
    assert isinstance(assignment, ast.Assign)
    assert isinstance(assignment.value, ast.Constant) and assignment.value.value is False
    assert "json.loads(research_path.read_text())" in cell.source


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
    answers = [cell.source for cell in notebook.cells if cell.cell_type == 'markdown'
               and re.match(r'### Замечание \d+\.', cell.source)]
    assert [re.match(r'### Замечание (\d+)\.', source).group(1)
            for source in answers] == list('1234567')
    for source in answers:
        assert len(source.split()) <= 200
        assert '**Простыми словами.**' not in source
    assert 'тест ещё не выполнен' in answers[0]
    assert 'Смысловая точность нормализации не измерена' in answers[1]
    assert '17,79%' in answers[2] and '17,54%' in answers[2] and '1,49' in answers[2]
    assert 'Три запуска выше базы, два ниже' in answers[2]
    assert 'не ансамбль пяти моделей' in answers[2]
    assert 'не типы товаров' in answers[2]
    assert '13,81%' in answers[3] and '15,71%' in answers[3]
    assert 'Общий пул кандидатов' in answers[3]
    assert '999 будущих заказов и просмотров' in answers[4]
    assert 'не глобальный оптимум' in answers[5]
    assert '954' in answers[6] and '15 шагов' in answers[6]
    assert 'не подтверждает обучение архивных весов' in answers[6]
    introduction = notebook.cells[0].source
    assert '#otvety-na-zamechaniya' in introduction
    assert any('id="otvety-na-zamechaniya"' in cell.source for cell in notebook.cells
               if cell.cell_type == 'markdown')


def test_prospective_check_can_run_without_recomputing_research(notebook, tmp_path, capsys):
    cell = next(cell for cell in notebook.cells if cell.cell_type == 'code'
                and 'FUTURE_FACTS =' in cell.source)
    assert 'RUN_PROSPECTIVE_CHECK = False' in cell.source
    assert 'RECOMPUTE_RESEARCH' not in cell.source
    assert 'train_network(' not in cell.source
    assert 'fit_fusion(' not in cell.source
    assert "protocol = json.loads(protocol_path.read_text())" in cell.source
    assert "sha256(protocol_path) == research['protocol_sha256']" in cell.source
    namespace = {'ROOT': tmp_path}
    exec(compile(cell.source, '<prospective opt-in>', 'exec'), namespace)
    assert 'Независимая оценка не выполнена' in capsys.readouterr().out
    assert not namespace['RUN_PROSPECTIVE_CHECK']


@pytest.mark.parametrize('scenario', ['complete', 'missing_day', 'changed_protocol', 'reused_rows'])
def test_prospective_notebook_cell_uses_frozen_inputs(notebook, tmp_path, capsys, scenario):
    import pandas as pd
    from cluster_data import sha256
    from notebook_research import evaluate_prospective, prospective_status

    source = next(cell.source for cell in notebook.cells if cell.cell_type == 'code'
                  and 'FUTURE_FACTS =' in cell.source)
    tree = ast.parse(source)
    flag = tree.body[0]
    assert isinstance(flag, ast.Assign) and isinstance(flag.value, ast.Constant)
    assert flag.targets[0].id == 'RUN_PROSPECTIVE_CHECK'
    flag.value.value = True
    data = tmp_path / 'data'
    reports = tmp_path / 'reports'
    data.mkdir()
    reports.mkdir()
    code_path = tmp_path / 'notebook_research.py'
    code_path.write_text((ROOT / 'src/notebook_research.py').read_text())
    records = [[day, 'a', 'seed', 1, 10, 1]
               for day in pd.date_range('2026-10-03', '2026-10-30')]
    records += [[pd.Timestamp('2026-10-17'), 'a', query, 1, 1, 1]
                for query in ['new', 'outside']]
    facts = pd.DataFrame(records, columns=['date', 'sku', 'query', 'order_count',
                                          'unique_view_users', 'unique_search_users'])
    history = facts.loc[facts.date.le('2026-10-07')]
    future = facts.loc[facts.date.gt('2026-10-07')]
    if scenario == 'missing_day':
        future = future.loc[future.date.ne('2026-10-20')]
    elif scenario == 'reused_rows':
        future = pd.concat([future, history.tail(1)], ignore_index=True)
    history.to_csv(data / 'facts.csv', index=False)
    future.to_csv(data / 'notebook_future_facts.csv', index=False)
    dictionary = pd.DataFrame({'query': ['seed', 'new', 'outside'],
        'normquery': ['seed', 'new', 'outside'], 'loaded_at': ['2026-06-26T00:00:00Z'] * 3,
        'status_code': [200] * 3, 'error': [''] * 3})
    dictionary.to_csv(data / 'dictionary.csv', index=False)
    vectors = np.array([[.8, .6], [1., 0.]], dtype=np.float32)
    frozen_path = data / 'notebook_frozen_vectors.npz'
    np.savez_compressed(frozen_path, catalog=np.array(['new', 'seed']),
                        e5=vectors, adapter=vectors, graph=vectors)
    config = {'inputs': {'facts': 'data/facts.csv', 'dictionary': 'data/dictionary.csv'}}
    (tmp_path / 'config.json').write_text(json.dumps(config))
    protocol = {'decision': '2026-10-16', 'frozen_at': '2026-10-14T00:00:00Z',
        'window_days': 14, 'train_end': '2026-08-21', 'selection_end': '2026-09-18',
        'dictionary_as_of': '2026-06-26T23:59:59.999999999Z', 'per_source': 5, 'max_views': 5,
        'weights': {'Полные четыре': {'semantic_adapter': .5, 'lightgcn': .5},
                    'База: E5 + co-sales': {'pretrained_embeddings': 1.}},
        'code_sha256': {'notebook_research.py': sha256(code_path)},
        'source_sha256': {key: sha256(tmp_path / path) for key, path in config['inputs'].items()},
        'frozen_vectors_sha256': sha256(frozen_path)}
    protocol_path = reports / 'notebook_research_protocol.json'
    protocol_path.write_text(json.dumps(protocol))
    (reports / 'notebook_research.json').write_text(json.dumps({'protocol_sha256': sha256(protocol_path)}))
    if scenario == 'changed_protocol':
        protocol_path.write_text(json.dumps({**protocol, 'decision': '2026-10-17'}))
    namespace = {'ROOT': tmp_path, 'json': json, 'pd': pd, 'np': np, 'sha256': sha256,
        'prospective_status': prospective_status, 'evaluate_prospective': evaluate_prospective,
        'display': print}
    if scenario in ['changed_protocol', 'reused_rows']:
        message = 'протокол изменён' if scenario == 'changed_protocol' else 'старые строки'
        with pytest.raises(AssertionError, match=message):
            exec(compile(tree, '<frozen notebook evaluation>', 'exec'), namespace)
    else:
        exec(compile(tree, '<frozen notebook evaluation>', 'exec'), namespace)
        if scenario == 'missing_day':
            assert 'prospective_result' not in namespace
            assert 'окно неполное' in capsys.readouterr().out
        else:
            result = namespace['prospective_result']
            assert result['status'] == 'evaluated_frozen_protocol'
            assert not result['training_performed']
            assert not result['hyperparameter_selection_performed']
            for metric in result['models'].values():
                assert metric['positive_pairs'] == 2
                assert metric['recall@10'] == .5
    assert sha256(frozen_path) == protocol['frozen_vectors_sha256']
