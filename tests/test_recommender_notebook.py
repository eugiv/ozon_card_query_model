import ast
from pathlib import Path

import nbformat
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