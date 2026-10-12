import hashlib
import json
from pathlib import Path

import pytest

from cluster_data import sha256, write_json


@pytest.mark.parametrize("payload", [b"", b"abc", b"x" * (2 * 1024 * 1024 + 3)])
def test_sha256_preserves_the_historical_byte_contract(tmp_path, payload):
    path = tmp_path / "input"
    path.write_bytes(payload)
    assert sha256(path) == hashlib.sha256(payload).hexdigest()


def test_json_is_utf8_and_matches_existing_artifact_format(tmp_path):
    path = tmp_path / "result.json"
    payload = {"кластер": "новый", "count": 3}
    write_json(path, payload)
    assert path.read_bytes() == (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_json_is_not_published(tmp_path, value):
    path = tmp_path / "result.json"
    with pytest.raises(ValueError):
        write_json(path, {"metric": value})
    assert not path.exists()


def requirements(path):
    dependencies = set()
    for line in path.read_text().splitlines():
        if line.startswith("-r "):
            dependencies.update(requirements(path.parent / line[3:]))
        elif line.strip() and not line.startswith("#"):
            dependencies.add(line.split(">=")[0])
    return dependencies


def test_minimal_and_neural_dependencies_do_not_install_legacy_or_presentation():
    """Единый комплект заменяет прежнее разделение, но не возвращает legacy."""
    root = Path(__file__).resolve().parents[1]
    full = requirements(root / "requirements.txt")
    assert {"catboost", "scikit-learn", "numpy", "pandas", "scipy", "clickhouse-connect",
            "python-dotenv", "torch", "sentence-transformers", "python-pptx", "Pillow",
            "python-docx", "pytest"} == full
    assert not {"mlflow", "openpyxl", "joblib"} & full