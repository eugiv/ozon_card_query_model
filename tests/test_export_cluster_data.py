import hashlib
import json
from datetime import date

import pandas as pd
import pytest

import cluster_data as export_cluster_data
from cluster_data import (
    DICTIONARY_SQL,
    FACT_SQL,
    dictionary_snapshot,
    export_data,
    source_client,
)


@pytest.fixture
def connection_settings(tmp_path, monkeypatch):
    for key in list(export_cluster_data.os.environ):
        if key.startswith("DBT_CLICKHOUSE_") or key == "OZON_CARD_ENV":
            monkeypatch.delenv(key)
    monkeypatch.setattr(export_cluster_data, "ROOT", tmp_path)
    calls = []
    monkeypatch.setattr(
        export_cluster_data.clickhouse_connect,
        "get_client",
        lambda **kwargs: calls.append(kwargs) or kwargs,
    )
    return tmp_path, calls


def test_source_client_reads_project_env_and_tls_files(
    connection_settings, monkeypatch
):
    root, calls = connection_settings
    (root / ".env").write_text(
        "DBT_CLICKHOUSE_HOST=example.test\n"
        "DBT_CLICKHOUSE_PORT=8443\n"
        "DBT_CLICKHOUSE_USER=reader\n"
        "DBT_CLICKHOUSE_PASSWORD=test-secret\n"
        "DBT_CLICKHOUSE_SCHEMA=analytics\n"
        "DBT_CLICKHOUSE_SECURE=true\n"
        "DBT_CLICKHOUSE_VERIFY=true\n"
        "DBT_CLICKHOUSE_CA_CERT=/certs/ca.pem\n"
        "DBT_CLICKHOUSE_CLIENT_CERT=/certs/client.pem\n"
        "DBT_CLICKHOUSE_CLIENT_KEY=/certs/client.key\n"
    )
    monkeypatch.chdir(root.parent)

    source_client()

    assert calls == [
        {
            "host": "example.test",
            "port": 8443,
            "username": "reader",
            "password": "test-secret",
            "database": "analytics",
            "secure": True,
            "verify": True,
            "ca_cert": "/certs/ca.pem",
            "client_cert": "/certs/client.pem",
            "client_cert_key": "/certs/client.key",
            "connect_timeout": 8,
            "send_receive_timeout": 120,
            "settings": {
                "readonly": 1,
                "max_result_rows": 0,
                "max_result_bytes": 0,
                "result_overflow_mode": "throw",
            },
        }
    ]


def test_source_client_explicit_env_and_process_override(
    connection_settings, monkeypatch
):
    root, calls = connection_settings
    (root / ".env").write_text("DBT_CLICKHOUSE_HOST=unused.test\n")
    explicit = root / "custom.env"
    explicit.write_text(
        "DBT_CLICKHOUSE_HOST=custom.test\nDBT_CLICKHOUSE_PASSWORD=file-secret\n"
        "DBT_CLICKHOUSE_VERIFY=false\n"
    )
    monkeypatch.setenv("OZON_CARD_ENV", str(explicit))
    monkeypatch.setenv("DBT_CLICKHOUSE_PASSWORD", "process-secret")

    source_client()

    assert calls[0]["host"] == "custom.test"
    assert calls[0]["password"] == "process-secret"
    assert calls[0]["verify"] is False
    assert calls[0]["ca_cert"] is None
    assert calls[0]["client_cert"] is None
    assert calls[0]["client_cert_key"] is None


def test_source_client_environment_only_and_secure_defaults(
    connection_settings, monkeypatch
):
    _, calls = connection_settings
    monkeypatch.setenv("DBT_CLICKHOUSE_HOST", "environment.test")
    source_client()
    assert calls[0]["host"] == "environment.test"
    assert calls[0]["verify"] is True


def test_source_client_requires_host_without_connecting(connection_settings):
    _, calls = connection_settings
    with pytest.raises(ValueError, match="DBT_CLICKHOUSE_HOST"):
        source_client()
    assert calls == []


def test_export_uses_bound_dates_and_freezes_dictionary(tmp_path):
    class Client:
        def __init__(self):
            self.calls = []

        def query_df(self, sql, parameters):
            self.calls.append((sql, parameters))
            if sql == FACT_SQL:
                return pd.DataFrame(
                    {"date": [parameters["start"]], "sku": ["1"], "order_count": [1]}
                )
            return pd.DataFrame({"example": [1]})

    client = Client()
    output = tmp_path / "source"
    result = export_data(client, date(2026, 1, 8), output, window_days=2)
    assert client.calls == [
        (FACT_SQL, {"start": date(2026, 1, 1), "end": date(2026, 1, 8)}),
        (DICTIONARY_SQL, {"snapshot_end": date(2026, 1, 2)}),
    ]
    assert result["dictionary_as_of_day_utc"] == "2026-01-02"
    assert result["history_start"] == "2026-01-01"
    assert (output / "source_manifest.json").exists()
    with pytest.raises(ValueError, match="уже существует"):
        export_data(client, date(2026, 1, 8), output, 2)


def test_empty_export_does_not_publish_partial_files(tmp_path):
    class Client:
        def query_df(self, sql, parameters):
            return pd.DataFrame()

    with pytest.raises(ValueError, match="пуста"):
        export_data(Client(), date(2026, 1, 8), tmp_path / "source", 2)
    assert not (tmp_path / "source").exists()


def test_compacted_dictionary_preserves_snapshot_and_ambiguity():
    records = pd.DataFrame(
        [
            [" Мыло ", "мыло", 200, "", "2026-01-01T12:00:00Z"],
            [" Мыло ", "мыло", 200, "", "2026-01-02T12:00:00Z"],
            ["мыло", "мыло новое", 200, "", "2026-01-02T12:00:00Z"],
            ["мыло", "ошибка", 500, "error", "2026-01-03T12:00:00Z"],
            ["мыло", "будущее", 200, "", "2026-02-01T12:00:00Z"],
        ],
        columns=["query", "normquery", "status_code", "error", "loaded_at"],
    )
    cutoff = pd.Timestamp("2026-01-03T23:59:59Z")
    historical = records.loc[pd.to_datetime(records["loaded_at"], utc=True) <= cutoff]
    compact = historical.groupby(
        ["query", "normquery", "status_code", "error"], as_index=False
    )["loaded_at"].max()
    expected, original_audit = dictionary_snapshot(records, cutoff)
    actual, compact_audit = dictionary_snapshot(compact, cutoff)
    pd.testing.assert_frame_equal(actual, expected)
    for key in ["ambiguous_queries", "mapped_queries", "clusters"]:
        assert original_audit[key] == compact_audit[key]
    assert "GROUP BY query, normquery, status_code, error" in DICTIONARY_SQL
    assert "max(toTimeZone(d.loaded_at, 'UTC'))" in DICTIONARY_SQL


@pytest.mark.parametrize(
    "start,cutoff,snapshot,missing",
    [
        ("2026-01-09", "2026-10-03", "2026-06-26", ["2026-02-04", "2026-08-12"]),
        ("2025-11-27", "2026-02-05", "2025-12-15", ["2025-12-31"]),
    ],
)
def test_full_arbitrary_export_manifest_and_missing_dates(
    tmp_path, start, cutoff, snapshot, missing
):
    expected_dates = [
        day.date() for day in pd.date_range(start, cutoff)
        if day.date().isoformat() not in missing
    ]
    facts = pd.DataFrame(
        [
            [day, sku, "soap", 10, 5, orders]
            for day in expected_dates for sku, orders in [("1", 1), ("2", 2)]
        ],
        columns=["date", "sku", "query", "unique_search_users", "unique_view_users", "order_count"],
    )
    dictionary = pd.DataFrame(
        [["soap", "soap", 200, "", f"{snapshot}T12:00:00Z"]],
        columns=["query", "normquery", "status_code", "error", "loaded_at"],
    )

    class Client:
        def __init__(self):
            self.calls = []

        def query_df(self, sql, parameters):
            self.calls.append((sql, parameters))
            return facts.copy() if sql == FACT_SQL else dictionary.copy()

    client = Client()
    output = tmp_path / "full"
    result = export_data(
        client, date.fromisoformat(cutoff), output,
        history_start=date.fromisoformat(start), dictionary_as_of=date.fromisoformat(snapshot),
    )
    assert client.calls == [
        (FACT_SQL, {"start": date.fromisoformat(start), "end": date.fromisoformat(cutoff)}),
        (DICTIONARY_SQL, {"snapshot_end": date.fromisoformat(snapshot)}),
    ]
    assert result["history_start"] == start
    assert result["cutoff"] == cutoff
    assert result["dictionary_as_of_day_utc"] == snapshot
    assert result["query_dates"] == {
        "facts": {"start": start, "end": cutoff, "inclusive": True},
        "dictionary": {"as_of_day_utc": snapshot, "exclusive_end_utc":
            (date.fromisoformat(snapshot) + pd.Timedelta(days=1)).isoformat() + "T00:00:00Z"},
    }
    audit = result["source_completeness_audit"]
    assert audit["date_min"] == start
    assert audit["date_max"] == cutoff
    assert audit["rows"] == 2 * len(expected_dates) == result["fact_rows"]
    assert audit["sku_count"] == 2
    assert audit["day_count"] == len(expected_dates)
    assert audit["expected_day_count"] == len(pd.date_range(start, cutoff))
    assert audit["missing_dates"] == missing
    assert audit["order_count"] == 3 * len(expected_dates)
    assert audit["daily"] == [
        {"date": day.isoformat(), "rows": 2, "sku_count": 2, "order_count": 3}
        for day in expected_dates
    ]
    months = sorted({day.strftime("%Y-%m") for day in expected_dates})
    assert audit["monthly"] == [
        {"month": month, "rows": 2 * sum(day.strftime("%Y-%m") == month for day in expected_dates),
         "sku_count": 2, "order_count": 3 * sum(day.strftime("%Y-%m") == month for day in expected_dates)}
        for month in months
    ]
    semantics = result["source_metric_semantics"]
    assert "card visible" in semantics["unique_view_users"]
    assert "not clicks" in semantics["unique_view_users"]
    assert "count" in semantics["order_count"]
    assert "not period-unique" in semantics["unique_search_users"]
    assert "not period-unique" in semantics["unique_view_users"]
    assert json.loads((output / "source_manifest.json").read_text()) == result
    pd.testing.assert_frame_equal(
        pd.read_csv(output / "facts.csv", dtype={"sku": str}),
        facts.assign(date=facts["date"].astype(str)),
    )
    for name, digest in result["files"].items():
        assert digest == hashlib.sha256((output / name).read_bytes()).hexdigest()
    for sql, _ in client.calls:
        assert sql.lstrip().startswith("SELECT")
        assert "LIMIT" not in sql.upper()
        assert "SAMPLE" not in sql.upper()
        assert "RAND" not in sql.upper()
    with pytest.raises(ValueError, match="уже существует"):
        export_data(client, date.fromisoformat(cutoff), output)
    assert len(client.calls) == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"history_start": date(2026, 10, 4)},
        {"dictionary_as_of": date(2026, 10, 4)},
        {"history_start": "2026-01-09"},
        {"dictionary_as_of": "2026-06-26"},
        {"cutoff": None},
        {"window_days": 0},
        {"window_days": -1},
        {"window_days": 1.5},
        {"training_windows": 0},
        {"training_windows": -1},
        {"training_windows": True},
        {"cutoff": "2026-10-03"},
        {"training_windows": 1.5},
        {"window_days": True},
    ],
)
def test_invalid_export_arguments_before_sql(tmp_path, kwargs):
    class Client:
        def query_df(self, *args, **kwargs):
            pytest.fail("Invalid arguments must not reach SQL")

    arguments = {"cutoff": date(2026, 10, 3), **kwargs}
    with pytest.raises(ValueError):
        export_data(Client(), output=tmp_path / "source", **arguments)
    assert not (tmp_path / "source").exists()


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["--cutoff", "not-a-date"],
        ["--cutoff", "2026-10-03", "--history-start", "2026-10-04"],
        ["--cutoff", "2026-10-03", "--dictionary-as-of", "2026-10-04"],
        ["--cutoff", "2026-10-03", "--training-windows", "0"],
        ["--cutoff", "2026-10-03", "--window-days", "-1"],
    ],
)
def test_invalid_cli_arguments_before_connection(tmp_path, monkeypatch, args):
    def forbidden_connection():
        pytest.fail("Invalid arguments must not open a connection")

    monkeypatch.setattr(export_cluster_data, "source_client", forbidden_connection)
    with pytest.raises(SystemExit) as error:
        export_cluster_data.main([*args, "--output", str(tmp_path / "source")])
    assert error.value.code == 2


def test_cli_passes_explicit_dates_and_closes_client(tmp_path, monkeypatch, capsys):
    class Client:
        closed = False

        def close(self):
            self.closed = True

    client = Client()
    calls = []

    def fake_export(*args, **kwargs):
        calls.append((args, kwargs))
        return {"fact_rows": 1000, "dictionary_rows": 100}

    monkeypatch.setattr(export_cluster_data, "source_client", lambda: client)
    monkeypatch.setattr(export_cluster_data, "export_data", fake_export)
    output = tmp_path / "source"
    export_cluster_data.main([
        "--cutoff", "2026-10-03", "--history-start", "2026-01-09",
        "--dictionary-as-of", "2026-06-26", "--output", str(output),
    ])
    assert calls == [
        ((client, date(2026, 10, 3), output, 28, 1),
         {"history_start": date(2026, 1, 9), "dictionary_as_of": date(2026, 6, 26)}),
    ]
    assert client.closed
    assert "1000" in capsys.readouterr().out


@pytest.mark.parametrize(
    "kwargs,expected_start,expected_snapshot",
    [
        ({"history_start": date(2025, 12, 1)}, date(2025, 12, 1), date(2026, 1, 2)),
        ({"dictionary_as_of": date(2026, 1, 6)}, date(2026, 1, 1), date(2026, 1, 6)),
        ({"history_start": date(2026, 1, 8), "dictionary_as_of": date(2026, 1, 8)},
         date(2026, 1, 8), date(2026, 1, 8)),
    ],
)
def test_export_independent_optional_dates(tmp_path, kwargs, expected_start, expected_snapshot):
    class Client:
        def __init__(self):
            self.calls = []

        def query_df(self, sql, parameters):
            self.calls.append((sql, parameters))
            return pd.DataFrame({"date": [date(2026, 1, 8)], "sku": ["1"], "order_count": [2]})

    client = Client()
    result = export_data(client, date(2026, 1, 8), tmp_path / "source", 2, **kwargs)
    assert client.calls == [
        (FACT_SQL, {"start": expected_start, "end": date(2026, 1, 8)}),
        (DICTIONARY_SQL, {"snapshot_end": expected_snapshot}),
    ]
    assert result["history_start"] == expected_start.isoformat()
    assert result["dictionary_as_of_day_utc"] == expected_snapshot.isoformat()


def test_completeness_audit_missing_boundaries_and_distinct_skus():
    facts = pd.DataFrame(
        [
            [date(2026, 1, 31), "1", 2],
            [date(2026, 1, 31), "1", 3],
            [date(2026, 2, 1), "1", 4],
            [date(2026, 2, 1), "2", 5],
        ],
        columns=["date", "sku", "order_count"],
    )
    result = export_cluster_data.source_completeness_audit(
        facts, date(2026, 1, 30), date(2026, 2, 2)
    )
    assert result == {
        "rows": 4, "sku_count": 2, "order_count": 14,
        "date_min": "2026-01-31", "date_max": "2026-02-01",
        "day_count": 2, "expected_day_count": 4,
        "missing_dates": ["2026-01-30", "2026-02-02"],
        "daily": [
            {"date": "2026-01-31", "rows": 2, "sku_count": 1, "order_count": 5},
            {"date": "2026-02-01", "rows": 2, "sku_count": 2, "order_count": 9},
        ],
        "monthly": [
            {"month": "2026-01", "rows": 2, "sku_count": 1, "order_count": 5},
            {"month": "2026-02", "rows": 2, "sku_count": 2, "order_count": 9},
        ],
    }


@pytest.mark.parametrize("fact_date", [None, "2026-01-09", "not-a-date"])
def test_invalid_fact_dates_do_not_publish(tmp_path, fact_date):
    class Client:
        def query_df(self, sql, parameters):
            return pd.DataFrame({"date": [fact_date], "sku": ["1"], "order_count": [1]})

    with pytest.raises(ValueError):
        export_data(Client(), date(2026, 1, 8), tmp_path / "source", 2)
    assert not (tmp_path / "source").exists()
    assert list(tmp_path.iterdir()) == []


def test_write_failure_cleans_temporary_files(tmp_path, monkeypatch):
    class Client:
        def query_df(self, sql, parameters):
            return pd.DataFrame({"date": [date(2026, 1, 8)], "sku": ["1"], "order_count": [1]})

    def failing_write(path, payload):
        raise OSError("test write failure")

    monkeypatch.setitem(export_cluster_data.__dict__, "write_json", failing_write)
    with pytest.raises(OSError, match="test write failure"):
        export_data(Client(), date(2026, 1, 8), tmp_path / "source", 2)
    assert list(tmp_path.iterdir()) == []


def test_concurrent_output_is_not_overwritten(tmp_path, monkeypatch):
    output = tmp_path / "source"
    original_write = export_cluster_data.write_json

    class Client:
        def query_df(self, sql, parameters):
            return pd.DataFrame({"date": [date(2026, 1, 8)], "sku": ["1"], "order_count": [1]})

    def concurrent_write(path, payload):
        original_write(path, payload)
        output.mkdir()
        (output / "existing.txt").write_text("keep")

    monkeypatch.setitem(export_cluster_data.__dict__, "write_json", concurrent_write)
    with pytest.raises(ValueError, match="публикация отменена"):
        export_data(Client(), date(2026, 1, 8), output, 2)
    assert (output / "existing.txt").read_text() == "keep"
    assert list(output.iterdir()) == [output / "existing.txt"]
    assert list(tmp_path.iterdir()) == [output]


def test_existing_cli_output_does_not_connect(tmp_path, monkeypatch):
    def forbidden_connection():
        pytest.fail("Existing output must not open a connection")

    monkeypatch.setattr(export_cluster_data, "source_client", forbidden_connection)
    output = tmp_path / "source"
    output.mkdir()
    with pytest.raises(SystemExit) as error:
        export_cluster_data.main(["--cutoff", "2026-01-08", "--output", str(output)])
    assert error.value.code == 2
