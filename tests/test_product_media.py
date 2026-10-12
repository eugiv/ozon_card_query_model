import csv
import hashlib
import json
from datetime import datetime, timezone

import pandas as pd
import pytest
from clickhouse_connect.driver.binding import bind_query
from clickhouse_connect.driver.exceptions import DatabaseError, StreamClosedError
from urllib3.exceptions import HTTPError as Urllib3HTTPError

import cluster_data as media

AS_OF = pd.Timestamp("2026-10-06T23:16:00+03:00")
LOADED = "2026-10-06T00:45:00Z"
URL = "https://ir.ozone.ru/s3/multimedia-test/main.png"


def attributes():
    return json.dumps([
        {"id": 85, "values": [{"value": "Brand", "dictionary_value_id": 1}]},
        {"id": "8498", "values": [{"value": "Category"}]},
        {"id": 4191, "values": [{"value": "Description"}, {"value": "Second"}]},
        {"id": 11254, "values": [{"value": json.dumps({"img": {"src": URL}})}]},
    ])


def records(raw=None):
    return pd.DataFrame([
        {"sku": "1", "name": "Product", "attributes": attributes() if raw is None else raw, "loaded_at": LOADED},
        {"sku": "2", "name": "Other", "attributes": "[]", "loaded_at": LOADED},
    ])


class Client:
    def __init__(self, frame=None):
        self.frame = records() if frame is None else frame
        self.calls = []
        self.closed = False

    def query_df(self, sql, parameters):
        self.calls.append((sql, parameters))
        return self.frame

    def close(self):
        self.closed = True


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_metadata(output):
    with (output / "metadata.csv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_valid_attributes_and_richcontent_is_not_main():
    parsed = media.parse_attributes(attributes())
    assert parsed == {
        "brand": "Brand", "category": "Category", "description": "Description\nSecond",
        "attributes_status": "ok", "attributes_error": "",
    }
    row = media.build_metadata(records(), AS_OF)[0]
    assert row == {
        "sku": "1", "name": "Product", "brand": "Brand", "category": "Category",
        "description": "Description\nSecond", "attributes_json": attributes(),
        "snapshot_at": "2026-10-06T00:45:00+00:00",
        "attributes_status": "ok", "attributes_error": "",
    }
    assert media.METADATA_FIELDS == [
        "sku", "name", "brand", "category", "description", "attributes_json",
        "snapshot_at", "attributes_status", "attributes_error",
    ]


@pytest.mark.parametrize("raw,status", [
    ("{broken", "invalid_json"), (None, "invalid_json"),
    ('[{"id":85,"values":[{"value":NaN}]}]', "invalid_json"),
    ('[{"id":85,"id":85,"values":[]}]', "invalid_json"),
    ('{"brand":"unsupported"}', "invalid_schema"),
    ('["Brand"]', "invalid_schema"),
    ('[{"id":85,"values":"Brand"}]', "invalid_schema"),
    ('[{"id":85,"values":[{"value":5}]}]', "invalid_schema"),
    ('[{"id":true,"values":[]}]', "invalid_schema"),
    ('[{"id":0,"values":[]}]', "invalid_schema"),
    ('[{"id":-1,"values":[]}]', "invalid_schema"),
    (json.dumps([{"id": "9" * 5000, "values": []}]), "invalid_schema"),
])
def test_bad_attributes_are_diagnosed_not_silently_dropped(raw, status):
    parsed = media.parse_attributes(raw)
    assert parsed["attributes_status"] == status
    assert parsed["attributes_error"]
    assert parsed["brand"] == parsed["description"] == parsed["category"] == ""


def test_repeated_attribute_values_are_deduplicated_and_unknown_fields_preserved():
    raw = json.dumps([
        {"id": 85, "values": [{"value": "Brand"}, {"value": "Brand"}]},
        {"id": "85", "values": [{"value": "Second brand"}], "extra": "allowed"},
        {"id": 4191, "values": [{"value": "Text"}, {"value": "Text"}]},
        {"id": 123, "values": [{"value": "Unknown"}]},
    ])
    row = media.build_metadata(records(raw), AS_OF)[0]
    assert row["brand"] == "Brand\nSecond brand" and row["description"] == "Text"
    assert row["category"] == "" and row["attributes_status"] == "ok"
    assert row["attributes_error"] == "" and row["attributes_json"] == raw


@pytest.mark.parametrize("attribute_id", [2**32, str(2**32), "", "1.0", 1.5])
def test_attribute_ids_require_positive_uint32(attribute_id):
    parsed = media.parse_attributes(json.dumps([{"id": attribute_id, "values": []}]))
    assert parsed["attributes_status"] == "invalid_schema" and parsed["attributes_error"]
    assert parsed["brand"] == parsed["description"] == parsed["category"] == ""


def test_export_parameter_binding_filter_and_per_sku_diagnostics(tmp_path):
    sku_file = write_csv(tmp_path / "skus.csv", ["sku"], [{"sku": "1"}, {"sku": "999"}])
    client = Client(records("not json"))
    output = tmp_path / "snapshot"
    manifest = media.export_snapshot(client, AS_OF, output, sku_file=sku_file)
    assert client.calls == [(media.PRODUCT_SQL, {"as_of": datetime(2026, 10, 6, 20, 16, tzinfo=timezone.utc)})]
    assert "WHERE loaded_at <= {as_of:DateTime64(6, 'UTC')}" in media.PRODUCT_SQL
    assert "ORDER BY loaded_at DESC, name ASC, attributes ASC" in media.PRODUCT_SQL
    assert "999" not in media.PRODUCT_SQL and media.PRODUCT_SQL.strip().startswith("SELECT")
    rows = read_metadata(output)
    assert len(rows) == 1 and rows[0]["sku"] == "1"
    assert rows[0]["attributes_json"] == "not json" and rows[0]["attributes_status"] == "invalid_json"
    assert rows[0]["snapshot_at"] == "2026-10-06T00:45:00+00:00"
    assert manifest["source_rows"] == 2 and manifest["missing_requested_skus"] == ["999"]
    assert manifest["attribute_diagnostics"] == [{"sku": "1", "status": "invalid_json", "reason": "invalid_json"}]
    assert json.loads((output / "manifest.json").read_text()) == manifest
    assert manifest["files"]["metadata.csv"] == hashlib.sha256((output / "metadata.csv").read_bytes()).hexdigest()
    assert manifest["schema_version"] == 2
    assert set(manifest["files"]) == {"metadata.csv"}
    assert {path.name for path in output.iterdir()} == {"metadata.csv", "manifest.json"}
    before = (output / "metadata.csv").read_bytes()
    with pytest.raises(ValueError, match="перезапись"):
        media.export_snapshot(client, AS_OF, output)
    assert (output / "metadata.csv").read_bytes() == before and len(client.calls) == 1


def test_successful_text_export_preserves_titles_and_manifest_dates(tmp_path):
    output = tmp_path / "snapshot"
    manifest = media.export_snapshot(Client(), AS_OF, output)
    rows = read_metadata(output)
    assert [row["name"] for row in rows] == ["Product", "Other"]
    assert rows[0]["brand"] == "Brand" and rows[0]["category"] == "Category"
    assert rows[0]["description"] == "Description\nSecond"
    assert rows[0]["attributes_json"] == attributes()
    assert rows[1]["brand"] == rows[1]["category"] == rows[1]["description"] == ""
    assert rows[1]["attributes_json"] == "[]" and rows[1]["attributes_status"] == "ok"
    assert manifest["as_of"] == "2026-10-06T20:16:00+00:00"
    assert manifest["text_snapshot_times"] == ["2026-10-06T00:45:00+00:00"]
    assert manifest["requested_skus"] is None and manifest["missing_requested_skus"] == []
    assert manifest["attribute_diagnostics"] == []
    assert "not a historical product archive" in manifest["source_semantics"]
    assert manifest["snapshot_at_semantics"] == "actual text loaded_at in UTC"
    assert not any("image" in key or "download" in key or "media_snapshot" in key for key in manifest)


def test_actual_driver_binding_preserves_microseconds_and_utc(tmp_path):
    client = Client()
    stamp = pd.Timestamp("2026-10-06T23:16:00.123456+03:00")
    media.export_snapshot(client, stamp, tmp_path / "snapshot")
    sql, parameters = client.calls[0]
    bound_sql, bound_parameters = bind_query(sql, parameters, server_tz=timezone.utc)
    assert bound_sql == media.PRODUCT_SQL
    assert bound_parameters == {"param_as_of": "2026-10-06 20:16:00.123456"}


@pytest.mark.parametrize("sku", ["01", "1.0", " 1", "1 ", "0", "-1", "1; DROP TABLE raw.x", "18446744073709551616"])
def test_strict_sku_rejected(sku):
    with pytest.raises(ValueError):
        media.strict_sku(sku)


def test_duplicate_csv_sku_rejected_before_query(tmp_path):
    row = {"sku": "1"}
    path = write_csv(tmp_path / "duplicate.csv", list(row), [row, row])
    client = Client()
    with pytest.raises(ValueError, match="повторный SKU"):
        media.export_snapshot(client, AS_OF, tmp_path / "output", sku_file=path)
    assert client.calls == [] and not (tmp_path / "output").exists()


@pytest.mark.parametrize("stamp", ["2026-10-06", "2026-10-06T23:16:00", "NaT", "2026-10-06T00:00:00.000000001Z"])
def test_as_of_requires_explicit_timezone(stamp):
    with pytest.raises(ValueError):
        media.explicit_timestamp(stamp)


def test_explicit_timestamp_api_normalizes_timezone_and_preserves_microseconds():
    assert media.explicit_timestamp("2026-10-06T23:16:00.123456+03:00") == pd.Timestamp(
        "2026-10-06T20:16:00.123456Z"
    )
    assert str(media.explicit_timestamp(AS_OF).tz) == "UTC"


@pytest.mark.parametrize("stamp", [None, "NaT", "not a date", "2026-10-06T00:00:00.000000001Z"])
def test_invalid_source_dates_never_publish(tmp_path, stamp):
    frame = records()
    frame.loc[0, "loaded_at"] = stamp
    with pytest.raises(ValueError, match="loaded_at"):
        media.export_snapshot(Client(frame), AS_OF, tmp_path / "snapshot")
    assert not (tmp_path / "snapshot").exists()


@pytest.mark.parametrize("column", ["sku", "name", "attributes", "loaded_at"])
def test_missing_source_columns_never_publish(tmp_path, column):
    frame = records().drop(columns=column)
    with pytest.raises(ValueError, match="обязательных колонок"):
        media.build_metadata(frame, AS_OF)
    with pytest.raises(ValueError, match="обязательных колонок"):
        media.export_snapshot(Client(frame), AS_OF, tmp_path / "snapshot")
    assert not (tmp_path / "snapshot").exists()


@pytest.mark.parametrize("content", ["other\n1\n", "sku,sku\n1,1\n", "sku\n1,extra\n", "sku,name\n1,Product,extra\n"])
def test_sku_csv_schema_errors_are_rejected_before_query(tmp_path, content):
    path = tmp_path / "skus.csv"
    path.write_text(content, encoding="utf-8")
    client = Client()
    with pytest.raises(ValueError, match="CSV"):
        media.export_snapshot(client, AS_OF, tmp_path / "snapshot", sku_file=path)
    assert client.calls == [] and not (tmp_path / "snapshot").exists()


def test_empty_sku_filter_does_not_create_products(tmp_path):
    path = write_csv(tmp_path / "skus.csv", ["sku"], [])
    assert media.read_sku_csv(path, {"sku"}) == []
    manifest = media.export_snapshot(Client(), AS_OF, tmp_path / "snapshot", sku_file=path)
    assert manifest["rows"] == 0 and manifest["requested_skus"] == []
    assert manifest["source_rows"] == 2 and manifest["missing_requested_skus"] == []
    assert read_metadata(tmp_path / "snapshot") == []


def test_source_duplicate_future_and_unknown_join_rejected(tmp_path):
    with pytest.raises(ValueError, match="Повторный SKU"):
        media.export_snapshot(Client(pd.concat([records(), records()])), AS_OF, tmp_path / "duplicate")
    future = records()
    future.loc[0, "loaded_at"] = "2026-10-07T00:00:00Z"
    with pytest.raises(ValueError, match="loaded_at позже"):
        media.export_snapshot(Client(future), AS_OF, tmp_path / "future")
    assert not (tmp_path / "duplicate").exists() and not (tmp_path / "future").exists()


def test_atomic_publish_rejects_racing_empty_directory(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "data").write_text("complete")
    destination = tmp_path / "snapshot"
    destination.mkdir()
    with pytest.raises(ValueError, match="перезапись"):
        media.publish_snapshot(staging, destination)
    assert list(destination.iterdir()) == [] and (staging / "data").exists()


def test_empty_slice_is_explicit_and_naive_utc_source_timestamp_is_supported(tmp_path):
    empty = media.export_snapshot(Client(records().iloc[:0]), AS_OF, tmp_path / "empty")
    assert empty["rows"] == 0 and empty["attribute_diagnostics"] == []
    assert empty["text_snapshot_times"] == []
    assert read_metadata(tmp_path / "empty") == []
    with (tmp_path / "empty" / "metadata.csv").open(encoding="utf-8", newline="") as stream:
        assert csv.DictReader(stream).fieldnames == media.METADATA_FIELDS
    current = records()
    current["loaded_at"] = datetime(2026, 10, 6, 0, 45, tzinfo=timezone.utc).replace(tzinfo=None)
    media.export_snapshot(Client(current), AS_OF, tmp_path / "naive-source")
    assert read_metadata(tmp_path / "naive-source")[0]["snapshot_at"] == "2026-10-06T00:45:00+00:00"


def test_sku_csv_injection_is_rejected_before_query(tmp_path):
    sku_file = write_csv(tmp_path / "skus.csv", ["sku"], [{"sku": "1); DROP TABLE raw.x; --"}])
    client = Client()
    with pytest.raises(ValueError, match="SKU"):
        media.export_snapshot(client, AS_OF, tmp_path / "snapshot", sku_file=sku_file)
    assert client.calls == []


def test_failed_manifest_write_never_publishes_partial_snapshot(tmp_path, monkeypatch):
    def fail(*_):
        raise OSError("disk failure")

    monkeypatch.setattr(media, "write_json", fail)
    with pytest.raises(OSError):
        media.export_snapshot(Client(), AS_OF, tmp_path / "snapshot")
    assert not (tmp_path / "snapshot").exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("error_type", [ValueError, DatabaseError, Urllib3HTTPError, OSError, TypeError])
def test_cli_uses_reused_source_client_closes_it_and_sanitizes_query_errors(tmp_path, monkeypatch, capsys, error_type):
    client = Client()
    monkeypatch.setattr(media, "source_client", lambda: client)
    media.main(["products", "--as-of", AS_OF.isoformat(), "--output", str(tmp_path / "snapshot")])
    assert client.closed and "Строк: 2" in capsys.readouterr().out

    def fail(*args, **kwargs):
        raise error_type("password=secret https://user:secret@host/")

    client.closed = False
    monkeypatch.setattr(client, "query_df", fail)
    with pytest.raises(SystemExit) as error:
        media.main(["products", "--as-of", AS_OF.isoformat(), "--output", str(tmp_path / "failed")])
    assert error.value.code == 1 and client.closed
    assert "secret" not in capsys.readouterr().err and not (tmp_path / "failed").exists()


def test_cli_sanitizes_unexpected_connection_error(tmp_path, monkeypatch, capsys):
    def fail():
        raise TypeError("password=secret https://user:secret@host/")

    monkeypatch.setattr(media, "source_client", fail)
    with pytest.raises(SystemExit) as error:
        media.main(["products", "--as-of", AS_OF.isoformat(), "--output", str(tmp_path / "snapshot")])
    assert error.value.code == 1 and not (tmp_path / "snapshot").exists()
    captured = capsys.readouterr()
    assert captured.out == "" and "password=secret" not in captured.err and "https://" not in captured.err


def test_cli_sku_file_filters_current_text(tmp_path, monkeypatch, capsys):
    client = Client()
    monkeypatch.setattr(media, "source_client", lambda: client)
    path = write_csv(tmp_path / "skus.csv", ["sku"], [{"sku": "2"}, {"sku": "999"}])
    output = tmp_path / "snapshot"
    media.main(["products", "--as-of", AS_OF.isoformat(), "--output", str(output), "--sku-file", str(path)])
    assert client.closed and len(client.calls) == 1
    assert [row["name"] for row in read_metadata(output)] == ["Other"]
    assert json.loads((output / "manifest.json").read_text())["missing_requested_skus"] == ["999"]
    assert "Строк: 1; ошибки attributes: 0" in capsys.readouterr().out


def test_cli_help_is_text_only_without_connecting(monkeypatch, capsys):
    monkeypatch.setattr(media, "source_client", lambda: pytest.fail("help must not connect"))
    with pytest.raises(SystemExit) as error:
        media.main(["products", "--help"])
    assert error.value.code == 0
    text = capsys.readouterr().out
    assert "--as-of" in text and "--output" in text and "--sku-file" in text
    assert "--main-images" not in text and "--download" not in text
    assert "--max-bytes" not in text and "--timeout" not in text


@pytest.mark.parametrize("extra", [
    ["--main-images", "retired.csv"], ["--download"],
    ["--max-bytes", "100"], ["--timeout", "1"],
])
def test_retired_photo_cli_flags_are_rejected_before_connection(tmp_path, monkeypatch, capsys, extra):
    monkeypatch.setattr(media, "source_client", lambda: pytest.fail("invalid arguments must not connect"))
    with pytest.raises(SystemExit) as error:
        media.main(["products", "--as-of", AS_OF.isoformat(), "--output", str(tmp_path / "snapshot"), *extra])
    assert error.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err
    assert not (tmp_path / "snapshot").exists()


@pytest.mark.parametrize("error_type", [StreamClosedError, Urllib3HTTPError, OSError, ValueError])
@pytest.mark.parametrize("query_failed", [False, True])
def test_cli_cleanup_errors_do_not_leak_secrets(tmp_path, monkeypatch, capsys, error_type, query_failed):
    client = Client()

    def close():
        client.closed = True
        if error_type is StreamClosedError:
            raise StreamClosedError()
        raise error_type("password=secret https://user:secret@host/")

    def fail(*args, **kwargs):
        raise DatabaseError("password=secret https://user:secret@host/")

    monkeypatch.setattr(media, "source_client", lambda: client)
    monkeypatch.setattr(client, "close", close)
    if query_failed:
        monkeypatch.setattr(client, "query_df", fail)
    argv = ["products", "--as-of", AS_OF.isoformat(), "--output", str(tmp_path / "snapshot")]
    if query_failed or error_type is ValueError:
        with pytest.raises(SystemExit) as error:
            media.main(argv)
        assert error.value.code == 1
    else:
        media.main(argv)
    assert client.closed
    assert (tmp_path / "snapshot").exists() is not query_failed
    captured = capsys.readouterr()
    assert "password=secret" not in captured.err and "https://" not in captured.err