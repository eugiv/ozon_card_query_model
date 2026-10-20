import ast
import hashlib
import importlib.util
import io
import json
import tempfile
import traceback
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import src
import src.fresh_snapshot as fresh_snapshot


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 10, 20, 20, 4, tzinfo=timezone.utc)
SECRET = "source.private.invalid password=never-public-token"
METADATA_SQL = """
SELECT min(date) AS first_day, max(date) AS last_day, count() AS rows,
groupBitXor(cityHash64(date,toString(sku),query,unique_search_users,
unique_view_users,order_count)) AS fingerprint
FROM default_staging.stg_ozon_product_queries_details
WHERE date <= {end:Date}
"""
PUBLIC_FIELDS = {
    "status", "checked_at", "snapshot_path", "manifest_sha256",
    "protocol_sha256", "source_fingerprint", "source_rows", "first_day",
    "last_day", "requested_through", "missing_trailing_closed_days",
    "source_completeness_audit", "dictionary_as_of",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


class IntegerValue:
    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value


class DatabaseError(Exception):
    pass


class SourceClient:
    def __init__(self, owner):
        self.owner = owner
        self.queries = []
        self.closed = False
        self.tls = True
        self.verify = True

    def query(self, sql, *, parameters):
        self.owner.require_private("query")
        self.owner.emit_or_fail("query")
        self.queries.append((sql, parameters.copy()))
        if self.owner.result_rows is not None:
            return SimpleNamespace(result_rows=self.owner.result_rows)
        sequence = self.owner.metadata_sequence
        row = sequence[min(len(self.queries) - 1, len(sequence) - 1)]
        return SimpleNamespace(result_rows=[row])

    def close(self):
        self.owner.require_private("close")
        self.closed = True
        self.owner.emit_or_fail("close")


class FreshSnapshotTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(
            dir=PROJECT_ROOT / "data", prefix=".test-fresh-snapshot-"
        )
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "data").mkdir()
        (self.root / "reports").mkdir()
        self.protocol = {
            "decision": "2026-10-21",
            "dictionary_as_of": "2026-06-26T23:59:59.999999999Z",
            "inputs": {},
            "artifacts": {},
        }
        for group, name, relative in (
            ("inputs", "facts", "data/fixed/facts.csv"),
            ("inputs", "dictionary", "data/fixed/dictionary.csv"),
            ("artifacts", "model", "data/fixed/model.bin"),
        ):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("frozen " + name).encode())
            self.protocol[group][name] = {"path": relative, "sha256": digest(path)}
        self.protocol_path = self.root / "reports/notebook_full_protocol.json"
        self.review_path = self.root / "reports/notebook_review.json"
        self.report_path = self.root / "reports/notebook_current_data.json"
        write_json(self.protocol_path, self.protocol)
        self.protocol_sha256 = digest(self.protocol_path)
        write_json(self.review_path, {"full_protocol_sha256": self.protocol_sha256})
        self.metadata_sequence = [
            (date(2026, 1, 9), date(2026, 10, 17), 2, 2**63 + 17)
        ]
        self.result_rows = None
        self.clients = []
        self.exports = []
        self.publications = []
        self.private_depth = 0
        self.private_calls = []
        self.failure_stage = None
        self.failure_error = ValueError(SECRET)
        self.manifest_overrides = {}
        self.audit = {"missing_days": ["2026-02-01"], "observed_days": 281}
        self.source_factory = Mock(side_effect=self.make_client)
        self.verifier = Mock(side_effect=self.verify_protocol)
        self.cluster_data = SimpleNamespace(
            source_client=self.source_factory,
            export_data=self.export_data,
            publish_snapshot=self.publish_snapshot,
        )
        dependencies = patch.multiple(
            src,
            automatic_check=SimpleNamespace(_private_source_output=self.private_output),
            cluster_data=self.cluster_data,
            notebook_review=SimpleNamespace(verify_full_protocol=self.verifier),
            create=True,
        )
        dependencies.start()
        self.addCleanup(dependencies.stop)

    @contextmanager
    def private_output(self):
        self.private_depth += 1
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                yield
        finally:
            self.private_depth -= 1

    def require_private(self, stage):
        self.assertGreater(self.private_depth, 0, stage)
        self.private_calls.append(stage)

    def emit_or_fail(self, stage):
        print(SECRET)
        print(SECRET, file=__import__("sys").stderr)
        if self.failure_stage == stage:
            raise self.failure_error

    def verify_protocol(self, root, protocol):
        self.require_private("verify_protocol")
        self.assertEqual(Path(root), self.root)
        self.assertEqual(protocol, self.protocol)
        for group in ("inputs", "artifacts"):
            for entry in protocol[group].values():
                if digest(self.root / entry["path"]) != entry["sha256"]:
                    raise ValueError(SECRET)

    def make_client(self):
        self.require_private("factory")
        self.emit_or_fail("factory")
        client = SourceClient(self)
        self.clients.append(client)
        return client

    def export_data(self, client, *, cutoff, output, window_days,
                    training_windows, history_start, dictionary_as_of):
        self.require_private("export")
        self.emit_or_fail("export")
        self.assertIn(client, self.clients)
        self.assertFalse(output.exists())
        self.assertTrue(output.is_relative_to(self.root / "data"))
        self.exports.append({
            "cutoff": cutoff, "output": output, "window_days": window_days,
            "training_windows": training_windows, "history_start": history_start,
            "dictionary_as_of": dictionary_as_of,
        })
        output.mkdir()
        (output / "facts.csv").write_text(
            "date,sku,query,unique_search_users,unique_view_users,order_count\n"
            "2026-01-09,123,private raw query,3,2,1\n"
            "2026-10-17,456,another private query,4,3,2\n", encoding="utf-8"
        )
        (output / "dictionary.csv").write_text(
            "query,normquery\nprivate raw query,private cluster\n", encoding="utf-8"
        )
        manifest = {
            "history_start": history_start.isoformat(),
            "cutoff": cutoff.isoformat(),
            "window_days": window_days,
            "training_windows": training_windows,
            "dictionary_as_of_day_utc": dictionary_as_of.isoformat(),
            "fact_rows": 2,
            "dictionary_rows": 1,
            "source_completeness_audit": self.audit,
            "fact_sql": "SELECT sku, query FROM private_source",
            "files": {name: digest(output / name)
                      for name in ("facts.csv", "dictionary.csv")},
        }
        manifest.update(self.manifest_overrides)
        write_json(output / "source_manifest.json", manifest)
        return manifest

    def publish_snapshot(self, folder, output):
        self.require_private("publish")
        self.emit_or_fail("publish")
        self.assertFalse(output.exists())
        self.assertEqual(output.parent, self.root / "data")
        self.publications.append(output)
        folder.rename(output)

    def refresh(self, **kwargs):
        return fresh_snapshot.refresh_source_snapshot(self.root, now=kwargs.pop("now", NOW),
                                                      **kwargs)

    def read_manifest(self, report):
        return json.loads((self.root / report["snapshot_path"] /
                           "source_manifest.json").read_text(encoding="utf-8"))

    def assert_no_snapshot(self):
        self.assertEqual(
            [path.name for path in (self.root / "data").iterdir()
             if path.name.startswith(("clusters_full_", ".fresh-snapshot-"))], []
        )

    def test_standalone_import_downloads_and_reuses_verified_cache(self):
        spec = importlib.util.spec_from_file_location("fresh_snapshot", fresh_snapshot.__file__)
        standalone = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(standalone)
        self.assertEqual(standalone.__package__, "")
        with patch.dict("sys.modules", {
            "automatic_check": src.automatic_check,
            "cluster_data": self.cluster_data,
            "notebook_review": src.notebook_review,
        }):
            report = standalone.refresh_source_snapshot(self.root, now=NOW)
            cached = standalone.refresh_source_snapshot(
                self.root, now=NOW + timedelta(minutes=30)
            )
        self.assertEqual(set(report), PUBLIC_FIELDS)
        self.assertEqual(report["status"], "downloaded")
        self.assertEqual(cached["status"], "verified_cached_snapshot")
        self.assertEqual(cached["snapshot_path"], report["snapshot_path"])
        self.assertEqual(cached["manifest_sha256"], report["manifest_sha256"])
        self.assertEqual(len(self.exports), 1)
        self.assertEqual(len(self.publications), 1)
        self.assertEqual([len(client.queries) for client in self.clients], [2, 1])
        self.assertTrue(all(client.closed for client in self.clients))

    def test_download_schema_closed_days_and_frozen_inputs(self):
        frozen = {path: path.read_bytes() for path in self.root.rglob("*")
                  if path.is_file()}
        report = self.refresh()
        self.assertEqual(set(report), PUBLIC_FIELDS)
        self.assertEqual(report["status"], "downloaded")
        self.assertEqual(report["checked_at"], NOW.isoformat())
        self.assertEqual(report["first_day"], "2026-01-09")
        self.assertEqual(report["last_day"], "2026-10-17")
        self.assertEqual(report["requested_through"], "2026-10-19")
        self.assertEqual(report["missing_trailing_closed_days"],
                         ["2026-10-18", "2026-10-19"])
        self.assertEqual(report["source_rows"], 2)
        self.assertEqual(report["source_fingerprint"], {
            "first_day": "2026-01-09", "last_day": "2026-10-17",
            "rows": 2, "fingerprint": 2**63 + 17,
        })
        self.assertEqual(report["protocol_sha256"], self.protocol_sha256)
        self.assertEqual(report["dictionary_as_of"], self.protocol["dictionary_as_of"])
        self.assertEqual(report["source_completeness_audit"], self.audit)
        self.assertRegex(report["snapshot_path"],
                         r"^data/clusters_full_20261020_[0-9a-f]{8,16}$")
        snapshot = self.root / report["snapshot_path"]
        self.assertEqual(report["manifest_sha256"], digest(snapshot / "source_manifest.json"))
        manifest = self.read_manifest(report)
        self.assertEqual(manifest["refreshed_at"], NOW.isoformat())
        self.assertEqual(manifest["source_fingerprint"], report["source_fingerprint"])
        self.assertEqual(manifest["protocol_sha256"], self.protocol_sha256)
        for name, expected in manifest["files"].items():
            self.assertEqual(digest(snapshot / name), expected)
        self.assertEqual(len(self.exports), 1)
        self.assertEqual({key: value for key, value in self.exports[0].items()
                          if key != "output"}, {
            "cutoff": date(2026, 10, 17), "window_days": 14, "training_windows": 1,
            "history_start": date(2026, 1, 9), "dictionary_as_of": date(2026, 6, 26),
        })
        self.assertEqual(len(self.clients[0].queries), 2)
        for sql, parameters in self.clients[0].queries:
            self.assertEqual("".join(sql.split()), "".join(METADATA_SQL.split()))
            self.assertEqual(parameters, {"end": date(2026, 10, 19)})
        self.assertTrue(self.clients[0].closed)
        self.source_factory.assert_called_once_with()
        self.verifier.assert_called_once_with(self.root, self.protocol)
        self.assertEqual(json.loads(self.report_path.read_text()), report)
        public = json.dumps(report)
        for private in (str(self.root), SECRET, "private raw query", "SELECT", "sku"):
            self.assertNotIn(private, public)
        for path, original in frozen.items():
            self.assertEqual(path.read_bytes(), original)

    def test_cache_checks_source_and_digests_without_export_or_rewrite(self):
        first = self.refresh()
        snapshot = self.root / first["snapshot_path"]
        original = {path.name: path.read_bytes() for path in snapshot.iterdir()}
        later = NOW + timedelta(minutes=30)
        cached = self.refresh(now=later)
        self.assertEqual(cached["status"], "verified_cached_snapshot")
        self.assertEqual(cached["checked_at"], later.isoformat())
        self.assertEqual(cached["snapshot_path"], first["snapshot_path"])
        self.assertEqual(cached["manifest_sha256"], first["manifest_sha256"])
        self.assertEqual(len(self.exports), 1)
        self.assertEqual(len(self.publications), 1)
        self.assertEqual(len(self.clients[-1].queries), 1)
        self.assertTrue(all(client.closed for client in self.clients))
        self.assertEqual({path.name: path.read_bytes() for path in snapshot.iterdir()}, original)

    def test_unchanged_source_reuses_previous_utc_day_snapshot(self):
        first = self.refresh()
        cached = self.refresh(now=NOW + timedelta(days=1))
        self.assertEqual(cached["status"], "verified_cached_snapshot")
        self.assertEqual(cached["snapshot_path"], first["snapshot_path"])
        self.assertEqual(cached["requested_through"], "2026-10-20")
        self.assertEqual(cached["missing_trailing_closed_days"],
                         ["2026-10-18", "2026-10-19", "2026-10-20"])
        self.assertEqual(len(self.exports), 1)

    def test_fingerprint_change_publishes_another_immutable_version(self):
        first = self.refresh()
        old_manifest = (self.root / first["snapshot_path"] / "source_manifest.json").read_bytes()
        row = self.metadata_sequence[0]
        self.metadata_sequence = [(*row[:3], row[3] + 1)]
        second = self.refresh()
        self.assertEqual(second["status"], "downloaded")
        self.assertNotEqual(first["snapshot_path"], second["snapshot_path"])
        self.assertEqual(len(self.exports), 2)
        self.assertEqual((self.root / first["snapshot_path"] /
                          "source_manifest.json").read_bytes(), old_manifest)

    def test_utc_yesterday_not_local_yesterday(self):
        local_now = datetime(2026, 10, 20, 1, tzinfo=timezone(timedelta(hours=3)))
        report = self.refresh(now=local_now)
        self.assertEqual(report["requested_through"], "2026-10-18")
        self.assertEqual(report["checked_at"], "2026-10-19T22:00:00+00:00")
        self.assertIn("clusters_full_20261019_", report["snapshot_path"])
        self.assertEqual(report["missing_trailing_closed_days"], ["2026-10-18"])

    def test_after_decision_never_requests_prospective_target_rows(self):
        report = self.refresh(now=datetime(2026, 11, 10, tzinfo=timezone.utc))
        self.assertEqual(report["requested_through"], "2026-10-21")
        for client in self.clients:
            for _, parameters in client.queries:
                self.assertEqual(parameters["end"], date(2026, 10, 21))
        self.assertLessEqual(self.exports[0]["cutoff"], date(2026, 10, 21))
        self.assertNotIn("2026-10-22", report["missing_trailing_closed_days"])

    def test_no_missing_trailing_days_when_source_reaches_cap(self):
        self.metadata_sequence = [(date(2026, 1, 9), date(2026, 10, 19), 2, 19)]
        self.assertEqual(self.refresh()["missing_trailing_closed_days"], [])

    def test_naive_now_is_rejected_before_source_access(self):
        with self.assertRaises(ValueError):
            self.refresh(now=datetime(2026, 10, 20))
        self.source_factory.assert_not_called()

    def test_indexable_source_integers_are_json_safe(self):
        self.metadata_sequence = [(date(2026, 1, 9), date(2026, 10, 17),
                                   IntegerValue(2), IntegerValue(2**64 - 1))]
        report = self.refresh()
        self.assertIs(type(report["source_rows"]), int)
        self.assertIs(type(report["source_fingerprint"]["fingerprint"]), int)
        json.dumps(report, allow_nan=False)

    def test_pandas_dates_and_numpy_integers_are_normalized(self):
        import numpy as np
        import pandas as pd

        self.metadata_sequence = [
            (pd.Timestamp("2026-01-09"), pd.Timestamp("2026-10-17"),
             np.uint64(2), np.uint64(2**64 - 1))
        ]
        report = self.refresh()
        self.assertEqual(report["first_day"], "2026-01-09")
        self.assertEqual(report["last_day"], "2026-10-17")
        self.assertIs(type(report["source_rows"]), int)
        self.assertIs(type(report["source_fingerprint"]["fingerprint"]), int)
        json.dumps(report, allow_nan=False)

    def test_pandas_nat_is_a_validation_error_before_export(self):
        import pandas as pd

        valid = self.metadata_sequence[0]
        for row in ((pd.NaT, *valid[1:]), (valid[0], pd.NaT, *valid[2:])):
            with self.subTest(row=row):
                self.metadata_sequence = [row]
                with self.assertRaises(ValueError) as caught:
                    self.refresh()
                self.assertIn("stage=metadata", str(caught.exception))
                self.assertTrue(self.clients[-1].closed)
        self.assertEqual(self.exports, [])
        self.assert_no_snapshot()

    def test_invalid_metadata_never_exports(self):
        valid = self.metadata_sequence[0]
        invalid_rows = [
            (None, valid[1], 2, 1),
            (valid[0], None, 2, 1),
            (valid[1], valid[0], 2, 1),
            (valid[0], date(2026, 10, 20), 2, 1),
            (valid[0], valid[1], 0, 1),
            (valid[0], valid[1], -1, 1),
            (valid[0], valid[1], 2.5, 1),
            (valid[0], valid[1], True, 1),
            (valid[0], valid[1], 2, -1),
            (valid[0], valid[1], 2, 2**64),
        ]
        for row in invalid_rows:
            with self.subTest(row=row):
                self.metadata_sequence = [row]
                with self.assertRaises(ValueError):
                    self.refresh()
                self.assertTrue(self.clients[-1].closed)
        self.assertEqual(self.exports, [])
        self.assert_no_snapshot()

    def test_target_dates_returned_despite_where_clause_are_rejected(self):
        self.metadata_sequence = [(date(2026, 1, 9), date(2026, 10, 22), 2, 1)]
        with self.assertRaises(ValueError):
            self.refresh(now=datetime(2026, 11, 10, tzinfo=timezone.utc))
        self.assertEqual(self.exports, [])
        self.assertTrue(self.clients[-1].closed)

    def test_malformed_metadata_result_is_rejected(self):
        for rows in ([], [(date(2026, 1, 9),)], [self.metadata_sequence[0]] * 2):
            with self.subTest(rows=rows):
                self.result_rows = rows
                with self.assertRaises(ValueError):
                    self.refresh()
                self.assertTrue(self.clients[-1].closed)
        self.assertEqual(self.exports, [])

    def test_source_changes_during_export_never_publish(self):
        before = self.metadata_sequence[0]
        changes = [(*before[:3], before[3] + 1),
                   (before[0], before[1], 3, before[3]),
                   (date(2026, 1, 10), *before[1:]),
                   (before[0], date(2026, 10, 18), *before[2:])]
        for after in changes:
            with self.subTest(after=after):
                self.metadata_sequence = [before, after]
                with self.assertRaisesRegex(ValueError, "[Пп]овтор"):
                    self.refresh()
                self.assertTrue(self.clients[-1].closed)
                self.assert_no_snapshot()
                self.assertFalse(self.report_path.exists())
        self.assertEqual(self.publications, [])

    def test_fact_count_must_match_metadata(self):
        self.manifest_overrides["fact_rows"] = 3
        with self.assertRaises(ValueError) as caught:
            self.refresh()
        self.assertIn("stage=consistency", str(caught.exception))
        self.assertIn("exception=_SnapshotError", str(caught.exception))
        self.assert_no_snapshot()
        self.assertFalse(self.report_path.exists())
        self.assertTrue(self.clients[-1].closed)

    def test_export_manifest_must_hash_both_data_files(self):
        self.manifest_overrides["files"] = {}
        with self.assertRaises(ValueError):
            self.refresh()
        self.assert_no_snapshot()

    def assert_corrupt_cache_fails(self, name):
        first = self.refresh()
        path = self.root / first["snapshot_path"] / name
        path.write_bytes(path.read_bytes() + b"corrupted")
        damaged = path.read_bytes()
        report_bytes = self.report_path.read_bytes()
        with self.assertRaises(ValueError):
            self.refresh()
        self.assertEqual(path.read_bytes(), damaged)
        self.assertEqual(self.report_path.read_bytes(), report_bytes)
        self.assertEqual(len(self.exports), 1)
        self.assertTrue(self.clients[-1].closed)

    def test_corrupt_cached_facts_are_not_overwritten(self):
        self.assert_corrupt_cache_fails("facts.csv")

    def test_corrupt_cached_dictionary_is_not_overwritten(self):
        self.assert_corrupt_cache_fails("dictionary.csv")

    def test_corrupt_cached_manifest_is_not_overwritten(self):
        self.assert_corrupt_cache_fails("source_manifest.json")

    def test_missing_cached_snapshot_is_not_silently_recreated(self):
        first = self.refresh()
        (self.root / first["snapshot_path"] / "facts.csv").unlink()
        with self.assertRaises(ValueError):
            self.refresh()
        self.assertEqual(len(self.exports), 1)

    def test_manifest_sha_is_checked_against_public_index(self):
        first = self.refresh()
        first["manifest_sha256"] = "0" * 64
        write_json(self.report_path, first)
        with self.assertRaises(ValueError):
            self.refresh()
        self.assertEqual(len(self.exports), 1)

    def test_cache_path_cannot_escape_project_data(self):
        first = self.refresh()
        first["snapshot_path"] = str(self.root / first["snapshot_path"])
        write_json(self.report_path, first)
        with self.assertRaises(ValueError) as caught:
            self.refresh()
        self.assertNotIn(str(self.root), str(caught.exception))
        self.assertEqual(len(self.exports), 1)

    def test_changed_protocol_sha_stops_before_network_or_verifier(self):
        self.protocol_path.write_bytes(self.protocol_path.read_bytes() + b" ")
        with self.assertRaises(ValueError):
            self.refresh()
        self.source_factory.assert_not_called()
        self.verifier.assert_not_called()

    def test_full_protocol_verification_precedes_network_and_is_sanitized(self):
        self.verifier.side_effect = ValueError(SECRET)
        with self.assertRaises(ValueError) as caught:
            self.refresh()
        self.source_factory.assert_not_called()
        self.assertNotIn(SECRET, "".join(traceback.format_exception(caught.exception)))

    def test_changed_frozen_private_file_is_rejected_by_verifier(self):
        (self.root / "data/fixed/dictionary.csv").write_text("changed", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.refresh()
        self.source_factory.assert_not_called()

    def saved_report(self):
        saved = {
            "status": "downloaded", "checked_at": "2026-10-11T12:00:00+00:00",
            "first_day": "2026-01-09", "last_day": "2026-10-07",
            "snapshot_path": "data/clusters_full_20261011_123456789abc",
            "source_rows": 2,
        }
        write_json(self.report_path, saved)
        return saved

    def test_public_checkout_returns_saved_report_without_network_or_fresh_date(self):
        saved = self.saved_report()
        original = self.report_path.read_bytes()
        (self.root / "data/fixed/model.bin").unlink()
        factory = Mock(side_effect=AssertionError("must not connect"))
        report = self.refresh(client_factory=factory)
        self.assertEqual(report, {**saved, "status": "saved_public_report"})
        self.assertEqual(self.report_path.read_bytes(), original)
        factory.assert_not_called()
        self.source_factory.assert_not_called()
        self.verifier.assert_not_called()

    def test_missing_private_input_also_uses_saved_public_report(self):
        self.saved_report()
        (self.root / "data/fixed/facts.csv").unlink()
        self.assertEqual(self.refresh()["status"], "saved_public_report")
        self.source_factory.assert_not_called()

    def test_saved_public_report_alone_is_sufficient(self):
        saved = self.saved_report()
        self.protocol_path.unlink()
        self.review_path.unlink()
        self.assertEqual(self.refresh(), {**saved, "status": "saved_public_report"})
        self.source_factory.assert_not_called()

    def test_public_checkout_without_saved_report_has_actionable_error(self):
        (self.root / "data/fixed/model.bin").unlink()
        with self.assertRaisesRegex(RuntimeError, "[Оо]тч[её]т") as caught:
            self.refresh()
        self.assertNotIn(str(self.root), str(caught.exception))
        self.source_factory.assert_not_called()

    def test_source_failures_and_prints_never_leak_credentials(self):
        stages = {"factory": "metadata", "query": "metadata", "export": "export",
                  "publish": "publish", "close": "publish"}
        for stage, diagnostic_stage in stages.items():
            with self.subTest(stage=stage):
                self.failure_stage = stage
                stdout, stderr = io.StringIO(), io.StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    with self.assertRaises(RuntimeError) as caught:
                        self.refresh()
                exposed = stdout.getvalue() + stderr.getvalue()
                exposed += "".join(traceback.format_exception(caught.exception))
                self.assertNotIn(SECRET, exposed)
                self.assertNotIn("never-public-token", exposed)
                self.assertNotIn(str(self.root), str(caught.exception))
                self.assertIn(f"stage={diagnostic_stage}", str(caught.exception))
                self.assertIn("exception=ValueError", str(caught.exception))
                if self.clients:
                    self.assertTrue(self.clients[-1].closed)

    def test_clickhouse_code_is_extracted_without_raw_messages(self):
        errors = [DatabaseError(SECRET),
                  DatabaseError(f"{SECRET} Code: 60. DB::Exception: {SECRET}"),
                  DatabaseError(SECRET)]
        errors[0].code = 60
        errors[2].code = "60"
        self.failure_stage = "export"
        for error in errors:
            with self.subTest(error_type=type(error).__name__, args=error.args):
                self.failure_error = error
                with self.assertRaises(RuntimeError) as caught:
                    self.refresh()
                message = str(caught.exception)
                self.assertIn("stage=export", message)
                self.assertIn("exception=DatabaseError", message)
                self.assertIn("clickhouse_code=60", message)
                self.assertNotIn(SECRET, "".join(traceback.format_exception(caught.exception)))
                self.assertTrue(caught.exception.__suppress_context__)
                self.assertTrue(self.clients[-1].closed)
                self.assert_no_snapshot()

    def test_exception_class_and_invalid_codes_cannot_leak_secrets(self):
        self.failure_stage = "export"
        for code in (SECRET, True, -1, 2**64, object()):
            with self.subTest(code_type=type(code).__name__):
                self.failure_error = type(SECRET, (Exception,), {})(SECRET)
                self.failure_error.code = code
                with self.assertRaises(RuntimeError) as caught:
                    self.refresh()
                self.assertIn("exception=Exception", str(caught.exception))
                self.assertNotIn("clickhouse_code=", str(caught.exception))
                self.assertNotIn(SECRET, "".join(traceback.format_exception(caught.exception)))

    def test_http_status_is_not_a_clickhouse_code(self):
        self.failure_stage = "query"
        self.failure_error = DatabaseError(f"{SECRET} HTTP response code 403")
        with self.assertRaises(RuntimeError) as caught:
            self.refresh()
        self.assertIn("stage=metadata", str(caught.exception))
        self.assertIn("exception=DatabaseError", str(caught.exception))
        self.assertNotIn("clickhouse_code=", str(caught.exception))

    def test_error_string_is_not_evaluated_for_diagnostics(self):
        class UnprintableError(Exception):
            def __str__(self):
                raise AssertionError("Нельзя читать сырое сообщение")

        self.failure_stage = "query"
        self.failure_error = UnprintableError(f"{SECRET} Code: 47. {SECRET}")
        with self.assertRaises(RuntimeError) as caught:
            self.refresh()
        self.assertIn("exception=UnprintableError", str(caught.exception))
        self.assertIn("clickhouse_code=47", str(caught.exception))
        self.assertNotIn(SECRET, "".join(traceback.format_exception(caught.exception)))

    def test_second_metadata_query_failure_has_consistency_stage(self):
        def export_then_fail_query(*args, **kwargs):
            manifest = self.export_data(*args, **kwargs)
            self.failure_stage = "query"
            return manifest

        self.cluster_data.export_data = export_then_fail_query
        self.failure_error = DatabaseError(f"{SECRET} Code: 47. {SECRET}")
        with self.assertRaises(RuntimeError) as caught:
            self.refresh()
        self.assertIn("stage=consistency", str(caught.exception))
        self.assertIn("exception=DatabaseError", str(caught.exception))
        self.assertIn("clickhouse_code=47", str(caught.exception))
        self.assertEqual(len(self.exports), 1)
        self.assertEqual(self.publications, [])
        self.assertFalse(self.report_path.exists())
        self.assert_no_snapshot()

    def test_close_failure_does_not_mask_primary_export_failure(self):
        close = SourceClient.close

        def failing_close(client):
            close(client)
            raise OSError(SECRET)

        self.failure_stage = "export"
        self.failure_error = DatabaseError(f"{SECRET} Code: 60. {SECRET}")
        with patch.object(SourceClient, "close", failing_close):
            with self.assertRaises(RuntimeError) as caught:
                self.refresh()
        self.assertIn("stage=export", str(caught.exception))
        self.assertIn("exception=DatabaseError", str(caught.exception))
        self.assertIn("clickhouse_code=60", str(caught.exception))
        self.assertTrue(self.clients[-1].closed)
        self.assertNotIn(SECRET, "".join(traceback.format_exception(caught.exception)))

    def test_public_index_write_failure_has_publish_stage(self):
        write = fresh_snapshot._write_json

        def fail_index(path, value):
            if path.name == "notebook_current_data.json":
                raise PermissionError(SECRET)
            return write(path, value)

        with patch.object(fresh_snapshot, "_write_json", side_effect=fail_index):
            with self.assertRaises(RuntimeError) as caught:
                self.refresh()
        self.assertIn("stage=publish", str(caught.exception))
        self.assertIn("exception=PermissionError", str(caught.exception))
        self.assertNotIn(SECRET, "".join(traceback.format_exception(caught.exception)))
        self.assertTrue(self.clients[-1].closed)
        self.assertFalse(self.report_path.exists())

    def test_injected_factory_is_used_inside_private_output(self):
        factory = Mock(side_effect=self.make_client)
        self.refresh(client_factory=factory)
        factory.assert_called_once_with()
        self.source_factory.assert_not_called()
        self.assertTrue(self.clients[-1].closed)

    def test_tls_and_certificate_verification_are_never_disabled(self):
        self.refresh()
        self.assertTrue(self.clients[-1].tls)
        self.assertTrue(self.clients[-1].verify)
        tree = ast.parse(Path(fresh_snapshot.__file__).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if keyword.arg in {"verify", "tls", "secure"}:
                        self.assertFalse(isinstance(keyword.value, ast.Constant)
                                         and keyword.value.value is False)


if __name__ == "__main__":
    unittest.main()