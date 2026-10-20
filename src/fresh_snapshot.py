import hashlib
import inspect
import json
import operator
import re
import tempfile
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


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


class _SnapshotError(ValueError):
    pass


class _PublicReportUnavailable(RuntimeError):
    pass


class _RefreshFailure(RuntimeError):
    def __init__(self, error, stage, *, validation_message=None):
        self.validation_error = validation_message is not None or isinstance(error, _SnapshotError)
        name = type(error).__name__
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,79}", name):
            name = "Exception"
        details = f"stage={stage}; exception={name}"
        code = _clickhouse_code(error)
        if code is not None:
            details += f"; clickhouse_code={code}"
        if self.validation_error:
            message = str(error) if validation_message is None else validation_message
            message = f"{message} ({details})"
        else:
            message = (
                f"Не удалось безопасно обновить снимок источника ({details}); повторите запуск."
            )
        super().__init__(message)


def _clickhouse_code(error):
    code = inspect.getattr_static(error, "code", None)
    if type(code) is str and re.fullmatch(r"[0-9]{1,10}", code):
        code = int(code)
    if type(code) is int and 0 < code <= 2**31 - 1:
        return code
    for argument in BaseException.args.__get__(error):
        if type(argument) is str:
            match = re.search(r"\bCode:\s*([0-9]{1,10})(?=[.\s]|$)", argument)
            if match is not None and 0 < int(match[1]) <= 2**31 - 1:
                return int(match[1])
    return None


@contextmanager
def _at_stage(stage):
    try:
        yield
    except (_RefreshFailure, _PublicReportUnavailable):
        raise
    except Exception as error:
        raise _RefreshFailure(error, stage) from None


@contextmanager
def _source_client(factory):
    client = factory()
    try:
        yield client
    except BaseException:
        try:
            client.close()
        except Exception:
            pass
        raise
    else:
        client.close()


def _sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def _invalid_constant(value):
    raise ValueError("non_json_constant")


def _read_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"),
                           parse_constant=_invalid_constant)
    except (OSError, ValueError, UnicodeError, RecursionError):
        raise _SnapshotError("Не удалось проверить JSON: файл отсутствует или повреждён.") from None
    if not isinstance(value, dict):
        raise _SnapshotError("Некорректная структура JSON.")
    return value


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _relative_path(root, value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise _SnapshotError("Некорректный относительный путь.")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise _SnapshotError("Допустимы только относительные пути внутри проекта.")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise _SnapshotError("Путь выходит за пределы проекта.")
    return path


def _snapshot_path(root, value):
    path = _relative_path(root, value)
    if (path.parent != root / "data" or not path.name.startswith("clusters_full_")
            or path.is_symlink()):
        raise _SnapshotError("Некорректный путь снимка в публичном индексе.")
    return path


def _private_files_available(root, protocol):
    paths = []
    for group in ("inputs", "artifacts"):
        entries = protocol.get(group)
        if not isinstance(entries, dict) or not entries:
            raise _SnapshotError("Полный протокол не содержит inputs/artifacts.")
        for entry in entries.values():
            if not isinstance(entry, dict):
                raise _SnapshotError("Некорректные inputs/artifacts полного протокола.")
            paths.append(_relative_path(root, entry.get("path")))
    if not {"facts", "dictionary"}.issubset(protocol["inputs"]):
        raise _SnapshotError("В полном протоколе отсутствуют зафиксированные входы.")
    return all(path.is_file() for path in paths)


def _saved_public_report(root):
    path = root / "reports/notebook_current_data.json"
    if not path.is_file():
        raise _PublicReportUnavailable(
            "Приватные входы или артефакты недоступны, а сохранённый публичный отчёт "
            "отсутствует. Сначала обновите снимок в приватном окружении."
        )
    saved = _read_json(path)
    if "snapshot_path" in saved:
        _snapshot_path(root, saved["snapshot_path"])
    return {**{key: value for key, value in saved.items() if key in PUBLIC_FIELDS},
            "status": "saved_public_report"}


def _utc_now(now):
    if now is None:
        return datetime.now(timezone.utc)
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise _SnapshotError("now должен быть datetime с явным timezone.")
    return now.astimezone(timezone.utc)


def _day(value):
    try:
        if isinstance(value, datetime) and value.tzinfo is not None:
            value = value.astimezone(timezone.utc)
        if isinstance(value, date):
            return date(value.year, value.month, value.day)
        return date.fromisoformat(value)
    except (TypeError, ValueError, OverflowError):
        raise _SnapshotError("Некорректная календарная дата.") from None


def _protocol_day(value):
    try:
        if not isinstance(value, str):
            raise ValueError
        if len(value) == 10:
            return date.fromisoformat(value)
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if stamp.tzinfo is None or stamp.utcoffset() is None:
            raise ValueError
        return stamp.astimezone(timezone.utc).date()
    except (TypeError, ValueError, OverflowError):
        raise _SnapshotError("Некорректная дата полного протокола.") from None


def _integer(value, minimum=0):
    try:
        if isinstance(value, bool):
            raise TypeError
        result = operator.index(value)
    except TypeError:
        raise _SnapshotError("Метаданные должны содержать целые числа.") from None
    if not minimum <= result <= 2**64 - 1:
        raise _SnapshotError("Числовые метаданные вне допустимых границ.")
    return int(result)


def _metadata(client, cap):
    result = client.query(METADATA_SQL, parameters={"end": cap})
    rows = result.result_rows
    if len(rows) != 1 or len(rows[0]) != 4:
        raise _SnapshotError("Источник вернул некорректные метаданные.")
    first, last, count, fingerprint = rows[0]
    first, last = _day(first), _day(last)
    if not first <= last <= cap:
        raise _SnapshotError("Даты источника выходят за границы закрытого периода.")
    return {
        "first_day": first.isoformat(), "last_day": last.isoformat(),
        "rows": _integer(count, minimum=1), "fingerprint": _integer(fingerprint),
    }


def _check_digest(path, expected):
    if (not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected)
            or path.is_symlink() or not path.is_file() or _sha256(path) != expected):
        raise _SnapshotError("SHA256 снимка не совпадает; повреждённые файлы не перезаписываются.")


def _manifest(folder, metadata, dictionary_day, expected_sha256=None):
    path = folder / "source_manifest.json"
    if expected_sha256 is not None:
        _check_digest(path, expected_sha256)
    manifest = _read_json(path)
    files = manifest.get("files")
    if not isinstance(files, dict) or not {"facts.csv", "dictionary.csv"}.issubset(files):
        raise _SnapshotError("Manifest не содержит SHA256 фактов и словаря.")
    for name, expected in files.items():
        if (not isinstance(name, str) or not name or Path(name).name != name
                or name in {".", "..", "source_manifest.json"} or "\\" in name):
            raise _SnapshotError("Некорректный путь файла в manifest.")
        _check_digest(folder / name, expected)
    if _integer(manifest.get("fact_rows"), minimum=1) != metadata["rows"]:
        raise _SnapshotError("Число выгруженных фактов не совпадает с метаданными источника.")
    _integer(manifest.get("dictionary_rows"), minimum=1)
    if (manifest.get("history_start") != metadata["first_day"]
            or manifest.get("cutoff") != metadata["last_day"]
            or manifest.get("dictionary_as_of_day_utc") != dictionary_day.isoformat()
            or manifest.get("window_days") != 14 or manifest.get("training_windows") != 1
            or not isinstance(manifest.get("source_completeness_audit"), dict)):
        raise _SnapshotError("Manifest не соответствует параметрам свежего снимка.")
    return manifest


def _public_index(root, folder, manifest, metadata, protocol, protocol_sha256,
                  checked_at, cap, status):
    last = date.fromisoformat(metadata["last_day"])
    return {
        "status": status,
        "checked_at": checked_at.isoformat(),
        "snapshot_path": folder.relative_to(root).as_posix(),
        "manifest_sha256": _sha256(folder / "source_manifest.json"),
        "protocol_sha256": protocol_sha256,
        "source_fingerprint": metadata,
        "source_rows": metadata["rows"],
        "first_day": metadata["first_day"],
        "last_day": metadata["last_day"],
        "requested_through": cap.isoformat(),
        "missing_trailing_closed_days": [
            (last + timedelta(days=offset)).isoformat()
            for offset in range(1, (cap - last).days + 1)
        ],
        "source_completeness_audit": manifest["source_completeness_audit"],
        "dictionary_as_of": protocol["dictionary_as_of"],
    }


def _refresh(root, now, client_factory):
    protocol_path = root / "reports/notebook_full_protocol.json"
    if not protocol_path.is_file():
        return _saved_public_report(root)
    protocol = _read_json(protocol_path)
    if not _private_files_available(root, protocol):
        return _saved_public_report(root)
    checked_at = _utc_now(now)
    protocol_sha256 = _sha256(protocol_path)
    review = _read_json(root / "reports/notebook_review.json")
    if review.get("full_protocol_sha256") != protocol_sha256:
        raise _SnapshotError("SHA256 полного протокола не совпадает с зафиксированным обзором.")
    cap = min(checked_at.date() - timedelta(days=1), _protocol_day(protocol.get("decision")))
    dictionary_day = _protocol_day(protocol.get("dictionary_as_of"))
    if dictionary_day > cap:
        raise _SnapshotError("Зафиксированный словарь ещё не входит в закрытый период.")
    data = _relative_path(root, "data")
    if data.is_symlink():
        raise _SnapshotError("Временные данные должны оставаться внутри data проекта.")
    data.mkdir(parents=True, exist_ok=True)
    report_path = _relative_path(root, "reports/notebook_current_data.json")
    if report_path.is_symlink():
        raise _SnapshotError("Публичный индекс не должен быть символической ссылкой.")

    if __package__:
        from . import automatic_check
    else:
        import automatic_check

    with automatic_check._private_source_output():
        if __package__:
            from . import cluster_data, notebook_review
        else:
            import cluster_data
            import notebook_review

        try:
            notebook_review.verify_full_protocol(root, protocol)
        except Exception as error:
            raise _RefreshFailure(
                error, "consistency",
                validation_message="Не удалось подтвердить неизменность полного протокола.",
            ) from None
        stage = "metadata"
        try:
            with _source_client(
                cluster_data.source_client if client_factory is None else client_factory
            ) as client:
                metadata = _metadata(client, cap)
                stage = "consistency"
                saved = _read_json(report_path) if report_path.exists() else None
                if saved is not None and saved.get("source_fingerprint") == metadata:
                    folder = _snapshot_path(root, saved.get("snapshot_path"))
                    expected = saved.get("manifest_sha256")
                    if not isinstance(expected, str) or saved.get("protocol_sha256") != protocol_sha256:
                        raise _SnapshotError("Индекс снимка не соответствует полному протоколу.")
                    manifest = _manifest(folder, metadata, dictionary_day, expected)
                    if (manifest.get("source_fingerprint") != metadata
                            or manifest.get("protocol_sha256") != protocol_sha256):
                        raise _SnapshotError("Manifest кеша не соответствует источнику и протоколу.")
                    stage = "publish"
                    report = _public_index(
                        root, folder, manifest, metadata, protocol, protocol_sha256,
                        checked_at, cap, "verified_cached_snapshot",
                    )
                else:
                    identity = hashlib.sha256(json.dumps(
                        metadata, sort_keys=True, separators=(",", ":")
                    ).encode("utf-8")).hexdigest()[:12]
                    folder = data / f"clusters_full_{checked_at:%Y%m%d}_{identity}"
                    if folder.exists() or folder.is_symlink():
                        raise _SnapshotError("Снимок уже существует без проверенного индекса; перезапись запрещена.")
                    stage = "export"
                    with tempfile.TemporaryDirectory(dir=data, prefix=".fresh-snapshot-") as temporary:
                        child = Path(temporary) / "snapshot"
                        cluster_data.export_data(
                            client, cutoff=date.fromisoformat(metadata["last_day"]), output=child,
                            window_days=14, training_windows=1,
                            history_start=date.fromisoformat(metadata["first_day"]),
                            dictionary_as_of=dictionary_day,
                        )
                        stage = "consistency"
                        if _metadata(client, cap) != metadata:
                            raise _SnapshotError("Источник изменился во время выгрузки; повторите запуск.")
                        manifest = _manifest(child, metadata, dictionary_day)
                        stage = "publish"
                        manifest.update(
                            refreshed_at=checked_at.isoformat(), source_fingerprint=metadata,
                            protocol_sha256=protocol_sha256,
                        )
                        _write_json(child / "source_manifest.json", manifest)
                        cluster_data.publish_snapshot(child, folder)
                    report = _public_index(
                        root, folder, manifest, metadata, protocol, protocol_sha256,
                        checked_at, cap, "downloaded",
                    )
        except Exception as error:
            raise _RefreshFailure(error, stage) from None
    with _at_stage("publish"):
        with tempfile.TemporaryDirectory(dir=data, prefix=".fresh-index-") as temporary:
            temporary_report = Path(temporary) / "notebook_current_data.json"
            _write_json(temporary_report, report)
            temporary_report.replace(report_path)
    return report


def refresh_source_snapshot(root, *, now=None, client_factory=None) -> dict:
    """Проверить источник и опубликовать отдельный неизменяемый снимок закрытых дней."""
    try:
        with _at_stage("consistency"):
            return _refresh(Path(root), now, client_factory)
    except _RefreshFailure as error:
        if error.validation_error:
            raise ValueError(str(error)) from None
        raise RuntimeError(str(error)) from None
    except _PublicReportUnavailable as error:
        raise RuntimeError(str(error)) from None