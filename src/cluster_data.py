"""Снимки WB-кластеров и текста Ozon, выгрузка и общие артефакты рекомендателя."""

from __future__ import annotations

import argparse
import csv
import ctypes
import errno
import hashlib
import json
import os
import re
import sys
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from datetime import date, timedelta
from http.client import HTTPException
from pathlib import Path

import clickhouse_connect
import numpy as np
import pandas as pd
from clickhouse_connect.driver.exceptions import ClickHouseError
from dotenv import dotenv_values
from urllib3.exceptions import HTTPError as Urllib3HTTPError

FACT_COLUMNS = [
    "date",
    "sku",
    "query",
    "unique_search_users",
    "unique_view_users",
    "order_count",
]
DAILY_COLUMNS = [
    "date",
    "sku",
    "normquery",
    "searches",
    "views",
    "orders",
    "cluster_searches",
]

ROOT = Path(__file__).resolve().parents[1]
FACT_SQL = """
SELECT date, toString(sku) AS sku, query,
       unique_search_users, unique_view_users, order_count
FROM default_staging.stg_ozon_product_queries_details
WHERE date BETWEEN {start:Date} AND {end:Date}
ORDER BY date, sku, query
"""
DICTIONARY_SQL = """
SELECT query, normquery, status_code, error,
       max(toTimeZone(d.loaded_at, 'UTC')) AS loaded_at
FROM raw.wb_normquery AS d
WHERE d.loaded_at < toDateTime({snapshot_end:Date}, 'UTC') + INTERVAL 1 DAY
GROUP BY query, normquery, status_code, error
ORDER BY query, loaded_at, normquery
"""
PRODUCT_SQL = """
SELECT sku, name, attributes, loaded_at
FROM (
    SELECT toString(sku) AS sku, name, attributes,
           toTimeZone(loaded_at, 'UTC') AS loaded_at,
           row_number() OVER (
               PARTITION BY sku ORDER BY loaded_at DESC, name ASC, attributes ASC
           ) AS row_number
    FROM raw.ozon_product_info_attributes
    WHERE loaded_at <= {as_of:DateTime64(6, 'UTC')}
)
WHERE row_number = 1
ORDER BY sku
"""
METADATA_FIELDS = [
    "sku", "name", "brand", "category", "description", "attributes_json",
    "snapshot_at", "attributes_status", "attributes_error",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Отсутствуют колонки: {', '.join(missing)}")


def clean_text(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("").str.strip().str.lower()


def dictionary_snapshot(
    records: pd.DataFrame, as_of: pd.Timestamp
) -> tuple[pd.DataFrame, dict]:
    require_columns(
        records, ["query", "normquery", "loaded_at", "status_code", "error"]
    )
    data = records.copy()
    data["clean_query"] = clean_text(data["query"])
    data["normquery"] = data["normquery"].astype("string").fillna("").str.strip()
    data["loaded_at"] = pd.to_datetime(data["loaded_at"], utc=True, errors="raise")
    if data["loaded_at"].isna().any():
        raise ValueError("В словаре отсутствует loaded_at; временной снимок невозможен")
    cutoff = pd.Timestamp(as_of)
    cutoff = (
        cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    )
    historical = data.loc[data["loaded_at"] <= cutoff]
    valid = historical.loc[
        historical["clean_query"].ne("")
        & historical["normquery"].ne("")
        & pd.to_numeric(historical["status_code"], errors="coerce").eq(200)
        & historical["error"].fillna("").astype(str).str.strip().eq("")
    ]
    variants = valid.groupby("clean_query")["normquery"].nunique()
    snapshot = (
        valid.sort_values(["clean_query", "loaded_at", "normquery"], kind="stable")
        .drop_duplicates("clean_query", keep="last")[
            ["clean_query", "normquery", "loaded_at"]
        ]
        .sort_values("clean_query")
        .reset_index(drop=True)
    )
    audit = {
        "as_of": cutoff.isoformat(),
        "input_rows": len(records),
        "future_rows_ignored": int((data["loaded_at"] > cutoff).sum()),
        "invalid_historical_rows": len(historical) - len(valid),
        "ambiguous_queries": int((variants > 1).sum()),
        "mapped_queries": len(snapshot),
        "clusters": int(snapshot["normquery"].nunique()),
        "selection": "latest successful mapping; ties resolved by normquery ascending",
        "algorithm_version_known": False,
    }
    return snapshot, audit


def aggregate_clusters(
    facts: pd.DataFrame, snapshot: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    require_columns(facts, FACT_COLUMNS)
    require_columns(snapshot, ["clean_query", "normquery"])
    if snapshot["clean_query"].duplicated().any():
        raise ValueError("Снимок должен содержать ровно одно соответствие для query")
    data = facts[FACT_COLUMNS].copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise").dt.normalize()
    data["sku"] = data["sku"].astype("string").fillna("").str.strip()
    data["clean_query"] = clean_text(data["query"])
    if data["date"].isna().any() or data["sku"].eq("").any():
        raise ValueError("Дата и SKU обязательны в каждой строке")
    for column in ["unique_search_users", "unique_view_users", "order_count"]:
        values = pd.to_numeric(data[column], errors="raise").astype(float)
        if (
            not np.isfinite(values).all()
            or (values < 0).any()
            or (values % 1 != 0).any()
        ):
            raise ValueError(
                f"{column}: ожидаются конечные неотрицательные целые числа"
            )
        data[column] = values
    empty = data["clean_query"].eq("")
    empty_orders = float(data.loc[empty, "order_count"].sum())
    data = data.loc[~empty].copy()
    if data.duplicated(["date", "sku", "query"]).any():
        raise ValueError(
            "Дубли date + sku + query: проверьте источник, нельзя повторно суммировать воронку"
        )
    frequency = data.groupby(["date", "clean_query"], as_index=False)[
        "unique_search_users"
    ].max()
    raw = data.groupby(["date", "sku", "clean_query"], as_index=False).agg(
        views=("unique_view_users", "sum"),
        orders=("order_count", "sum"),
    )
    raw = raw.merge(frequency, on=["date", "clean_query"], validate="many_to_one")
    joined = raw.merge(
        snapshot[["clean_query", "normquery"]],
        on="clean_query",
        how="left",
        validate="many_to_one",
    )
    matched = joined["normquery"].notna()
    total_orders = float(joined["orders"].sum())
    covered_orders = float(joined.loc[matched, "orders"].sum())
    mapped = joined.loc[matched]
    audit = {
        "input_rows": len(facts),
        "empty_query_rows": int(empty.sum()),
        "empty_query_orders": empty_orders,
        "raw_pairs": int(raw[["sku", "clean_query"]].drop_duplicates().shape[0]),
        "mapped_pairs": int(mapped[["sku", "clean_query"]].drop_duplicates().shape[0]),
        "unmapped_queries": sorted(
            joined.loc[~matched, "clean_query"].unique().tolist()
        ),
        "orders": total_orders,
        "covered_orders": covered_orders,
        "order_coverage": covered_orders / total_orders if total_orders else None,
        "orders_without_views_daily_pairs": int(
            ((raw["orders"] > 0) & (raw["views"] == 0)).sum()
        ),
    }
    if mapped.empty:
        return pd.DataFrame(columns=DAILY_COLUMNS), audit
    demand = (
        mapped.drop_duplicates(["date", "clean_query"])
        .groupby(
            ["date", "normquery"],
            as_index=False,
        )
        .agg(cluster_searches=("unique_search_users", "sum"))
    )
    daily = mapped.groupby(["date", "sku", "normquery"], as_index=False).agg(
        searches=("unique_search_users", "sum"),
        views=("views", "sum"),
        orders=("orders", "sum"),
    )
    daily = daily.merge(demand, on=["date", "normquery"], validate="many_to_one")
    return daily[DAILY_COLUMNS].sort_values(["date", "sku", "normquery"]).reset_index(
        drop=True
    ), audit


def period_pairs(daily: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    part = daily.loc[daily["date"].between(pd.Timestamp(start), pd.Timestamp(end))]
    return part.groupby(["sku", "normquery"], as_index=False).agg(
        searches=("searches", "sum"),
        views=("views", "sum"),
        orders=("orders", "sum"),
        days=("date", "nunique"),
    )


@dataclass(frozen=True)
class RecommendationWindow:
    history_start: date
    history_end: date
    target_start: date
    target_end: date

    def to_dict(self) -> dict:
        return {key: value.isoformat() for key, value in vars(self).items()}


def recommendation_windows(
    cutoff: date, window_days: int = 28
) -> dict[str, RecommendationWindow]:
    if (
        isinstance(window_days, bool)
        or not isinstance(window_days, int)
        or window_days < 1
    ):
        raise ValueError("window_days должен быть положительным целым числом")
    start = cutoff - timedelta(days=4 * window_days - 1)
    windows = {}
    for offset, name in enumerate(["train", "validation", "test"]):
        history_start = start + timedelta(days=offset * window_days)
        history_end = history_start + timedelta(days=window_days - 1)
        windows[name] = RecommendationWindow(
            history_start,
            history_end,
            history_end + timedelta(days=1),
            history_end + timedelta(days=window_days),
        )
    return windows


def rolling_training_windows(
    cutoff: date, window_days: int = 28, count: int = 1
) -> list[RecommendationWindow]:
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("Число обучающих окон должно быть положительным целым")
    base = recommendation_windows(cutoff, window_days)["train"]
    return [
        RecommendationWindow(**{
            key: value - timedelta(days=offset * window_days)
            for key, value in vars(base).items()
        })
        for offset in reversed(range(count))
    ]


def future_positives(history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
    seeds = history.loc[history["orders"] > 0, ["sku", "normquery"]]
    eligible = set(seeds["sku"])
    sold = future.loc[
        (future["orders"] > 0) & future["sku"].isin(eligible), ["sku", "normquery"]
    ]
    return (
        sold.merge(seeds.assign(_seed=True), on=["sku", "normquery"], how="left")
        .loc[
            lambda frame: frame["_seed"].isna(),
            ["sku", "normquery"],
        ]
        .reset_index(drop=True)
    )


def source_client():
    env_path = Path(os.environ.get("OZON_CARD_ENV", ROOT / ".env"))
    env = {**dotenv_values(env_path), **os.environ}
    if not env.get("DBT_CLICKHOUSE_HOST"):
        raise ValueError(
            "DBT_CLICKHOUSE_HOST не задан; заполните .env проекта, "
            "укажите OZON_CARD_ENV или переменные окружения"
        )
    return clickhouse_connect.get_client(
        host=env["DBT_CLICKHOUSE_HOST"],
        port=int(env.get("DBT_CLICKHOUSE_PORT", "8123")),
        username=env.get("DBT_CLICKHOUSE_USER", "default"),
        password=env.get("DBT_CLICKHOUSE_PASSWORD", ""),
        database=env.get("DBT_CLICKHOUSE_SCHEMA") or None,
        secure=str(env.get("DBT_CLICKHOUSE_SECURE", "false")).lower() == "true",
        verify=str(env.get("DBT_CLICKHOUSE_VERIFY", "true")).lower() == "true",
        ca_cert=env.get("DBT_CLICKHOUSE_CA_CERT") or None,
        client_cert=env.get("DBT_CLICKHOUSE_CLIENT_CERT") or None,
        client_cert_key=env.get("DBT_CLICKHOUSE_CLIENT_KEY") or None,
        connect_timeout=8,
        send_receive_timeout=120,
        settings={
            "readonly": 1,
            "max_result_rows": 0,
            "max_result_bytes": 0,
            "result_overflow_mode": "throw",
        },
    )


def export_dates(
    cutoff: date, window_days: int, training_windows: int,
    history_start: date | None, dictionary_as_of: date | None,
) -> tuple[date, date]:
    for name, value in (
        ("cutoff", cutoff), ("history_start", history_start),
        ("dictionary_as_of", dictionary_as_of),
    ):
        if (name == "cutoff" or value is not None) and type(value) is not date:
            raise ValueError(f"{name} должен быть date")
    for name, value in (("window_days", window_days), ("training_windows", training_windows)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} должен быть положительным целым числом")
    if history_start is not None and history_start > cutoff:
        raise ValueError("history_start не может быть позже cutoff")
    if dictionary_as_of is not None and dictionary_as_of > cutoff:
        raise ValueError("dictionary_as_of не может быть позже cutoff")
    if history_start is None or dictionary_as_of is None:
        try:
            first = rolling_training_windows(cutoff, window_days, training_windows)[0]
        except (OverflowError, ValueError) as error:
            raise ValueError("Даты rolling windows вне допустимого диапазона") from error
        if history_start is None:
            history_start = first.history_start
        if dictionary_as_of is None:
            dictionary_as_of = first.history_end
    if dictionary_as_of == date.max:
        raise ValueError("dictionary_as_of не допускает вычисления следующего дня")
    return history_start, dictionary_as_of


def source_completeness_audit(facts: pd.DataFrame, start: date, cutoff: date) -> dict:
    daily_facts = facts[["date", "sku", "order_count"]].copy()
    daily_facts["date"] = pd.to_datetime(daily_facts["date"], errors="raise").dt.date
    if daily_facts["date"].isna().any() or not daily_facts["date"].between(start, cutoff).all():
        raise ValueError("Даты фактов отсутствуют или вне запрошенного диапазона")
    observed = set(daily_facts["date"])
    expected_day_count = (cutoff - start).days + 1
    missing_dates = [
        (start + timedelta(days=offset)).isoformat()
        for offset in range(expected_day_count)
        if start + timedelta(days=offset) not in observed
    ]

    def counts(frame: pd.DataFrame) -> dict:
        return {
            "rows": len(frame),
            "sku_count": int(frame["sku"].nunique()),
            "order_count": int(frame["order_count"].sum()),
        }

    daily = [
        {"date": day.isoformat(), **counts(frame)}
        for day, frame in daily_facts.groupby("date", sort=True)
    ]
    daily_facts["month"] = daily_facts["date"].map(lambda day: day.strftime("%Y-%m"))
    monthly = [
        {"month": month, **counts(frame)}
        for month, frame in daily_facts.groupby("month", sort=True)
    ]
    return {
        **counts(daily_facts),
        "date_min": min(observed).isoformat(),
        "date_max": max(observed).isoformat(),
        "day_count": len(observed),
        "expected_day_count": expected_day_count,
        "missing_dates": missing_dates,
        "daily": daily,
        "monthly": monthly,
    }


def export_data(ch, cutoff: date, output: Path, window_days: int = 28,
                training_windows: int = 1, history_start: date | None = None,
                dictionary_as_of: date | None = None) -> dict:
    start, snapshot = export_dates(
        cutoff, window_days, training_windows, history_start, dictionary_as_of
    )
    if output.exists():
        raise ValueError("Папка выгрузки уже существует; выберите новую")
    facts = ch.query_df(
        FACT_SQL, parameters={"start": start, "end": cutoff}
    )
    dictionary = ch.query_df(
        DICTIONARY_SQL, parameters={"snapshot_end": snapshot}
    )
    if facts.empty or dictionary.empty:
        raise ValueError(
            "Выгрузка пуста: проверьте наличие фактов и исторических записей словаря"
        )
    completeness = source_completeness_audit(facts, start, cutoff)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output.parent, prefix=".cluster-export-"
    ) as temporary:
        folder = Path(temporary) / "data"
        folder.mkdir()
        facts.to_csv(folder / "facts.csv", index=False)
        dictionary.to_csv(folder / "dictionary.csv", index=False)
        manifest = {
            "sources": [
                "default_staging.stg_ozon_product_queries_details",
                "raw.wb_normquery",
            ],
            "history_start": start.isoformat(),
            "cutoff": cutoff.isoformat(),
            "window_days": window_days,
            "training_windows": training_windows,
            "dictionary_as_of_day_utc": snapshot.isoformat(),
            "query_dates": {
                "facts": {"start": start.isoformat(), "end": cutoff.isoformat(), "inclusive": True},
                "dictionary": {
                    "as_of_day_utc": snapshot.isoformat(),
                    "exclusive_end_utc": (snapshot + timedelta(days=1)).isoformat() + "T00:00:00Z",
                },
            },
            "dictionary_representation": (
                "latest historical timestamp per query, normquery, status_code, error"
            ),
            "fact_rows": len(facts),
            "dictionary_rows": len(dictionary),
            "source_completeness_audit": completeness,
            "source_metric_semantics": {
                "unique_search_users": (
                    "search users unique within each source row; sums are not period-unique users"
                ),
                "unique_view_users": (
                    "card visible users, not clicks; unique within each source row; "
                    "sums are not period-unique users"
                ),
                "order_count": "orders=count of orders, not unique users",
            },
            "files": {path.name: sha256(path) for path in folder.iterdir()},
            "fact_sql": FACT_SQL,
            "dictionary_sql": DICTIONARY_SQL,
        }
        write_json(folder / "source_manifest.json", manifest)
        if output.exists():
            raise ValueError(
                "Папка выгрузки появилась во время чтения; публикация отменена"
            )
        folder.replace(output)
    return manifest


def explicit_timestamp(value) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError, OverflowError):
        raise ValueError("Нужен timestamp с явным timezone") from None
    if pd.isna(stamp) or stamp.tzinfo is None or stamp.nanosecond:
        raise ValueError("Нужен timestamp с timezone и точностью не выше микросекунд")
    return stamp.tz_convert("UTC")


def strict_sku(value) -> str:
    sku = str(value)
    if not re.fullmatch(r"[1-9][0-9]{0,19}", sku) or int(sku) > 2**64 - 1:
        raise ValueError("SKU должен быть каноническим положительным UInt64")
    return sku


def read_sku_csv(path: Path, required: set[str]) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not required.issubset(fields):
            raise ValueError("CSV: отсутствуют обязательные или повторены имена колонок")
        rows = []
        seen = set()
        for row in reader:
            if None in row or any(row.get(field) is None for field in required):
                raise ValueError("CSV: некорректное число значений")
            row["sku"] = strict_sku(row["sku"])
            if row["sku"] in seen:
                raise ValueError(f"CSV: повторный SKU {row['sku']}")
            seen.add(row["sku"])
            rows.append(row)
    return rows


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError("non_json_constant")


def parse_attributes(raw) -> dict:
    empty = {"brand": "", "category": "", "description": ""}
    if not isinstance(raw, str):
        return {**empty, "attributes_status": "invalid_json", "attributes_error": "expected_json_string"}
    try:
        attributes = json.loads(raw, object_pairs_hook=_json_object, parse_constant=_invalid_constant)
    except (ValueError, TypeError, RecursionError):
        return {**empty, "attributes_status": "invalid_json", "attributes_error": "invalid_json"}
    names = {85: "brand", 8498: "category", 4191: "description"}
    values = {name: [] for name in empty}
    if not isinstance(attributes, list):
        return {**empty, "attributes_status": "invalid_schema", "attributes_error": "expected_ozon_attribute_list"}
    for attribute in attributes:
        if not isinstance(attribute, dict):
            return {**empty, "attributes_status": "invalid_schema", "attributes_error": "expected_attribute_object"}
        attribute_id = attribute.get("id")
        if (type(attribute_id) is not int and not (
            isinstance(attribute_id, str) and re.fullmatch(r"[0-9]{1,10}", attribute_id)
        )) or not isinstance(attribute.get("values"), list):
            return {**empty, "attributes_status": "invalid_schema", "attributes_error": "expected_id_and_values"}
        attribute_id = int(attribute_id)
        if not 0 < attribute_id <= 2**32 - 1:
            return {**empty, "attributes_status": "invalid_schema", "attributes_error": "expected_positive_uint32_id"}
        for item in attribute["values"]:
            if not isinstance(item, dict) or not isinstance(item.get("value"), str):
                return {**empty, "attributes_status": "invalid_schema", "attributes_error": "expected_string_value"}
            name = names.get(attribute_id)
            if name and item["value"] not in values[name]:
                values[name].append(item["value"])
    return {
        **{name: "\n".join(items) for name, items in values.items()},
        "attributes_status": "ok", "attributes_error": "",
    }


def build_metadata(records: pd.DataFrame, as_of: pd.Timestamp) -> list[dict]:
    required = {"sku", "name", "attributes", "loaded_at"}
    if not required.issubset(records.columns):
        raise ValueError("В источнике нет обязательных колонок")
    result = []
    seen = set()
    for record in records.to_dict("records"):
        sku = strict_sku(record["sku"])
        if sku in seen:
            raise ValueError(f"Повторный SKU источника {sku}")
        seen.add(sku)
        try:
            stamp = pd.Timestamp(record["loaded_at"])
            if stamp.tzinfo is None:
                stamp = stamp.tz_localize("UTC")
            stamp = explicit_timestamp(stamp)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"Некорректный loaded_at для SKU {sku}") from None
        if stamp > as_of:
            raise ValueError(f"loaded_at позже --as-of для SKU {sku}")
        raw = record["attributes"]
        row = {field: "" for field in METADATA_FIELDS}
        row.update({
            "sku": sku, "name": record["name"], "attributes_json": raw,
            "snapshot_at": stamp.isoformat(),
            **parse_attributes(raw),
        })
        result.append(row)
    return sorted(result, key=lambda row: row["sku"])


def publish_snapshot(folder: Path, output: Path):
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        arguments = (os.fsencode(folder), os.fsencode(output), 4)
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        arguments = (-100, os.fsencode(folder), -100, os.fsencode(output), 1)
    else:
        raise RuntimeError("Нет поддержки atomic rename без перезаписи")
    rename.restype = ctypes.c_int
    if rename(*arguments) != 0:
        error = ctypes.get_errno()
        if error in {errno.EEXIST, errno.ENOTEMPTY}:
            raise ValueError("Папка выгрузки уже существует; перезапись запрещена")
        raise OSError(error, "Не удалось атомарно опубликовать snapshot")


def export_snapshot(ch, as_of, output: Path, *, sku_file: Path | None = None) -> dict:
    as_of = explicit_timestamp(as_of)
    output = Path(output)
    if os.path.lexists(output):
        raise ValueError("Папка выгрузки уже существует; перезапись запрещена")
    requested = None if sku_file is None else {
        row["sku"] for row in read_sku_csv(sku_file, {"sku"})
    }
    try:
        records = ch.query_df(PRODUCT_SQL, parameters={"as_of": as_of.to_pydatetime()})
    except (ClickHouseError, Urllib3HTTPError, HTTPException, OSError, ValueError, EOFError):
        raise RuntimeError("Ошибка чтения источника; подробности соединения не выводятся") from None
    if not {"sku", "name", "attributes", "loaded_at"}.issubset(records.columns):
        raise ValueError("В источнике нет обязательных колонок")
    source_skus = records["sku"].map(strict_sku)
    if source_skus.duplicated().any():
        raise ValueError("Повторный SKU источника")
    records = records.copy()
    records["sku"] = source_skus
    if requested is not None:
        records = records.loc[records["sku"].isin(requested)]
    rows = build_metadata(records, as_of)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".product-media-") as temporary:
        folder = Path(temporary) / "snapshot"
        folder.mkdir()
        with (folder / "metadata.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=METADATA_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        files = [folder / "metadata.csv"]
        manifest = {
            "schema_version": 2, "source": "raw.ozon_product_info_attributes",
            "source_sql": PRODUCT_SQL, "as_of": as_of.isoformat(),
            "source_semantics": "current dated text slice; not a historical product archive",
            "snapshot_at_semantics": "actual text loaded_at in UTC",
            "source_rows": len(source_skus), "rows": len(rows),
            "requested_skus": None if requested is None else sorted(requested),
            "missing_requested_skus": [] if requested is None else sorted(requested - set(source_skus)),
            "text_snapshot_times": sorted({row["snapshot_at"] for row in rows}),
            "attribute_diagnostics": [
                {"sku": row["sku"], "status": row["attributes_status"], "reason": row["attributes_error"]}
                for row in rows if row["attributes_status"] != "ok"
            ],
            "files": {str(path.relative_to(folder)): sha256(path) for path in files},
        }
        write_json(folder / "manifest.json", manifest)
        publish_snapshot(folder, output)
    return manifest


def main(argv: list[str] | None = None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "products":
        parser = argparse.ArgumentParser(description="Текущий датированный текст Ozon; без обработки фото")
        parser.add_argument("--as-of", type=explicit_timestamp, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--sku-file", type=Path)
        args = parser.parse_args(argv[1:])
        try:
            ch = source_client()
        except Exception:  # noqa: BLE001 — граница CLI: неожиданные ошибки SDK могут содержать secrets.
            parser.exit(1, "Не удалось подключиться к источнику; проверьте .env без публикации secrets\n")
        try:
            try:
                result = export_snapshot(
                    ch, args.as_of, args.output, sku_file=args.sku_file,
                )
            except ValueError as exc:
                parser.exit(2, f"{exc}\n")
            finally:
                with suppress(ClickHouseError, Urllib3HTTPError, HTTPException, OSError):
                    ch.close()
        except Exception:  # noqa: BLE001 — граница CLI, включая cleanup: не раскрывать secrets в traceback.
            parser.exit(1, "Экспорт не опубликован; подробности ошибок/соединения не выводятся\n")
        print(f"Строк: {result['rows']}; ошибки attributes: {len(result['attribute_diagnostics'])}")
        return

    parser = argparse.ArgumentParser(
        description="Выгрузить факты Ozon и исторический словарь WB"
    )
    parser.add_argument("--cutoff", type=date.fromisoformat, required=True)
    parser.add_argument("--history-start", type=date.fromisoformat)
    parser.add_argument("--dictionary-as-of", type=date.fromisoformat)
    parser.add_argument("--window-days", type=int, default=28)
    parser.add_argument("--training-windows", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        export_dates(
            args.cutoff, args.window_days, args.training_windows,
            args.history_start, args.dictionary_as_of,
        )
        if args.output.exists():
            raise ValueError("Папка выгрузки уже существует; выберите новую")
    except ValueError as error:
        parser.error(str(error))
    ch = source_client()
    try:
        result = export_data(
            ch, args.cutoff, args.output, args.window_days, args.training_windows,
            history_start=args.history_start, dictionary_as_of=args.dictionary_as_of,
        )
        print(
            f"Выгружено {result['fact_rows']} фактов и {result['dictionary_rows']} строк словаря: {args.output}"
        )
    finally:
        ch.close()


if __name__ == "__main__":
    main()
