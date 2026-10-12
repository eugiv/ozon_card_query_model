"""Консервативная SEO-аннотация: точная фраза не доказывает совместимость.

К кандидатам добавляются строковые seo_presence, compatibility, decision, reason,
product_name, own_brand и snapshot_at (datetime64[ns, UTC], NaT без товара).
product_volume_ml и query_volume_ml — nullable Float64, только однозначный объём.
product/query_material, purpose, type — строки, неизвестное значение — "".
policy_version и constraint_vocabulary_version фиксируют правила для аудита.
Сырой JSON, URL и описания из products не переносятся в результат.
ML ранжирует кандидатов; здесь лишь явные свидетельства и защитный фильтр.
Словари неполны: неизвестное не доказывает ни конфликт, ни совместимость.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal
from numbers import Integral, Real

import numpy as np
import pandas as pd

_VOLUME_PATTERN = re.compile(
    r"(?<![\w.,])(?P<sign>[-+−]?)\s*"
    r"(?P<number>[0-9]+(?:[.,][0-9]+)?)\s*"
    r"(?P<unit>(?:миллилитр|литр)(?:а|ов|ы|е|у|ом|ами|ах)?|мл|ml|л|l)(?!\w)",
    re.IGNORECASE,
)
_VOLUME_AMBIGUITY = re.compile(
    r"\b(?:комплект|набор|концентрат)(?:а|ов|ы|е|у|ом|ами|ах)?\b", re.IGNORECASE
)

POLICY_VERSION = "seo-compatibility-v2"
CONSTRAINT_VOCABULARY_VERSION = "ru-explicit-v1"
_ADJECTIVE_ENDINGS = r"(?:ый|ая|ое|ые|ого|ой|ому|ым|ую|ых|ыми|ом)"
_MATERIAL_VOCABULARY = {
    "microfiber": rf"микрофибр(?:а|ы|е|у|ой|ою)|микрофибров{_ADJECTIVE_ENDINGS}",
    "cellulose": rf"целлюлоз(?:а|ы|е|у|ой|ою)|целлюлозн{_ADJECTIVE_ENDINGS}",
    "latex": rf"латекс(?:а|у|ом|е)?|латексн{_ADJECTIVE_ENDINGS}",
    "nitrile": rf"нитрил(?:а|у|ом|е)?|нитрилов{_ADJECTIVE_ENDINGS}",
    "vinyl": rf"винил(?:а|у|ом|е)?|винилов{_ADJECTIVE_ENDINGS}",
    "melamine": rf"меламин(?:а|у|ом|е)?|меламинов{_ADJECTIVE_ENDINGS}",
}
_TYPE_VOCABULARY = {
    "bags": r"пакет(?:ы|а|ов|у|ом|е|ам|ами|ах)?|меш(?:ок|ки|ка|ков|ку|ком|ке|кам|ками|ках)",
    "wipes": r"салфет(?:ка|ки|ку|ке|кой|ок|кам|ками|ках)",
    "sponges": r"губ(?:ка|ки|ку|ке|кой|ок|кам|ками|ках)",
    "gloves": r"перчат(?:ка|ки|ку|ке|кой|ок|кам|ками|ках)",
}
_SPONGE_WIPE = rf"губчат{_ADJECTIVE_ENDINGS}"
_PURPOSE_VOCABULARY = {
    "waste": ("для мусора", "для сбора мусора", rf"мусорн{_ADJECTIVE_ENDINGS}"),
    "freezing": ("для заморозки", "для замораживания"),
    "food_storage": (
        "для хранения пищевых продуктов", "для хранения продуктов",
        "для пищевых продуктов", "для хранения пищи",
    ),
}
_PURPOSE_CONFLICTS = {
    frozenset(("waste", "freezing")), frozenset(("waste", "food_storage")),
}
_TYPE_CONFLICTS = {frozenset(("microfiber_wipes", "sponge_wipes"))}
_MIXTURE_ONLY_VOCABULARY = re.compile(
    rf"полиэстер(?:а|у|ом|е)?|полиэстеров{_ADJECTIVE_ENDINGS}|"
    rf"полиамид(?:а|у|ом|е)?|полиамидн{_ADJECTIVE_ENDINGS}|"
    rf"поролон(?:а|у|ом|е)?|поролонов{_ADJECTIVE_ENDINGS}|"
    rf"хлоп(?:ок|ка|ку|ком|ке)|хлопков{_ADJECTIVE_ENDINGS}|"
    rf"вискоз(?:а|ы|е|у|ой)|вискозн{_ADJECTIVE_ENDINGS}"
)
_CONTEXT_AMBIGUITY = re.compile(
    r"комплект(?:а|ы|ов|е|ом)?|набор(?:а|ы|ов|е|ом)?|"
    r"сравнен(?:ие|ия|ию|ием|ии)|сравнить|вместо|против|аналог(?:а|и|ов)?|"
    r"альтернатив(?:а|ы|ой)|замен(?:а|ы|ой|яет|яют)|лучше|хуже|отличи(?:е|я)|"
    r"смесь|смеси|смешанн" + _ADJECTIVE_ENDINGS + r"|комбинированн" + _ADJECTIVE_ENDINGS
    + r"|слой|слои|слоя|слоев|покрыти(?:е|я|ем|и)|вставк(?:а|и|ой|е|у)|"
    r"подкладк(?:а|и|ой|е|у)|(?:двух|трех|много)слойн" + _ADJECTIVE_ENDINGS + r"|"
    r"подходит|подходят|совместим" + _ADJECTIVE_ENDINGS + r"|как|vs|versus|и|или"
)
_NEGATIVE_MODIFIERS = {
    "содержит", "содержания", "добавления", "использования", "примеси",
    "натурального", "натуральной", "натуральный", "натуральная",
    "из", "предназначен", "предназначены",
}
_NEGATIVE_WORDS = {"без", "не", "нет", "исключая"}
_EXPLICIT_FIELDS = ("material", "purpose", "type")


def review_top_k(annotated, k=10):
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k должен быть положительным целым")
    rows = annotated.loc[annotated["decision"].eq("review")].sort_values(
        ["sku", "score", "normquery"], ascending=[True, False, True], kind="stable"
    ).copy()
    rows["rank"] = rows.groupby("sku").cumcount() + 1
    return rows.loc[rows["rank"] <= k].reset_index(drop=True)


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    if not frame.columns.is_unique:
        raise ValueError("Имена колонок должны быть уникальными")
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Отсутствуют колонки: {', '.join(missing)}")


def _sku_key(value: object) -> str:
    if isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)):
        text = str(value)
    else:
        text = value
    if (
        not isinstance(text, str)
        or re.fullmatch(r"[1-9][0-9]{0,19}", text) is None
        or int(text) > 2**64 - 1
    ):
        raise ValueError("SKU должен быть каноническим положительным UInt64")
    return text


def _utc_timestamp(value: object, field: str) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field}: нужен timestamp с явным часовым поясом") from error
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError(f"{field}: нужен timestamp с явным часовым поясом")
    return stamp.tz_convert("UTC")


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[^\W_]+", text.casefold().replace("ё", "е")))


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _plain_segments(value: object) -> list[str]:
    text = _text(value)
    if not text:
        return []
    try:
        json.loads(text)
    except (ValueError, TypeError):
        pass
    else:
        return []
    return [
        part.strip()
        for part in re.split(
            r"(?:[a-z][a-z0-9+.-]*://|www\.|//)[^\s<>\"']+",
            text,
            flags=re.IGNORECASE,
        )
        if part.strip()
    ]


def _explicit_volume_ml(value: object) -> Decimal | None:
    segments = _plain_segments(value)
    if any(_VOLUME_AMBIGUITY.search(segment) for segment in segments):
        return None
    volumes = set()
    for segment in segments:
        for match in _VOLUME_PATTERN.finditer(segment):
            if segment[:match.start("number")].rstrip().endswith(("-", "−")):
                return None
            amount = Decimal(match["number"].replace(",", "."))
            unit = match["unit"].casefold()
            volume = amount * (1 if unit.startswith(("м", "m")) else 1000)
            if volume <= 0 or not np.isfinite(float(volume)):
                return None
            volumes.add(volume)
    return next(iter(volumes)) if len(volumes) == 1 else None


def _attribute_texts(raw: object, status: object) -> tuple[list[str], bool]:
    if not isinstance(status, str) or status != "ok" or not isinstance(raw, str):
        return [], False
    try:
        attributes = json.loads(raw)
    except (ValueError, TypeError):
        return [], False
    if not isinstance(attributes, list):
        return [], False
    texts = []
    for attribute in attributes:
        if not isinstance(attribute, dict):
            return [], False
        identifier = attribute.get("id")
        if isinstance(identifier, int) and not isinstance(identifier, bool):
            number = identifier
        elif isinstance(identifier, str) and re.fullmatch(r"[0-9]{1,10}", identifier):
            number = int(identifier)
        else:
            return [], False
        values = attribute.get("values")
        if not 0 < number <= 2**32 - 1 or not isinstance(values, list):
            return [], False
        for entry in values:
            if not isinstance(entry, dict) or not isinstance(entry.get("value"), str):
                return [], False
            if number != 11254:
                texts.extend(_plain_segments(entry["value"]))
    return texts, True


def _contains(tokens: tuple[str, ...], phrase: tuple[str, ...]) -> bool:
    length = len(phrase)
    return any(
        tokens[start : start + length] == phrase
        for start in range(len(tokens) - length + 1)
    )


def _negated(tokens: tuple[str, ...], start: int, end: int) -> bool:
    previous = start - 1
    while previous >= 0 and tokens[previous] in _NEGATIVE_MODIFIERS:
        previous -= 1
    return (
        previous >= 0 and tokens[previous] in _NEGATIVE_WORDS
    ) or (
        end < len(tokens) and tokens[end] in {"free", "отсутствует", "отсутствуют", "нет"}
    ) or tokens[end:end + 2] == ("не", "содержит")


def _positive_matches(
    tokens: tuple[str, ...], pattern: str, stop: int | None = None
) -> list[int]:
    """Целые токены без отрицания; граница поиска не обрезает контекст отрицания."""
    words = pattern.split()
    length = len(words)
    stop = len(tokens) if stop is None else stop
    return [
        start
        for start in range(stop - length + 1)
        if all(re.fullmatch(word, token)
               for word, token in zip(words, tokens[start:start + length]))
        and not _negated(tokens, start, start + length)
    ]


def _explicit_attributes(value: object) -> dict[str, str]:
    unknown = dict.fromkeys(_EXPLICIT_FIELDS, "")
    segments = _plain_segments(value)
    if len(segments) != 1:
        return unknown
    text = segments[0]
    tokens = _tokens(text)
    if (
        any(_CONTEXT_AMBIGUITY.fullmatch(token) for token in tokens)
        or any(symbol in text for symbol in ("+", "/", "&"))
        or any(Decimal(number.replace(",", ".")) != 100
               for number in re.findall(r"([0-9]+(?:[.,][0-9]+)?)\s*%", text))
    ):
        return unknown
    own_end = next(
        (index for index, token in enumerate(tokens) if token in {"для", "от"}), len(tokens)
    )
    types = {
        kind
        for kind, pattern in _TYPE_VOCABULARY.items()
        if _positive_matches(tokens, pattern, own_end)
    }
    if len(types) != 1:
        return unknown
    kind = next(iter(types))
    if kind == "wipes" and any(
        any(
            0 <= adjacent < own_end
            and re.fullmatch(_TYPE_VOCABULARY["wipes"], tokens[adjacent])
            for adjacent in (index - 1, index + 1)
        )
        for index in _positive_matches(tokens, _SPONGE_WIPE, own_end)
    ):
        kind = "sponge_wipes"
    materials = {
        material
        for material, pattern in _MATERIAL_VOCABULARY.items()
        if any(
            not (index > 0 and tokens[index - 1] == "с")
            for index in _positive_matches(tokens, pattern, own_end)
        )
    }
    material = next(iter(materials)) if len(materials) == 1 else ""
    if (kind == "sponge_wipes" and material == "microfiber") or (
        materials and _positive_matches(tokens, _MIXTURE_ONLY_VOCABULARY.pattern, own_end)
    ):
        return unknown
    purposes = {
        purpose
        for purpose, patterns in _PURPOSE_VOCABULARY.items()
        if kind == "bags" and any(_positive_matches(tokens, pattern) for pattern in patterns)
    }
    return {
        "material": material,
        "purpose": next(iter(purposes)) if len(purposes) == 1 else "",
        "type": kind,
    }


def _explicit_conflict(product: dict[str, str], query: dict[str, str]) -> tuple[str, str]:
    product_type, query_type = product["type"], query["type"]
    family = {"sponge_wipes": "wipes"}
    if (
        product_type and query_type
        and family.get(product_type, product_type) == family.get(query_type, query_type)
        and product["material"] and query["material"]
        and product["material"] != query["material"]
    ):
        return "material_conflict", "explicit_material_mismatch"
    product_subtype = (
        "microfiber_wipes"
        if product_type == "wipes" and product["material"] == "microfiber" else product_type
    )
    query_subtype = (
        "microfiber_wipes"
        if query_type == "wipes" and query["material"] == "microfiber" else query_type
    )
    if frozenset((product_subtype, query_subtype)) in _TYPE_CONFLICTS:
        return "material_conflict", "explicit_product_type_mismatch"
    if frozenset((product["purpose"], query["purpose"])) in _PURPOSE_CONFLICTS:
        return "purpose_conflict", "explicit_purpose_mismatch"
    return "", ""


def _brand_conflict(
    phrase: tuple[str, ...],
    own: tuple[str, ...],
    vocabulary: dict[str, list[tuple[str, ...]]],
) -> bool:
    if not own:
        return False
    start = 0
    while start < len(phrase):
        for brand in vocabulary.get(phrase[start], []):
            if phrase[start : start + len(brand)] == brand:
                if brand != own:
                    return True
                start += len(brand)
                break
        else:
            start += 1
    return False


def annotate_recommendations(
    candidates: pd.DataFrame, products: pd.DataFrame, as_of: pd.Timestamp
) -> pd.DataFrame:
    """Сохраняет строки, индекс и дополнительные колонки candidates без мутации.

    SKU — положительный UInt64 (целое либо каноническая строка), normquery —
    непустая строка с токенами, score — конечное число. Обе таблицы уникальны
    по своим ключам. as_of и все snapshot_at требуют явного timezone и
    приводятся к UTC; даже неиспользуемая будущая строка products запрещена.

    seo_presence: present | not_found_in_available_fields | unknown;
    compatibility: not_verified | brand_conflict | volume_conflict |
    material_conflict | purpose_conflict;
    decision: already_present | review | exclude_candidate. add_candidate
    не создаётся: ни совпадение, ни отсутствие фразы не подтверждает совместимость.
    Неполные поля мешают заключению об отсутствии, но не отменяют найденную фразу.
    Объём берётся только из name и normquery: одно различное положительное значение
    с явной единицей, без комплекта/набора/концентрата. Несовпадение свыше 1e-6 мл
    исключает кандидата даже при present; это фильтр очереди, не удаление из SEO.
    Свидетельства product_volume_ml/query_volume_ml — Float64, пропуски — pd.NA
    (None при to_dict("records")), без вывода о совместимости при равенстве объёмов.
    Материал/тип/назначение — только закрытый словарь name и normquery, не описание
    и не неизвестные ID характеристик. Неоднозначные составы/комплекты/сравнения
    не блокируются. Приоритет: объём > бренд > материал > подтип > назначение.
    Имена аннотаций зарезервированы и заменяются при повторном вызове.
    """
    cutoff = _utc_timestamp(as_of, "as_of")
    _require_columns(candidates, ["sku", "normquery", "score"])
    _require_columns(
        products,
        [
            "sku", "name", "description", "brand", "attributes_json",
            "snapshot_at", "attributes_status",
        ],
    )
    candidate_keys = candidates["sku"].map(_sku_key)
    product_keys = products["sku"].map(_sku_key)
    if product_keys.duplicated().any():
        raise ValueError("Дубли SKU в products")
    phrases = []
    for query in candidates["normquery"]:
        if not isinstance(query, str) or not _tokens(query):
            raise ValueError("normquery должен быть непустой фразой")
        phrases.append(_tokens(query))
    keys = pd.DataFrame(
        {"sku": candidate_keys.to_numpy(), "normquery": candidates["normquery"].to_numpy()}
    )
    if keys.duplicated().any():
        raise ValueError("Дубли SKU + normquery в candidates")
    for score in candidates["score"]:
        if (
            isinstance(score, (bool, np.bool_))
            or not isinstance(score, Real)
            or not np.isfinite(score)
        ):
            raise ValueError("score должен быть конечным числом")

    stamps = [_utc_timestamp(value, "snapshot_at") for value in products["snapshot_at"]]
    if any(stamp > cutoff for stamp in stamps):
        raise ValueError("snapshot_at > as_of: будущие метаданные запрещены")

    vocabulary: dict[str, list[tuple[str, ...]]] = {}
    for brand in sorted(
        {_tokens(_text(value)) for value in products["brand"]},
        key=lambda value: (-len(value), value),
    ):
        if brand:
            vocabulary.setdefault(brand[0], []).append(brand)

    metadata = {}
    for sku, stamp, product in zip(product_keys, stamps, products.to_dict("records")):
        attributes, valid = _attribute_texts(
            product["attributes_json"], product["attributes_status"]
        )
        names = _plain_segments(product["name"])
        descriptions = _plain_segments(product["description"])
        metadata[sku] = {
            "snapshot_at": stamp,
            "product_name": " ".join(names),
            "own_brand": " ".join(_plain_segments(product["brand"])),
            "brand_tokens": _tokens(_text(product["brand"])),
            "volume_ml": _explicit_volume_ml(product["name"]),
            "explicit": _explicit_attributes(product["name"]),
            "fields": [_tokens(text) for text in [*names, *descriptions, *attributes]],
            "attributes_valid": valid,
            "text_complete": bool(names and descriptions),
        }

    annotations = []
    for sku, phrase, query in zip(candidate_keys, phrases, candidates["normquery"]):
        product = metadata.get(sku)
        query_volume = _explicit_volume_ml(query)
        query_explicit = _explicit_attributes(query)
        record = {
            "seo_presence": "unknown",
            "compatibility": "not_verified",
            "decision": "review",
            "reason": "product_metadata_missing",
            "snapshot_at": pd.NaT,
            "product_name": "",
            "own_brand": "",
            "product_volume_ml": None,
            "query_volume_ml": float(query_volume) if query_volume is not None else None,
            "policy_version": POLICY_VERSION,
            "constraint_vocabulary_version": CONSTRAINT_VOCABULARY_VERSION,
        }
        for field in _EXPLICIT_FIELDS:
            record[f"product_{field}"] = ""
            record[f"query_{field}"] = query_explicit[field]
        if product is not None:
            product_volume = product["volume_ml"]
            if product_volume is not None:
                record["product_volume_ml"] = float(product_volume)
            for field in ["snapshot_at", "product_name", "own_brand"]:
                record[field] = product[field]
            for field in _EXPLICIT_FIELDS:
                record[f"product_{field}"] = product["explicit"][field]
            if any(_contains(field, phrase) for field in product["fields"]):
                record.update(
                    seo_presence="present", decision="already_present", reason="seo_phrase_found"
                )
            elif not product["attributes_valid"]:
                record["reason"] = "attributes_invalid"
            elif not product["text_complete"]:
                record["reason"] = "seo_fields_incomplete"
            else:
                record.update(
                    seo_presence="not_found_in_available_fields",
                    reason="absence_in_all_seo_fields_not_verified",
                )
            if (
                product_volume is not None
                and query_volume is not None
                and abs(product_volume - query_volume) > Decimal("0.000001")
            ):
                conflict, reason = "volume_conflict", "explicit_volume_mismatch"
            elif _brand_conflict(phrase, product["brand_tokens"], vocabulary):
                conflict, reason = "brand_conflict", "known_brand_conflict"
            else:
                conflict, reason = _explicit_conflict(product["explicit"], query_explicit)
            if conflict:
                record.update(
                    compatibility=conflict, decision="exclude_candidate", reason=reason
                )
        annotations.append(record)

    output = candidates.copy(deep=True)
    for field in [
        "seo_presence", "compatibility", "decision", "reason", "product_name", "own_brand",
        "policy_version", "constraint_vocabulary_version",
        *[f"{side}_{field}" for side in ("product", "query") for field in _EXPLICIT_FIELDS],
    ]:
        output[field] = pd.array([record[field] for record in annotations], dtype="string")
    output["snapshot_at"] = pd.array(
        [record["snapshot_at"] for record in annotations], dtype="datetime64[ns, UTC]"
    )
    for field in ["product_volume_ml", "query_volume_ml"]:
        output[field] = pd.array([record[field] for record in annotations], dtype="Float64")
    return output