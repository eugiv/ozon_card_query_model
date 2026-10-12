import hashlib
import io
import json

import numpy as np
import pandas as pd
import pytest

from seo_recommendation_policy import (
    CONSTRAINT_VOCABULARY_VERSION,
    POLICY_VERSION,
    annotate_recommendations,
)

AS_OF = pd.Timestamp("2026-10-06T20:16:00Z")


def products(**changes):
    row = {
        "sku": "1",
        "name": "Белое мыло",
        "description": "Для ухода за кожей",
        "brand": "Невская косметика",
        "attributes_json": "[]",
        "snapshot_at": "2026-10-06T00:45:00Z",
        "attributes_status": "ok",
    }
    row.update(changes)
    return pd.DataFrame([row])


def candidates(*queries, sku="1"):
    return pd.DataFrame(
        {"sku": [sku] * len(queries), "normquery": queries, "score": [0.5] * len(queries)}
    )


def attributes(*entries):
    return json.dumps(
        [{"id": identifier, "values": [{"value": value} for value in values]}
         for identifier, values in entries],
        ensure_ascii=False,
    )


def test_whole_contiguous_tokens_not_substrings_or_reordered_words():
    result = annotate_recommendations(
        candidates("мыло", "мыльница", "белое мыло", "мыло белое", "для кожи"),
        products(description="Мыльница для ухода за кожей"), AS_OF,
    ).set_index("normquery")
    assert result.loc["мыло", "decision"] == "already_present"
    assert result.loc["мыльница", "decision"] == "already_present"
    assert result.loc["белое мыло", "decision"] == "already_present"
    for query in ["мыло белое", "для кожи"]:
        assert result.loc[query, "seo_presence"] == "not_found_in_available_fields"
    absent = annotate_recommendations(
        candidates("мыло"), products(name="Мыльница", description="Soapdish"), AS_OF
    )
    assert absent.iloc[0]["decision"] == "review"
    english = annotate_recommendations(
        candidates("soap"), products(name="Soapdish"), AS_OF
    )
    assert english.iloc[0]["seo_presence"] == "not_found_in_available_fields"


def test_casefold_yo_and_punctuation_separators():
    result = annotate_recommendations(
        candidates("ёлочное мыло", "ЕЛОЧНОЕ-МЫЛО", "STRASSE", "мыло белое"),
        products(name="Елочное—мыло, белое", description="Straße"), AS_OF,
    )
    assert result["seo_presence"].eq("present").all()
    assert result["compatibility"].eq("not_verified").all()


def test_attributes_only_plain_values_not_ids_keys_rich_json_or_urls():
    raw = attributes(
        (100, ["Цвет: нежно_белый"]),
        ("200", ["натуральное мыло"]),
        (11254, ["богатое мыло", '{"text": "rich мыло"}']),
        (300, ['{"text": "json мыло"}', '["list мыло"]']),
        (400, ["https://example.test/url-мыло", "www.example.test/ссылка-мыло"]),
    )
    result = annotate_recommendations(
        candidates("нежно белый", "натуральное мыло", "богатое мыло", "rich мыло",
                   "json мыло", "list мыло", "url мыло", "ссылка мыло", "values", "300"),
        products(attributes_json=raw), AS_OF,
    )
    assert result.iloc[:2]["decision"].eq("already_present").all()
    assert result.iloc[2:]["seo_presence"].eq("not_found_in_available_fields").all()
    assert "attributes_json" not in result
    assert "description" not in result
    assert "https://" not in result.to_csv(index=False)


def test_fields_values_and_url_gaps_are_never_joined_into_a_phrase():
    result = annotate_recommendations(
        candidates("белое мыло", "мыло мягкое", "мыло душистое"),
        products(
            name="Белое", description="мыло https://example.test мягкое",
            attributes_json=attributes((100, ["мыло", "душистое"])),
        ), AS_OF,
    )
    assert result["seo_presence"].eq("not_found_in_available_fields").all()


def test_absence_does_not_authorize_automatic_addition():
    result = annotate_recommendations(candidates("новый запрос"), products(), AS_OF)
    assert result.iloc[0]["seo_presence"] == "not_found_in_available_fields"
    assert result.iloc[0]["reason"] == "absence_in_all_seo_fields_not_verified"
    assert result.iloc[0]["decision"] == "review"
    assert result.iloc[0]["compatibility"] == "not_verified"


@pytest.mark.parametrize("field,value", [
    ("name", ""), ("name", None), ("description", " "), ("description", np.nan),
])
def test_missing_text_is_unknown_not_absence(field, value):
    result = annotate_recommendations(
        candidates("новый запрос"), products(**{field: value}), AS_OF
    )
    assert result.iloc[0]["seo_presence"] == "unknown"
    assert result.iloc[0]["decision"] == "review"
    assert result.iloc[0]["reason"] == "seo_fields_incomplete"


@pytest.mark.parametrize("raw,status", [
    (None, "ok"), ("{", "invalid_json"), ("{", "ok"), ("{}", "ok"),
    ('[{"id":1,"values":[{"value":42}]}]', "ok"),
    ('[{"id":true,"values":[]}]', "ok"),
    ('[{"id":0,"values":[]}]', "ok"),
    ('[{"id":4294967296,"values":[]}]', "ok"),
    ('[{"id":1,"values":{}}]', "ok"),
    ('[{"id":1,"values":[{}]}]', "ok"),
    ('[null]', "ok"), ('[{"values":[]}]', "ok"),
    ("[]", "invalid_schema"), ("[]", None),
])
def test_invalid_or_unavailable_attributes_cannot_establish_absence(raw, status):
    result = annotate_recommendations(
        candidates("новый запрос"),
        products(attributes_json=raw, attributes_status=status), AS_OF,
    )
    assert result.iloc[0]["seo_presence"] == "unknown"
    assert result.iloc[0]["decision"] == "review"
    assert result.iloc[0]["reason"] == "attributes_invalid"


def test_direct_presence_remains_evidence_with_other_fields_missing():
    result = annotate_recommendations(
        candidates("белое мыло"),
        products(description=None, attributes_json="invalid", attributes_status="invalid_json"),
        AS_OF,
    )
    assert result.iloc[0]["decision"] == "already_present"
    assert result.iloc[0]["seo_presence"] == "present"


def test_missing_product_metadata_is_unknown_and_has_no_source_timestamp():
    result = annotate_recommendations(candidates("мыло", sku="2"), products(), AS_OF)
    assert result.iloc[0]["reason"] == "product_metadata_missing"
    assert result.iloc[0]["seo_presence"] == "unknown"
    assert result.iloc[0]["compatibility"] == "not_verified"
    assert pd.isna(result.iloc[0]["snapshot_at"])
    assert result.iloc[0]["own_brand"] == ""


def test_explicit_foreign_brand_overrides_presence_but_aliases_are_not_inferred():
    catalog = pd.concat([products(), products(sku="2", brand="Dove")], ignore_index=True)
    catalog.loc[0, "description"] = "Мыло Dove для сравнения"
    result = annotate_recommendations(
        candidates("мыло DOVE", "мыло дав", "мыло dovetail", "мыло невская косметика",
                   "невская косметика и dove"), catalog, AS_OF,
    )
    assert result.iloc[0]["seo_presence"] == "present"
    for index in [0, 4]:
        assert result.iloc[index]["compatibility"] == "brand_conflict"
        assert result.iloc[index]["decision"] == "exclude_candidate"
        assert result.iloc[index]["reason"] == "known_brand_conflict"
    assert result.iloc[1:4]["compatibility"].eq("not_verified").all()


@pytest.mark.parametrize("brand", ["", None, np.nan])
def test_no_own_brand_does_not_block_known_brand(brand):
    catalog = pd.concat(
        [products(brand=brand), products(sku="2", brand="Dove")], ignore_index=True
    )
    result = annotate_recommendations(candidates("мыло dove"), catalog, AS_OF)
    assert result.iloc[0]["compatibility"] == "not_verified"
    assert result.iloc[0]["decision"] == "review"


def test_longest_known_brand_prevents_prefix_conflicts():
    catalog = pd.concat(
        [products(), products(sku="2", brand="Невская")], ignore_index=True
    )
    result = annotate_recommendations(
        candidates("невская косметика мыло", "невская мыло",
                   "невская косметика невская мыло"), catalog, AS_OF,
    )
    assert result.iloc[0]["compatibility"] == "not_verified"
    assert result.iloc[1:]["compatibility"].eq("brand_conflict").all()
    shorter_own = annotate_recommendations(
        candidates("невская косметика мыло", sku="2"), catalog, AS_OF
    )
    assert shorter_own.iloc[0]["compatibility"] == "brand_conflict"


def test_metadata_at_cutoff_is_allowed_and_timestamp_is_utc():
    stamp = "2026-10-06T23:16:00+03:00"
    result = annotate_recommendations(
        candidates("мыло"), products(snapshot_at=stamp), pd.Timestamp(stamp)
    )
    assert result.iloc[0]["snapshot_at"] == AS_OF
    assert str(result["snapshot_at"].dtype) == "datetime64[ns, UTC]"
    assert result.iloc[0]["product_name"] == "Белое мыло"
    assert result.iloc[0]["own_brand"] == "Невская косметика"


def test_october_snapshot_is_forbidden_for_september_evaluation():
    with pytest.raises(ValueError, match="будущие метаданные"):
        annotate_recommendations(
            candidates("мыло"), products(), pd.Timestamp("2026-09-30T23:59:59Z")
        )


def test_even_unmatched_future_metadata_is_rejected():
    catalog = pd.concat(
        [products(), products(sku="2", snapshot_at=AS_OF + pd.Timedelta(microseconds=1))],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="snapshot_at > as_of"):
        annotate_recommendations(candidates("мыло"), catalog, AS_OF)


@pytest.mark.parametrize("stamp", [None, pd.NaT, "2026-10-06", "invalid"])
def test_naive_missing_and_invalid_timestamps_are_rejected(stamp):
    with pytest.raises(ValueError, match="as_of"):
        annotate_recommendations(candidates("мыло"), products(), stamp)
    with pytest.raises(ValueError, match="snapshot_at"):
        annotate_recommendations(candidates("мыло"), products(snapshot_at=stamp), AS_OF)


@pytest.mark.parametrize("sku", ["", " 1", "01", "0", "-1", "1.0", "1e3",
                                 1.0, True, None, "18446744073709551616"])
def test_invalid_identifiers_are_rejected_in_either_input(sku):
    with pytest.raises(ValueError, match="SKU"):
        annotate_recommendations(candidates("мыло", sku=sku), products(), AS_OF)
    with pytest.raises(ValueError, match="SKU"):
        annotate_recommendations(candidates("мыло"), products(sku=sku), AS_OF)


@pytest.mark.parametrize("query", ["", " ", "!!!", None, 123])
def test_invalid_query_identifiers_are_rejected(query):
    with pytest.raises(ValueError, match="normquery"):
        annotate_recommendations(candidates(query), products(), AS_OF)


@pytest.mark.parametrize("score", [np.nan, np.inf, -np.inf, None, True, "bad"])
def test_invalid_scores_are_rejected(score):
    rows = candidates("мыло")
    rows["score"] = score
    with pytest.raises(ValueError, match="score"):
        annotate_recommendations(rows, products(), AS_OF)


def test_duplicate_keys_and_required_columns_are_rejected():
    with pytest.raises(ValueError, match=r"Дубли SKU \+ normquery"):
        annotate_recommendations(candidates("мыло", "мыло"), products(), AS_OF)
    catalog = pd.concat([products(), products(sku=1)], ignore_index=True)
    with pytest.raises(ValueError, match="Дубли SKU в products"):
        annotate_recommendations(candidates("мыло"), catalog, AS_OF)
    rows = pd.concat([candidates("мыло"), candidates("мыло", sku=1)], ignore_index=True)
    with pytest.raises(ValueError, match=r"Дубли SKU \+ normquery"):
        annotate_recommendations(rows, products(), AS_OF)
    with pytest.raises(ValueError, match="Отсутствуют колонки: score"):
        annotate_recommendations(candidates("мыло").drop(columns="score"), products(), AS_OF)
    with pytest.raises(ValueError, match="Отсутствуют колонки: brand"):
        annotate_recommendations(candidates("мыло"), products().drop(columns="brand"), AS_OF)


def test_inputs_index_optional_columns_and_distinct_queries_are_preserved():
    rows = candidates("Ёлочное мыло", "елочное-мыло", "мыло")
    rows.index = [8, 8, 3]
    rows["retrieval_sources"] = ["graph", "text", "embedding"]
    catalog = products(name="Ёлочное мыло")
    before_rows, before_products = rows.copy(deep=True), catalog.copy(deep=True)
    result = annotate_recommendations(rows, catalog, AS_OF)
    pd.testing.assert_frame_equal(rows, before_rows)
    pd.testing.assert_frame_equal(catalog, before_products)
    pd.testing.assert_frame_equal(result[rows.columns], rows)
    assert len(result) == 3
    assert result["decision"].eq("already_present").all()
    assert result["decision"].dtype == pd.StringDtype()


def test_empty_inputs_keep_a_typed_schema_and_missing_catalog_is_allowed():
    rows = candidates()
    result = annotate_recommendations(rows, products().iloc[:0], AS_OF)
    assert result.empty
    assert str(result["snapshot_at"].dtype) == "datetime64[ns, UTC]"
    assert result["seo_presence"].dtype == pd.StringDtype()
    missing = annotate_recommendations(candidates("мыло"), products().iloc[:0], AS_OF)
    assert missing.iloc[0]["reason"] == "product_metadata_missing"


def test_explicit_volume_conflict_excludes_candidate_even_when_phrase_is_present():
    query = "мешки для мусора paclan 35 л"
    result = annotate_recommendations(
        candidates(query),
        products(
            name="Паклан мешки для мусора CLASSIC 120л x 10шт (20 мкм)",
            brand="Paclan", description=query,
        ), AS_OF,
    )
    row = result.iloc[0]
    assert row["compatibility"] == "volume_conflict"
    assert row["decision"] == "exclude_candidate"
    assert row["reason"] == "explicit_volume_mismatch"
    assert row["seo_presence"] == "present"
    assert row["product_volume_ml"] == 120000.0
    assert row["query_volume_ml"] == 35000.0


@pytest.mark.parametrize("title,query,volume", [
    ("Лосьон 150мл", "лосьон 0.15л", 150.0),
    ("Лосьон 0,15 Л", "лосьон 150 МЛ", 150.0),
    ("Лосьон 150,5 миллилитра", "лосьон 0.1505 литра", 150.5),
    ("Лосьон 150.5 ML", "лосьон 0,1505 L", 150.5),
    ("Лосьон 150 миллилитров", "лосьон 150 миллилитр", 150.0),
    ("Мешки 35 литров", "мешки 35 литр", 35000.0),
    ("Лосьон 150мл / 0.15л", "лосьон 0,15л / 150 ml", 150.0),
    ("Лосьон 150мл+0.15л", "лосьон 150 мл", 150.0),
    ("Лосьон 150 мл x 10шт", "лосьон 150 мл 2шт", 150.0),
])
def test_volume_units_decimals_and_repeated_equal_values_match(title, query, volume):
    result = annotate_recommendations(candidates(query), products(name=title), AS_OF)
    row = result.iloc[0]
    assert row["compatibility"] == "not_verified"
    assert row["decision"] != "exclude_candidate"
    assert row["product_volume_ml"] == volume
    assert row["query_volume_ml"] == volume
    assert row["reason"] != "explicit_volume_mismatch"


@pytest.mark.parametrize("query_volume,conflict", [
    ("150.0000009", False), ("150.000001", False), ("150.0000011", True),
])
def test_volume_tolerance_is_absolute_in_millilitres(query_volume, conflict):
    row = annotate_recommendations(
        candidates(f"лосьон {query_volume} мл"), products(name="Лосьон 150 мл"), AS_OF
    ).iloc[0]
    assert (row["compatibility"] == "volume_conflict") == conflict


@pytest.mark.parametrize("text", [
    "Паклан CLASSIC", "Модель 150CLASSIC", "Модель 150LUX", "Модель L150",
    "Мешки 20 мкм", "Мешки 20 мм", "Средство 150 кг", "Средство 150 г",
    "Модель 150mlx", "Модель 150литраж", "Модель X150л", "Средство 150",
])
def test_volume_requires_a_complete_unit_and_number_boundaries(text):
    for title, query, field in [
        (text, "средство 35 мл", "product_volume_ml"),
        ("Средство 35 мл", text, "query_volume_ml"),
    ]:
        row = annotate_recommendations(candidates(query), products(name=title), AS_OF).iloc[0]
        assert row["compatibility"] == "not_verified"
        assert row["reason"] != "explicit_volume_mismatch"
        assert pd.isna(row[field])


@pytest.mark.parametrize("text", [
    "Средство 0 мл", "Средство -150 мл", "Средство −150 мл",
    "Средство -0,15 л", "Средство - 150 ml", "Средство 0.0 l",
    "Средство-150 мл", "Средство 150мл-50мл",
])
def test_nonpositive_volumes_do_not_trigger_conflicts(text):
    for title, query, field in [
        (text, "средство 35 мл", "product_volume_ml"),
        ("Средство 35 мл", text, "query_volume_ml"),
    ]:
        row = annotate_recommendations(candidates(query), products(name=title), AS_OF).iloc[0]
        assert row["compatibility"] == "not_verified"
        assert pd.isna(row[field])


@pytest.mark.parametrize("text", [
    "Средство 150 мл + 50 мл", "Средство 150 мл / 1 л",
    "Средство 150мл+50мл", "Средство 150мл+50мл+20мл",
    "Комплект средств 150 мл", "Набор средств 150 мл", "Концентрат 150 мл",
])
def test_multiple_volumes_sets_and_concentrates_are_ambiguous(text):
    for title, query, field in [
        (text, "средство 35 мл", "product_volume_ml"),
        ("Средство 35 мл", text, "query_volume_ml"),
    ]:
        row = annotate_recommendations(candidates(query), products(name=title), AS_OF).iloc[0]
        assert row["compatibility"] == "not_verified"
        assert row["decision"] != "exclude_candidate"
        assert pd.isna(row[field])


def test_volume_does_not_use_description_attributes_package_count_or_inference():
    row = annotate_recommendations(
        candidates("средство 35 мл"),
        products(
            name="Средство 150 кг 10шт", description="Объём 120 л",
            attributes_json=attributes((100, ["120л"])),
        ), AS_OF,
    ).iloc[0]
    assert row["compatibility"] == "not_verified"
    assert pd.isna(row["product_volume_ml"])
    assert row["query_volume_ml"] == 35.0
    assert row["reason"] == "absence_in_all_seo_fields_not_verified"


@pytest.mark.parametrize("title", [
    "Средство 150 https://example.test мл",
    "Средство https://example.test/150мл",
    '{"name": "Средство 150 мл"}',
])
def test_volume_does_not_read_urls_json_or_join_url_gaps(title):
    row = annotate_recommendations(
        candidates("средство 35 мл"), products(name=title), AS_OF
    ).iloc[0]
    assert row["compatibility"] == "not_verified"
    assert pd.isna(row["product_volume_ml"])


def test_volume_evidence_is_nullable_json_safe_and_inputs_are_not_mutated():
    rows = candidates("лосьон 35 мл", "лосьон", "лосьон 150 мл")
    rows.index = [4, 4, 9]
    rows["retrieval_sources"] = ["text", "graph", "text"]
    catalog = products(name="Лосьон 150 мл")
    before_rows, before_products = rows.copy(deep=True), catalog.copy(deep=True)
    result = annotate_recommendations(rows, catalog, AS_OF)
    pd.testing.assert_frame_equal(rows, before_rows)
    pd.testing.assert_frame_equal(catalog, before_products)
    pd.testing.assert_frame_equal(result[rows.columns], rows)
    fields = ["product_volume_ml", "query_volume_ml"]
    for field in fields:
        assert result[field].dtype == pd.Float64Dtype()
    evidence = result[fields].to_dict("records")
    assert evidence[1]["query_volume_ml"] is None
    json.dumps(evidence, allow_nan=False)
    empty = annotate_recommendations(candidates(), catalog, AS_OF)
    for field in fields:
        assert empty[field].dtype == pd.Float64Dtype()
    missing = annotate_recommendations(candidates("лосьон 35 мл", sku="2"), catalog, AS_OF)
    assert pd.isna(missing.iloc[0]["product_volume_ml"])
    assert missing.iloc[0]["query_volume_ml"] == 35.0
    repeated = annotate_recommendations(result, catalog, AS_OF)
    pd.testing.assert_frame_equal(repeated, result)


@pytest.mark.parametrize("title,query,conflict,reason", [
    ("Пакеты для замораживания", "пакеты для мусора", "purpose_conflict",
     "explicit_purpose_mismatch"),
    ("Салфетки из микрофибры", "губчатые салфетки", "material_conflict",
     "explicit_product_type_mismatch"),
    ("Перчатки нитриловые", "перчатки латексные", "material_conflict",
     "explicit_material_mismatch"),
])
def test_explicit_product_compatibility_regressions(title, query, conflict, reason):
    row = annotate_recommendations(
        candidates(query), products(name=title, description=query), AS_OF
    ).iloc[0]
    assert row["seo_presence"] == "present"
    assert row["compatibility"] == conflict
    assert row["decision"] == "exclude_candidate"
    assert row["reason"] == reason


@pytest.mark.parametrize("title,query,product_material,query_material", [
    ("Салфетки из микрофибры", "салфетки целлюлозные", "microfiber", "cellulose"),
    ("Салфетка целлюлозная", "микрофибровая салфетка", "cellulose", "microfiber"),
    ("Перчатки из латекса", "нитриловые перчатки", "latex", "nitrile"),
    ("Нитриловые перчатки", "перчатки из винила", "nitrile", "vinyl"),
    ("Виниловые перчатки", "перчатки латексные", "vinyl", "latex"),
    ("Губки меламиновые", "губки из целлюлозы", "melamine", "cellulose"),
    ("Перчатками из нитрила", "перчатками из латекса", "nitrile", "latex"),
    ("САЛФЕТКИ—МИКРОФИБРОВЫЕ", "салфетки, целлюлозные", "microfiber", "cellulose"),
    ("Салфетки 100% микрофибра", "целлюлозные салфетки", "microfiber", "cellulose"),
])
def test_known_explicit_materials_conflict_with_string_evidence(
    title, query, product_material, query_material
):
    row = annotate_recommendations(candidates(query), products(name=title), AS_OF).iloc[0]
    assert row["compatibility"] == "material_conflict"
    assert row["reason"] == "explicit_material_mismatch"
    assert row["decision"] == "exclude_candidate"
    assert row["product_material"] == product_material
    assert row["query_material"] == query_material
    assert row["product_purpose"] == row["query_purpose"] == ""


@pytest.mark.parametrize("word,material", [
    ("микрофибровые", "microfiber"), ("из микрофибры", "microfiber"),
    ("целлюлозные", "cellulose"), ("из целлюлозы", "cellulose"),
    ("латексные", "latex"), ("из латекса", "latex"),
    ("нитриловые", "nitrile"), ("из нитрила", "nitrile"),
    ("виниловые", "vinyl"), ("из винила", "vinyl"),
    ("меламиновые", "melamine"), ("из меламина", "melamine"),
])
def test_equal_explicit_materials_are_not_verified(word, material):
    row = annotate_recommendations(
        candidates(f"салфетки {word}"), products(name=f"Салфетки {word}"), AS_OF
    ).iloc[0]
    assert row["product_material"] == row["query_material"] == material
    assert row["compatibility"] == "not_verified"
    assert row["decision"] == "already_present"


@pytest.mark.parametrize("food,purpose", [
    ("Пакеты для замораживания", "freezing"),
    ("Мешки для заморозки продуктов", "freezing"),
    ("Пакет для хранения пищевых продуктов", "food_storage"),
    ("Пакеты для пищевых продуктов", "food_storage"),
    ("Мешки для хранения продуктов", "food_storage"),
    ("Пакеты для хранения пищи", "food_storage"),
])
@pytest.mark.parametrize("waste", ["пакеты для мусора", "мусорные мешки"])
def test_bag_purpose_conflicts_are_symmetric(food, purpose, waste):
    for title, query, own, requested in [
        (food, waste, purpose, "waste"), (waste, food, "waste", purpose),
    ]:
        row = annotate_recommendations(candidates(query), products(name=title), AS_OF).iloc[0]
        assert row["compatibility"] == "purpose_conflict"
        assert row["reason"] == "explicit_purpose_mismatch"
        assert row["decision"] == "exclude_candidate"
        assert row["product_type"] == row["query_type"] == "bags"
        assert row["product_purpose"] == own
        assert row["query_purpose"] == requested


@pytest.mark.parametrize("query", ["губчатые салфетки", "салфетка губчатая"])
def test_explicit_wipe_subtype_is_not_an_inferred_cellulose_material(query):
    for title, request, own, requested in [
        ("Салфетки из микрофибры", query, "wipes", "sponge_wipes"),
        (query, "салфетки микрофибровые", "sponge_wipes", "wipes"),
    ]:
        row = annotate_recommendations(candidates(request), products(name=title), AS_OF).iloc[0]
        assert row["compatibility"] == "material_conflict"
        assert row["reason"] == "explicit_product_type_mismatch"
        assert row["product_type"] == own
        assert row["query_type"] == requested
        side = "query" if requested == "sponge_wipes" else "product"
        assert row[f"{side}_material"] == ""


@pytest.mark.parametrize("title,query", [
    ("Губка меламиновая", "губка для чистки подошвы"),
    ("Губка меламиновая", "чистка подошвы"),
    ("Салфетки из микрофибры", "салфетки для стекол"),
    ("Салфетки", "губчатые салфетки"),
    ("Губчатые салфетки", "салфетки целлюлозные"),
    ("Салфетки из микрофибры", "губки"),
    ("Салфетки из микрофибры", "перчатки латексные"),
    ("Пакеты для заморозки", "пакеты для хранения пищевых продуктов"),
    ("Пакеты для мусора", "пакеты для переезда"),
    ("Пакеты", "пакеты для мусора"),
    ("Перчатки для заморозки", "перчатки для мусора"),
    ("Салфетки", "салфетки латексные"),
    ("Перчатки нитриловые", "нитриловая резина"),
    ("Салфетки из микрофибры", "микрофибра и целлюлоза"),
    ("Перчатки латексные", "перчатки безлатекса"),
    ("Салфетки из микрофибры", "не губчатые салфетки"),
])
def test_missing_different_properties_and_new_use_cases_do_not_block(title, query):
    row = annotate_recommendations(candidates(query), products(name=title), AS_OF).iloc[0]
    assert row["compatibility"] == "not_verified"
    assert row["decision"] in {"review", "already_present"}


@pytest.mark.parametrize("word", [
    "безлатекса", "безлатексные", "не латексные", "не из латекса",
    "без латекса", "без натурального латекса", "без содержания латекса",
    "не содержит латекс", "латекс-free", "латекс отсутствует", "латекс не содержит",
])
def test_negations_are_not_positive_material_on_either_side(word):
    negative = f"перчатки {word}"
    for title, query, side in [
        ("Перчатки нитриловые", negative, "query"),
        (negative, "перчатки нитриловые", "product"),
    ]:
        row = annotate_recommendations(candidates(query), products(name=title), AS_OF).iloc[0]
        assert row[f"{side}_material"] == ""
        assert row["compatibility"] == "not_verified"
        assert row["decision"] != "exclude_candidate"


def test_negated_secondary_material_does_not_erase_explicit_positive_material():
    row = annotate_recommendations(
        candidates("перчатки виниловые без латекса"),
        products(name="Нитриловые перчатки без латекса"), AS_OF,
    ).iloc[0]
    assert row["product_material"] == "nitrile"
    assert row["query_material"] == "vinyl"
    assert row["compatibility"] == "material_conflict"


@pytest.mark.parametrize("text", [
    "Салфетки микрофибра и целлюлоза", "Салфетки микрофибра, целлюлоза",
    "Салфетки микрофибровые целлюлозные", "Салфетки микрофибра/целлюлоза",
    "Салфетки микрофибра + полиэстер", "Салфетки из микрофибры и полиэстера",
    "Салфетки 80% микрофибра 20% полиэстер", "Салфетки смесь микрофибры",
    "Салфетки микрофибра, полиэстер", "Салфетки микрофибра полиамид",
    "Салфетки из микрофибры хлопок", "Салфетки микрофибра вискоза",
    "Комплект салфеток из микрофибры", "Набор салфеток из микрофибры",
    "Салфетки микрофибровые с латексным покрытием",
    "Салфетки губчатые из микрофибры", "Салфетки из микрофибры, губки",
    "Салфетки из микрофибры для сравнения", "Салфетки из микрофибры вместо губчатых",
    "Салфетки из микрофибры альтернатива губчатым",
    "Салфетки из микрофибры как губчатые",
    "Салфетки двухслойные микрофибровые",
])
def test_composites_sets_and_comparisons_abstain_on_either_side(text):
    for title, query, side in [
        (text, "салфетки латексные", "product"),
        ("Салфетки латексные", text, "query"),
    ]:
        row = annotate_recommendations(candidates(query), products(name=title), AS_OF).iloc[0]
        assert row[f"{side}_material"] == ""
        assert row["compatibility"] == "not_verified"
        assert row["decision"] != "exclude_candidate"


@pytest.mark.parametrize("text", [
    "Пакеты для мусора и заморозки", "Комплект пакетов для заморозки",
    "Пакеты для заморозки подходят для мусора", "Пакеты не для мусора",
    "Пакеты не предназначены для мусора", "Пакеты без мусора",
    "Пакеты для мусора для хранения пищевых продуктов",
])
def test_ambiguous_and_negated_bag_purposes_do_not_block(text):
    for title, query, side in [
        (text, "пакеты для заморозки", "product"),
        ("Пакеты для заморозки", text, "query"),
    ]:
        row = annotate_recommendations(candidates(query), products(name=title), AS_OF).iloc[0]
        assert row[f"{side}_purpose"] == ""
        assert row["compatibility"] == "not_verified"


@pytest.mark.parametrize("title,query", [
    ("Губки меламин, поролон", "губки целлюлозные"),
    ("Перчатки нитриловые с хлопковой подкладкой", "перчатки латексные"),
])
def test_other_explicit_components_only_trigger_abstention(title, query):
    for name, request, side in [(title, query, "product"), (query, title, "query")]:
        row = annotate_recommendations(candidates(request), products(name=name), AS_OF).iloc[0]
        assert row[f"{side}_material"] == ""
        assert row["compatibility"] == "not_verified"
        assert row["decision"] != "exclude_candidate"


@pytest.mark.parametrize("text", [
    "Салфетки микрофибровыеXYZ", "Салфетки супермикрофибра", "Салфетки микрофибра123",
    "Салфеточный материал микрофибра", "Салфетки для чистки микрофибры",
    "Салфетки для ухода за микрофиброй", "Чехол для микрофибровых салфеток",
    "Салфетки с микрофиброй", "Салфетки из полиэстера",
    "Салфетки https://example.test/микрофибра",
    "Салфетки из https://example.test микрофибры",
    '{"name": "Салфетки из микрофибры"}',
])
def test_material_boundaries_target_context_unknown_materials_and_urls(text):
    for title, query, side in [
        (text, "салфетки целлюлозные", "product"),
        ("Салфетки целлюлозные", text, "query"),
    ]:
        row = annotate_recommendations(candidates(query), products(name=title), AS_OF).iloc[0]
        assert row[f"{side}_material"] == ""
        assert row["compatibility"] == "not_verified"


def test_description_and_unknown_attribute_ids_never_supply_own_material_or_purpose():
    row = annotate_recommendations(
        candidates("перчатки нитриловые"),
        products(name="Перчатки", description="Перчатки латексные",
                 attributes_json=attributes((100, ["латексные"]))), AS_OF,
    ).iloc[0]
    assert row["product_material"] == ""
    assert row["query_material"] == "nitrile"
    assert row["compatibility"] == "not_verified"
    bag = annotate_recommendations(
        candidates("пакеты для мусора"),
        products(name="Пакеты", description="Пакеты для заморозки",
                 attributes_json=attributes((100, ["для заморозки"]))), AS_OF,
    ).iloc[0]
    assert bag["product_purpose"] == ""
    assert bag["compatibility"] == "not_verified"
    missing_title = annotate_recommendations(
        candidates("перчатки нитриловые"),
        products(name=None, description="Перчатки латексные",
                 attributes_json=attributes((100, ["латексные"]))), AS_OF,
    ).iloc[0]
    assert missing_title["product_type"] == missing_title["product_material"] == ""
    assert missing_title["compatibility"] == "not_verified"


def test_explicit_title_evidence_survives_incomplete_or_contradictory_other_fields():
    for changes in [
        {"description": None, "attributes_json": "{", "attributes_status": "invalid_json"},
        {"description": "Перчатки виниловые", "attributes_json": attributes((100, ["латекс"]))},
    ]:
        row = annotate_recommendations(
            candidates("перчатки латексные"), products(name="Перчатки нитриловые", **changes),
            AS_OF,
        ).iloc[0]
        assert row["product_material"] == "nitrile"
        assert row["query_material"] == "latex"
        assert row["compatibility"] == "material_conflict"
        assert row["decision"] == "exclude_candidate"


@pytest.mark.parametrize("title,query,status", [
    ("Перчатки нитриловые 150 мл", "dove перчатки латексные 35 мл", "volume_conflict"),
    ("Перчатки нитриловые 150 мл", "dove перчатки латексные 150 мл", "brand_conflict"),
    ("Перчатки нитриловые 150 мл 50 мл", "перчатки латексные 35 мл", "material_conflict"),
    ("Пакеты для заморозки 35 л", "пакеты для мусора 20 л", "volume_conflict"),
    ("Пакеты для заморозки 35 л", "dove пакеты для мусора 35 л", "brand_conflict"),
    ("Комплект салфеток микрофибра 150 мл", "dove салфетки латексные", "brand_conflict"),
    ("Салфетки микрофибра и целлюлоза 150 мл", "салфетки латексные 35 мл", "volume_conflict"),
])
def test_existing_specific_conflicts_are_not_overwritten_by_new_checks(title, query, status):
    catalog = pd.concat([products(name=title), products(sku="2", brand="Dove")])
    row = annotate_recommendations(candidates(query), catalog, AS_OF).iloc[0]
    assert row["compatibility"] == status
    assert row["decision"] == "exclude_candidate"


def test_explicit_evidence_schema_versions_missing_metadata_and_no_mutation():
    rows = candidates("салфетки целлюлозные", "губчатые салфетки", "новый запрос")
    rows.index = [4, 4, 9]
    rows["retrieval_sources"] = ["text", "graph", "text"]
    catalog = products(name="Салфетки из микрофибры https://example.test/private",
                       description="https://example.test/secret", attributes_json="[]")
    before_rows, before_products = rows.copy(deep=True), catalog.copy(deep=True)
    result = annotate_recommendations(rows, catalog, AS_OF)
    pd.testing.assert_frame_equal(rows, before_rows)
    pd.testing.assert_frame_equal(catalog, before_products)
    pd.testing.assert_frame_equal(result[rows.columns], rows)
    assert len(result) == len(rows)
    assert not result["decision"].eq("add_candidate").any()
    assert not result["compatibility"].eq("verified").any()
    fields = [f"{side}_{field}" for side in ("product", "query")
              for field in ("material", "purpose", "type")]
    for field in [*fields, "policy_version", "constraint_vocabulary_version"]:
        assert isinstance(result[field].dtype, pd.StringDtype)
        assert result[field].notna().all()
    assert result["policy_version"].eq(POLICY_VERSION).all()
    assert result["constraint_vocabulary_version"].eq(CONSTRAINT_VOCABULARY_VERSION).all()
    assert "https://" not in result.to_csv(index=False)
    assert "description" not in result and "attributes_json" not in result
    json.dumps(result[fields].to_dict("records"), allow_nan=False)
    repeated = annotate_recommendations(result, catalog, AS_OF)
    pd.testing.assert_frame_equal(repeated, result)
    missing = annotate_recommendations(
        candidates("пакеты для мусора", sku="2"), catalog, AS_OF
    ).iloc[0]
    assert missing["product_material"] == missing["product_purpose"] == missing["product_type"] == ""
    assert missing["query_purpose"] == "waste"
    assert missing["reason"] == "product_metadata_missing"
    assert missing["compatibility"] == "not_verified"
    empty = annotate_recommendations(candidates(), catalog, AS_OF)
    for field in [*fields, "policy_version", "constraint_vocabulary_version"]:
        assert isinstance(empty[field].dtype, pd.StringDtype)


@pytest.fixture
def policy_parity_inputs():
    titles = [
        "Перчатки нитриловые 150 мл", "Перчатки латексные 150 мл",
        "Салфетки из микрофибры 35 л", "Губчатые салфетки латексные",
        "Пакеты латексные для заморозки 35 л", "Пакеты для мусора 20 л",
        "Салфетки 80% микрофибра 20% полиэстер", "Перчатки виниловые без латекса",
        "Пакеты для мусора для хранения продуктов", "Салфетки с микрофиброй",
        "Мыло", "Белое мыло", None, "Салфетки https://example.test микрофибра",
    ]
    queries = (
        "перчатки латексные 35 мл", "dove перчатки латексные 150 мл",
        "перчатки нитриловые 0.15 л", "перчатки без натурального латекса",
        "перчатки латекс отсутствует", "салфетки целлюлозные",
        "салфетки микрофибровые", "губчатые салфетки",
        "салфетки из полиэстера", "салфетки микрофибра полиамид",
        "салфетки микрофибра и целлюлоза", "салфетки микрофибра/целлюлоза",
        "салфетки 100% микрофибра", "салфетки для чистки микрофибры",
        "пакеты нитриловые для мусора", "пакеты для замораживания 35 л",
        "пакеты для хранения пищевых продуктов", "пакеты не для мусора",
        "пакеты для мусора нет", "пакеты для мусора для хранения пищи",
        "мыло dove", "невская косметика мыло", "невская мыло", "новый запрос",
        "нежно белый", "средство 150 мл + 50 мл", "средство -150 мл",
    )
    catalog = pd.concat([
        products(sku=str(index), name=title)
        for index, title in enumerate(titles, start=1)
    ], ignore_index=True)
    catalog.loc[0, "description"] = " / ".join(queries)
    catalog.loc[1, "attributes_json"] = attributes((100, ["нежно белый"]))
    catalog.loc[2, "attributes_json"] = "{"
    catalog.loc[3, "description"] = None
    catalog.loc[10, "brand"] = "Dove"
    catalog.loc[11, "brand"] = "Невская"
    catalog.loc[12, "brand"] = None
    rows = pd.concat([
        candidates(*queries, sku=sku) for sku in [*catalog["sku"], "99"]
    ], ignore_index=True)
    rows["retrieval_sources"] = "text,graph"
    rows.index = [index // 2 for index in range(len(rows))]
    return rows, catalog


def test_synthetic_csv_bytes_match_pre_refactor_snapshot(policy_parity_inputs):
    """Эталон исходной политики: все колонки CSV, без чтения реальных данных."""
    rows, catalog = policy_parity_inputs
    result = annotate_recommendations(rows, catalog, AS_OF)
    pd.testing.assert_frame_equal(result[rows.columns], rows)
    saved = io.BytesIO()
    result.to_csv(saved, index=False, lineterminator="\n")
    content = saved.getvalue()
    assert result.shape == (405, 21)
    assert len(content) == 118438
    assert hashlib.sha256(content).hexdigest() == (
        "b7db7666b320f4dd0a9dca596d1788d07140f1bbc3a7882d2bcd70ac4e01eaf0"
    )