import pytest

from catalog import resolve_country, resolve_metric
from entities import MAX_TOP_N, Entities, parse_llm_json


@pytest.mark.parametrize("text,expected", [
    ("Orders", "Orders"), ("pedidos", "Orders"), ("órdenes perfectas", "Perfect Orders"),
    ("ordenes perfectas", "Perfect Orders"), ("la ganancia por orden", "Gross Profit per Order"),
    ("CVR restaurantes", "Restaurant Conversion Rate"), ("bananas", None), ("", None),
])
def test_resolve_metric(text, expected):
    assert resolve_metric(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("Colombia", "CO"), ("co", "CO"), ("México", "MX"), ("Bogota", "CO"), ("Narnia", None),
])
def test_resolve_country(text, expected):
    assert resolve_country(text) == expected


def test_markdown_fenced_json_and_list_metric():
    e = parse_llm_json('```json\n{"intent":"ranking","metric":["pedidos"],"country":"colombia","top_n":"5"}\n```')
    assert (e.intent, e.metric, e.country, e.top_n) == ("ranking", "Orders", "CO", 5)


@pytest.mark.parametrize("raw,expected", [("five", 5), ("top 10", 10), (999, MAX_TOP_N), (-3, None), ("muchas", None)])
def test_top_n_is_coerced_and_clamped(raw, expected):
    assert Entities(top_n=raw).top_n == expected


def test_unknown_values_are_dropped_not_guessed():
    e = parse_llm_json('{"intent":"hack","metric":"DROP TABLE","country":"Atlantis","concept":"x == 1","sort_order":"sideways"}')
    assert e.intent == "summary"
    assert e.metric is None and e.country is None and e.concept is None
    assert e.sort_order == "desc"


@pytest.mark.parametrize("raw", ["", "not json", "[1,2]", "{broken"])
def test_garbage_never_raises(raw):
    assert parse_llm_json(raw) == Entities()


@pytest.mark.parametrize("raw,expected", [
    ("Wealthy vs Non Wealthy", "comparison"), ("non wealthy", "Non Wealthy"), ("Wealthy", "Wealthy"), ("rich", None),
])
def test_zone_type(raw, expected):
    assert Entities(zone_type=raw).zone_type == expected


def test_as_context_drops_nones():
    assert Entities(intent="ranking", metric="Orders").as_context() == {
        "intent": "ranking", "metric": "Orders", "sort_order": "desc"}
