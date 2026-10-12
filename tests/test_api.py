import json

import pytest
from fastapi.testclient import TestClient

import app as app_module
from tests.conftest import FakeLLM


@pytest.fixture
def client():
    fake = FakeLLM({
        "caída de órdenes": json.dumps({"intent": "ranking", "metric": "Orders", "top_n": 5, "sort_order": "asc"}),
        "basura": "esto no es json",
        "Argentina": json.dumps({"country": "Argentina", "is_new_topic": False}),
        "Venezuela": json.dumps({"intent": "ranking", "metric": "Orders", "country": "Venezuela", "top_n": 5}),
        "NPS": json.dumps({"intent": "filter", "metric": "NPS", "city": "Bogota"}),
        "Narnia": json.dumps({"intent": "summary", "metric": "Orders", "zone": "Narnia Central"}),
        "llover": json.dumps({"intent": "out_of_scope"}),
        "más pedidos": json.dumps({"intent": "ranking", "metric": "Orders"}),
        "cuántas zonas": json.dumps({"intent": "summary"}),
        "Guadalajara": json.dumps({"city": "Guadalajara", "is_new_topic": False}),
        "Perfect Orders?": json.dumps({"metric": "Perfect Orders", "is_new_topic": False}),
    })
    app_module.DATA["client"] = fake
    with TestClient(app_module.app) as c:
        yield c, fake
    app_module.DATA.pop("client", None)


def test_chat_answer_is_built_from_queried_rows(client, data):
    c, _ = client
    r = c.post("/api/chat", json={"message": "Top 5 zonas con mayor caída de órdenes"}).json()
    worst = data["orders_long"].nsmallest(1, "PCT_CHANGE_WOW").ZONE.iloc[0]
    assert r["rows_found"] == 5
    assert worst in r["answer"]
    assert r["chart"]["labels"] and r["context"]["metric"] == "Orders"


def test_malformed_extraction_degrades_gracefully(client):
    c, _ = client
    r = c.post("/api/chat", json={"message": "basura"})
    assert r.status_code == 200 and r.json()["context"]["intent"] == "summary"


def test_follow_up_keeps_session(client):
    c, fake = client
    first = c.post("/api/chat", json={"message": "caída de órdenes"}).json()
    c.post("/api/chat", json={"message": "caída de órdenes", "session_id": first["session_id"]})
    extraction_calls = [m for m in fake.calls if m[0]["content"].startswith("Eres un extractor")]
    assert "Preguntas anteriores" in extraction_calls[-1][-1]["content"]


@pytest.mark.parametrize("url", [
    "/", "/health", "/api/insights?severity=high&tipo=anomaly", "/api/insights/summary", "/api/filters",
    "/api/ranking?country=CO", "/api/analysis/country-summary", "/api/correlations", "/api/executive-report",
    "/api/analysis/timeseries-multi?country=CO&city=Bogota", "/api/multivariable", "/api/cities?country=CO",
])
def test_endpoints(client, url):
    c, _ = client
    assert c.get(url).status_code == 200


def test_insight_filters_combine(client):
    c, _ = client
    rows = c.get("/api/insights?severity=high&tipo=anomaly&limit=100").json()["insights"]
    assert rows and all(i["severity"] == "high" and i["tipo"] == "anomaly" for i in rows)


def test_chat_without_key_returns_503():
    app_module.DATA["client"] = None
    with TestClient(app_module.app) as c:
        assert c.post("/api/chat", json={"message": "hola"}).status_code == 503
        assert c.get("/health").json()["chat_enabled"] is False
    app_module.DATA.pop("client", None)


def test_follow_up_only_changes_what_the_model_sent(client):
    c, _ = client
    first = c.post("/api/chat", json={"message": "Top 5 zonas con mayor caída de órdenes"}).json()
    r = c.post("/api/chat", json={"message": "¿Y en Argentina?", "session_id": first["session_id"]}).json()
    ctx = r["context"]
    assert ctx["country"] == "AR"
    assert (ctx["metric"], ctx["top_n"], ctx["sort_order"]) == ("Orders", 5, "asc")  # kept, not reset to defaults
    assert "city" not in ctx


def test_extraction_requests_json_mode_with_room(client):
    c, fake = client
    c.post("/api/chat", json={"message": "caída de órdenes"})
    # FakeLLM records messages only; check the call contract through app code instead
    import inspect
    src = inspect.getsource(app_module.extract_entities)
    assert '"json_object"' in src and "max_tokens=400" in src


@pytest.mark.parametrize("text,expected", [
    ("Las 3 zonas de Medellin con mayor caída", 3), ("top 5 zonas", 5), ("dame 10 zonas", 10),
    ("¿cómo va Bogota?", None), ("zonas con 20% de caída", None),
])
def test_top_n_from_text(text, expected):
    assert app_module.top_n_from_text(text) == expected


@pytest.mark.parametrize("message,needle", [
    ("Top 5 caídas en Venezuela", "«Venezuela»"),
    ("¿Cuál es el NPS de Bogota?", "«NPS»"),
    ("¿Cuántas órdenes tuvo Narnia?", "«Narnia Central»"),
])
def test_unknown_filters_get_an_explicit_answer_not_global_data(client, message, needle):
    c, fake = client
    r = c.post("/api/chat", json={"message": message}).json()
    assert needle in r["answer"] and r["rows_found"] == 0
    assert len(fake.calls) == 1  # extraction only: no analysis call over unrelated data


def test_out_of_scope_gets_fixed_reply_without_data(client):
    c, fake = client
    r = c.post("/api/chat", json={"message": "¿Va a llover mañana?"}).json()
    assert r["answer"] == app_module.OUT_OF_SCOPE_ANSWER and r["rows_found"] == 0
    assert len(fake.calls) == 1


@pytest.mark.parametrize("message", ["", "   ", "x" * (app_module.MAX_MESSAGE_CHARS + 1)])
def test_blank_or_huge_messages_are_rejected_before_the_llm(client, message):
    c, fake = client
    assert c.post("/api/chat", json={"message": message}).status_code == 422
    assert fake.calls == []


def test_most_orders_ranks_by_volume_not_change(client, data):
    c, _ = client
    r = c.post("/api/chat", json={"message": "¿Cuál es la zona con más pedidos?"}).json()
    assert r["context"]["rank_by"] == "value"
    assert data["orders_long"].nlargest(1, "L0W_VALUE").ZONE.iloc[0] in r["answer"]


def test_aggregates_come_from_the_full_result(client, data):
    c, fake = client
    c.post("/api/chat", json={"message": "¿cuántas zonas hay?"})
    sent = fake.calls[-1][-1]["content"]
    assert f"Zonas distintas: {data['orders_long'].ZONE.nunique()}" in sent
    assert "muestra: primeras 20" in sent


def test_extractor_sees_filters_in_force(client):
    # Regression: after "¿y en Perfect Orders?", "ahora solo Guadalajara" reverted the metric to Orders
    c, fake = client
    s = c.post("/api/chat", json={"message": "caída de órdenes"}).json()["session_id"]
    c.post("/api/chat", json={"message": "¿y en Perfect Orders?", "session_id": s})
    r = c.post("/api/chat", json={"message": "ahora solo Guadalajara", "session_id": s}).json()
    assert r["context"]["metric"] == "Perfect Orders" and r["context"]["city"] == "Guadalajara"
    assert "Filtros vigentes" in fake.calls[-2][-1]["content"]


@pytest.mark.parametrize("url,status", [
    ("/api/insights?limit=200", 200), ("/api/insights?limit=-5", 422), ("/api/insights?limit=100000", 422),
    ("/api/search-zones?q=%25%27%22(", 200), ("/api/search-zones?q=los&limit=-3", 422),
    ("/api/zone-detail?zone=Los&country=CO", 409), ("/api/zone-detail?zone=Los%20Cedros&country=co", 200),
    ("/api/correlations?min_r=5", 422),
])
def test_hostile_query_params(client, url, status):
    c, _ = client
    assert c.get(url).status_code == status


def test_country_codes_are_case_insensitive(client):
    c, _ = client
    assert c.get("/api/ranking?country=co").json()["data"]
    assert c.get("/api/ranking?country=Colombia").json()["data"]


def test_nan_never_breaks_json():
    from app import _json_safe

    assert _json_safe({"a": [float("nan"), 1.0, {"b": float("inf")}]}) == {"a": [None, 1.0, {"b": None}]}
