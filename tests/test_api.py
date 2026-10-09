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
