"""The eval's scoring, checked offline with a fake LLM."""

import json

from fastapi.testclient import TestClient

import app as app_module
from tests.conftest import FakeLLM
from tests.eval.run_eval import score_case

CASE = {"id": "drops", "question": "caída de órdenes",
        "expect": {"metric": "Orders", "top_n": 5, "sort_order": "asc"}, "mention_top": 3}


def run(extraction):
    app_module.DATA["client"] = FakeLLM({"caída": json.dumps(extraction)})
    try:
        with TestClient(app_module.app) as c:
            zones = set(app_module.DATA["orders_long"]["ZONE"])
            return score_case(CASE, c, app_module.DATA, zones)
    finally:
        app_module.DATA.pop("client", None)


def test_correct_extraction_passes():
    r = run({"intent": "ranking", "metric": "Orders", "top_n": 5, "sort_order": "asc"})
    assert r["passed"], r["failures"]


def test_wrong_direction_is_caught():
    r = run({"intent": "ranking", "metric": "Orders", "top_n": 5, "sort_order": "desc"})
    assert not r["passed"]
    assert any("sort_order" in f for f in r["failures"])
    assert any("zones not in result" in f or "missing top zone" in f for f in r["failures"])
