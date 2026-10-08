import pandas as pd

from catalog import COUNTRIES, METRICS, ORDERS
from scripts.generate_data import generate
from validate_data import EXPECTED_METRICS_COLS, EXPECTED_ORDERS_COLS


def test_deterministic():
    a, b = generate(7), generate(7)
    pd.testing.assert_frame_equal(a[0], b[0])
    pd.testing.assert_frame_equal(a[1], b[1])


def test_schema_matches_loader_expectations():
    metrics, orders = generate(1)
    assert list(metrics.columns) == EXPECTED_METRICS_COLS
    assert list(orders.columns) == EXPECTED_ORDERS_COLS
    assert set(metrics.METRIC) == set(METRICS) - {ORDERS}
    assert set(orders.COUNTRY) <= set(COUNTRIES)
    assert orders.ZONE.is_unique


def test_planted_patterns_are_detectable(data):
    orders = data["orders_long"]
    assert (orders.PCT_CHANGE_WOW < -20).sum() >= 5   # sharp drops
    assert (orders.PCT_CHANGE_WOW > 20).sum() >= 5    # strong growth
    perfect = data["metrics_long"].query("METRIC == 'Perfect Orders'")
    assert perfect.IS_DECLINING_3W.sum() >= 8          # sustained declines
