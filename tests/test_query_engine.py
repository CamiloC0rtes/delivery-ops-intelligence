from query_engine import execute_correlation, execute_multivariable, execute_query


def run(data, **ctx):
    return execute_query(ctx, data["metrics_long"], data["orders_long"])


def test_top_drops_match_pandas(data):
    df, _ = run(data, intent="ranking", metric="Orders", top_n=5, sort_order="asc")
    o = data["orders_long"].dropna(subset=["PCT_CHANGE_WOW"])
    # Business rule: tiny zones (bottom 5% by volume) are excluded so 3 -> 1 orders isn't "the biggest drop"
    o = o[o.L0W_VALUE >= o.L0W_VALUE.quantile(0.05)]
    assert list(df.ZONE) == list(o.nsmallest(5, "PCT_CHANGE_WOW").ZONE)


def test_country_filter(data):
    df, _ = run(data, metric="Perfect Orders", country="CO")
    assert not df.empty and set(df.COUNTRY) == {"CO"}


def test_comparison_wealthy_vs_non(data):
    df, _ = run(data, intent="comparison", metric="Perfect Orders", zone_type="comparison")
    assert set(df.ZONE_TYPE) == {"Wealthy", "Non Wealthy"}


def test_sustained_decline_concept(data):
    df, _ = run(data, metric="Perfect Orders", concept="deterioro sostenido")
    assert len(df) > 0 and df.IS_DECLINING_3W.all()


def test_unknown_metric_and_city_do_not_crash(data):
    df, desc = run(data, metric="bananas", city="Gotham", top_n=3)
    assert len(df) == 3 and "registros" in desc


def test_multivariable_and_correlation(data):
    df, _ = execute_multivariable({"metric_high": "Lead Penetration", "metric_low": "Perfect Orders"}, data["metrics_long"])
    assert not df.empty
    corr, _ = execute_correlation(data["metrics_long"], min_r=0.5)
    pairs = {frozenset((a, b)) for a, b in zip(corr.METRIC_A, corr.METRIC_B, strict=True)}
    assert frozenset(("Gross Profit per Order", "Subscription Adoption")) in pairs
