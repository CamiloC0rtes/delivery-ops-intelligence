"""
query_engine.py
Ejecuta consultas pandas sobre los DataFrames según entidades extraídas por el LLM.
"""

import unicodedata

import pandas as pd

from catalog import ORDERS, resolve_country, resolve_metric

SEMANTIC_CONCEPTS = {
    "zonas problemáticas":  "PCT_CHANGE_WOW < -10",
    "alto crecimiento":     "PCT_CHANGE_WOW > 10",
    "bajo performance":     "ZSCORE_VS_CITY < -1.5",
    "deterioro sostenido":  "IS_DECLINING_3W == True",
    "mejora sostenida":     "IS_IMPROVING_3W == True",
    "mejor zona":           None,
    "peor zona":            None,
    "outliers":             "ZSCORE_VS_CITY.abs() > 2",
    "problem zones":        "PCT_CHANGE_WOW < -10",
    "high growth":          "PCT_CHANGE_WOW > 10",
    "low performance":      "ZSCORE_VS_CITY < -1.5",
    "sustained decline":    "IS_DECLINING_3W == True",
    "best zone":            None,
    "worst zone":           None,
}

def _cc(country: str) -> str:
    return resolve_country(country) or str(country).upper()


def execute_multivariable(entities: dict, df_metrics) -> tuple:
    """Zonas con alto valor en metric_high y bajo en metric_low."""
    metric_high = entities.get("metric_high") or "Lead Penetration"
    metric_low  = entities.get("metric_low")  or "Perfect Orders"
    country     = entities.get("country") or ""
    threshold   = float(entities.get("threshold_pct") or 0.25)

    df = df_metrics.copy()
    if country:
        cc = _cc(country)
        df = df[df["COUNTRY"] == cc]

    pivot = df.pivot_table(
        index=["COUNTRY","CITY","ZONE"], columns="METRIC", values="L0W_VALUE"
    ).reset_index()

    if metric_high not in pivot.columns or metric_low not in pivot.columns:
        available = [c for c in pivot.columns if c not in ("COUNTRY","CITY","ZONE")]
        return pd.DataFrame(), (
            f"Metricas no encontradas. Disponibles: {available[:5]}"
        )

    q_high = pivot[metric_high].quantile(1 - threshold)
    q_low  = pivot[metric_low].quantile(threshold)
    result = pivot[(pivot[metric_high] >= q_high) & (pivot[metric_low] <= q_low)].copy()
    result = result.sort_values(metric_high, ascending=False).head(20)
    cols   = [c for c in ["COUNTRY","CITY","ZONE", metric_high, metric_low] if c in result.columns]
    return result[cols].round(4).reset_index(drop=True), (
        f"Encontre {len(result)} zonas con alto {metric_high} y bajo {metric_low}"
        + (f" en {country.upper()}" if country else "")
    )


def execute_correlation(df_metrics, country: str = "", min_r: float = 0.4) -> tuple:
    """Calcula correlaciones de Pearson entre metricas."""
    df = df_metrics.copy()
    if country:
        cc = _cc(country)
        df = df[df["COUNTRY"] == cc]

    pivot = df.pivot_table(
        index=["COUNTRY","CITY","ZONE"], columns="METRIC", values="L0W_VALUE"
    )
    corr  = pivot.corr(method="pearson").round(3)
    cols  = list(corr.columns)
    pairs = []
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            r = float(corr.iloc[i,j])
            if abs(r) >= min_r and r == r:  # not NaN
                pairs.append({
                    "METRIC_A":    cols[i][:40],
                    "METRIC_B":    cols[j][:40],
                    "CORRELATION": r,
                    "STRENGTH":    "fuerte" if abs(r) >= 0.7 else "moderada",
                    "DIRECTION":   "positiva" if r > 0 else "negativa",
                })
    pairs.sort(key=lambda x: abs(x["CORRELATION"]), reverse=True)
    result = pd.DataFrame(pairs[:15])
    return result, (
        "Top correlaciones entre metricas"
        + (f" en {country.upper()}" if country else " (global)")
    )


def execute_query(
    entities: dict,
    df_metrics: pd.DataFrame,
    df_orders: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    """
    Traduce entidades del LLM a una consulta pandas.
    Siempre retorna (DataFrame, descripción_string) sin lanzar excepciones.
    """
    # Valores seguros (nunca None en la lógica)
    raw_m = entities.get("metric")
    metric_raw = raw_m[0] if isinstance(raw_m, list) else (raw_m or "")
    city       = entities.get("city") or ""
    country    = entities.get("country") or ""
    zone       = entities.get("zone") or ""
    concept    = (entities.get("concept") or "").strip().lower()
    zone_type  = entities.get("zone_type") or ""
    priority   = entities.get("priority") or ""
    intent     = entities.get("intent") or ""
    sort_order = entities.get("sort_order") or "desc"
    top_n_raw  = entities.get("top_n")
    top_n      = int(top_n_raw) if top_n_raw else None

    # ── 1. Seleccionar DataFrame base ─────────────────────────
    resolved_metric = resolve_metric(metric_raw)

    if resolved_metric == ORDERS:
        df = df_orders.copy()
        source_label = "órdenes"
    elif resolved_metric:
        df = df_metrics.copy()
        source_label = "métricas"
    else:
        df = pd.concat([df_metrics, df_orders], ignore_index=True)
        source_label = "datos"

    # ── 2. Filtro por métrica ─────────────────────────────────
    if resolved_metric:
        df = df[df["METRIC"] == resolved_metric]

    # ── 3. Filtros geográficos ────────────────────────────────
    def _norm(s):
        return unicodedata.normalize("NFD", str(s)).encode("ascii","ignore").decode().lower()

    if city:
        df_by_city = df[df["CITY"].apply(_norm) == _norm(city)]
        if not df_by_city.empty:
            df = df_by_city
        else:
            # Fallback: city podria ser una zona (ej: Chapinero -> zona de Bogota)
            df_by_zone = df[df["ZONE"].apply(_norm).str.contains(_norm(city), na=False)]
            if not df_by_zone.empty:
                df = df_by_zone

    if country:
        cc = _cc(country)
        df = df[df["COUNTRY"] == cc]

    if zone:
        df = df[df["ZONE"].apply(_norm).str.contains(_norm(zone), na=False)]

    # ── 4. Filtros de tipo / prioridad ────────────────────────
    # Detectar comparacion Wealthy vs Non Wealthy
    is_comparison = (
        intent == "comparison" or
        zone_type == "comparison"
    )

    if is_comparison and "ZONE_TYPE" in df.columns:
        grp = df.groupby("ZONE_TYPE").agg(
            L0W_VALUE=("L0W_VALUE","mean"),
            PCT_CHANGE_WOW=("PCT_CHANGE_WOW","mean"),
            TREND_SLOPE=("TREND_SLOPE","mean"),
            pct_declining=("IS_DECLINING_3W","mean"),
            n_zonas=("ZONE","nunique"),
        ).reset_index()
        grp["pct_declining"]  = (grp["pct_declining"]*100).round(1)
        grp["L0W_VALUE"]      = grp["L0W_VALUE"].round(4)
        grp["PCT_CHANGE_WOW"] = grp["PCT_CHANGE_WOW"].round(3)
        grp["TREND_SLOPE"]    = grp["TREND_SLOPE"].round(6)
        grp["METRIC"]  = resolved_metric or "todas"
        grp["COUNTRY"] = _cc(country) if country else "todos"
        cols = ["ZONE_TYPE","METRIC","COUNTRY","L0W_VALUE","PCT_CHANGE_WOW","TREND_SLOPE","pct_declining","n_zonas"]
        cols = [c for c in cols if c in grp.columns]
        return grp[cols].reset_index(drop=True), f"Comparacion Wealthy vs Non Wealthy: {resolved_metric or 'todas'} en {country.upper() if country else 'global'}"

    if zone_type in ("Wealthy", "Non Wealthy") and "ZONE_TYPE" in df.columns:
        df = df[df["ZONE_TYPE"] == zone_type]

    if priority and "ZONE_PRIORITIZATION" in df.columns:
        df = df[df["ZONE_PRIORITIZATION"] == priority]

    # ── 5. Concepto semántico ─────────────────────────────────
    # Conditions come only from the fixed SEMANTIC_CONCEPTS table, never from LLM text.
    if concept and concept in SEMANTIC_CONCEPTS:
        condition = SEMANTIC_CONCEPTS[concept]
        if condition:
            try:
                df = df.query(condition)
            except Exception:
                pass  # Si falla el query (columna ausente), continuar sin filtro

    # ── 6. Ranking / Top N ────────────────────────────────────
    # Filtro mínimo de volumen: excluir zonas con volumen insignificante
    # Solo aplica cuando ordenamos por L0W_VALUE, no por PCT_CHANGE_WOW
    _sort_preview = _pick_sort_col(intent, concept)
    if top_n and "L0W_VALUE" in df.columns and _sort_preview == "L0W_VALUE":
        min_val = df["L0W_VALUE"].quantile(0.10)
        if min_val > 0:
            df_ranked = df[df["L0W_VALUE"] >= min_val]
            if len(df_ranked) >= (top_n or 5):
                df = df_ranked

    # Cuando se buscan caídas/crecimiento, filtrar nulos en PCT_CHANGE_WOW
    if top_n and _sort_preview == "PCT_CHANGE_WOW" and "PCT_CHANGE_WOW" in df.columns:
        df = df.dropna(subset=["PCT_CHANGE_WOW"])
        # Volumen mínimo absoluto para evitar zonas con 1-5 unidades
        if "L0W_VALUE" in df.columns:
            min_abs = df["L0W_VALUE"].quantile(0.05)
            if min_abs > 0:
                df_vol = df[df["L0W_VALUE"] >= min_abs]
                if len(df_vol) >= (top_n or 5):
                    df = df_vol

    if "mejor" in concept or "best" in concept:
        df = df.nlargest(top_n or 5, "L0W_VALUE")
    elif "peor" in concept or "worst" in concept:
        df = df.nsmallest(top_n or 5, "L0W_VALUE")
    elif top_n:
        ascending = (sort_order == "asc")
        # Si piden orden ascendente (caídas/peores), priorizar PCT_CHANGE_WOW
        # Si piden orden descendente (crecimiento/mejores), priorizar L0W_VALUE o PCT_CHANGE_WOW
        if "PCT_CHANGE_WOW" in df.columns:
            # asc = mayores caídas, desc = mayor crecimiento → siempre PCT_CHANGE_WOW
            sort_col = "PCT_CHANGE_WOW"
        else:
            sort_col = _pick_sort_col(intent, concept)
        if sort_col not in df.columns:
            sort_col = "L0W_VALUE"
        df = (df.nsmallest(top_n, sort_col) if ascending
              else df.nlargest(top_n, sort_col))

    # ── 7. Columnas de salida ─────────────────────────────────
    cols   = _display_columns(df)
    result = df[cols].reset_index(drop=True)

    desc = f"Encontré {len(result)} registros en {source_label}"
    if resolved_metric:
        desc += f" · métrica: '{resolved_metric}'"
    if city:
        desc += f" · ciudad: {city}"
    if country:
        desc += f" · país: {country.upper()}"

    return result, desc


# ── Helpers ───────────────────────────────────────────────────

def _pick_sort_col(intent: str, concept: str) -> str:
    combined = f"{intent} {concept}".lower()
    if any(w in combined for w in ("growth","crecimiento","caída","caida","caídas","caidas","bajada","descenso","drop","decline")):
        return "PCT_CHANGE_WOW"
    if any(w in combined for w in ("trend","tendencia")):
        return "TREND_SLOPE"
    return "L0W_VALUE"


def _display_columns(df: pd.DataFrame) -> list:
    preferred = [
        "COUNTRY", "CITY", "ZONE", "METRIC",
        "L0W_VALUE", "L1W_VALUE", "PCT_CHANGE_WOW",
        "TREND_SLOPE", "IS_DECLINING_3W", "ZSCORE_VS_CITY",
        "ZONE_TYPE", "ZONE_PRIORITIZATION",
    ]
    return [c for c in preferred if c in df.columns]