"""
data_loader.py
Carga las 3 hojas del Excel y genera los DataFrames listos para análisis.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

WEEK_COLS_ROLL = ["L8W_ROLL", "L7W_ROLL", "L6W_ROLL", "L5W_ROLL",
                  "L4W_ROLL", "L3W_ROLL", "L2W_ROLL", "L1W_ROLL", "L0W_ROLL"]
WEEK_COLS_ORD  = ["L8W", "L7W", "L6W", "L5W", "L4W", "L3W", "L2W", "L1W", "L0W"]
WEEK_NUM_MAP   = {f"L{8-i}W": i for i in range(9)}   # L8W→0 ... L0W→8


def load_excel(path: str = "data.xlsx") -> dict:
    """
    Retorna dict con:
      raw_metrics  - hoja RAW_INPUT_METRICS tal cual
      raw_orders   - hoja RAW_ORDERS tal cual
      metrics_long - métricas en formato long con features
      orders_long  - órdenes en formato long con features
      combined     - métricas + órdenes unidas
    """
    xl = pd.ExcelFile(path)
    df_m = pd.read_excel(xl, sheet_name="RAW_INPUT_METRICS")
    df_o = pd.read_excel(xl, sheet_name="RAW_ORDERS")

    # Normalizar nombres de columna (strip espacios)
    df_m.columns = df_m.columns.str.strip()
    df_o.columns = df_o.columns.str.strip()

    metrics_long = _process_metrics(df_m)
    orders_long  = _process_orders(df_o)
    combined     = _combine(metrics_long, orders_long)

    return {
        "raw_metrics":  df_m,
        "raw_orders":   df_o,
        "metrics_long": metrics_long,
        "orders_long":  orders_long,
        "combined":     combined,
    }


# ── Transformaciones ──────────────────────────────────────────────────────────

def _process_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Wide → long para RAW_INPUT_METRICS + features derivados."""
    id_vars = ["COUNTRY", "CITY", "ZONE", "ZONE_TYPE", "ZONE_PRIORITIZATION", "METRIC"]
    long = df.melt(id_vars=id_vars, value_vars=WEEK_COLS_ROLL,
                   var_name="WEEK_LABEL", value_name="VALUE")

    # Semana como int 0-8 (0=más antigua, 8=más reciente)
    # L8W_ROLL tiene el número 8 (más antigua → 0), L0W_ROLL → 8 (más reciente)
    week_num = {col: 8 - int(col[1]) for col in WEEK_COLS_ROLL}

    long["WEEK_NUM"] = long["WEEK_LABEL"].map(week_num)
    long = long.sort_values(["COUNTRY", "CITY", "ZONE", "METRIC", "WEEK_NUM"]).reset_index(drop=True)

    features = _compute_features(long, group_cols=["COUNTRY", "CITY", "ZONE", "ZONE_TYPE",
                                                    "ZONE_PRIORITIZATION", "METRIC"])
    return features


def _process_orders(df: pd.DataFrame) -> pd.DataFrame:
    """Wide → long para RAW_ORDERS + features derivados."""
    id_vars = ["COUNTRY", "CITY", "ZONE", "METRIC"]
    long = df.melt(id_vars=id_vars, value_vars=WEEK_COLS_ORD,
                   var_name="WEEK_LABEL", value_name="VALUE")

    week_num = {col: 8 - int(col[1]) for col in WEEK_COLS_ORD}

    long["WEEK_NUM"] = long["WEEK_LABEL"].map(week_num)
    long = long.sort_values(["COUNTRY", "CITY", "ZONE", "METRIC", "WEEK_NUM"]).reset_index(drop=True)

    # Agregar columnas vacías para unificar schema
    long["ZONE_TYPE"]           = None
    long["ZONE_PRIORITIZATION"] = None

    features = _compute_features(long, group_cols=["COUNTRY", "CITY", "ZONE", "METRIC"])
    return features


def _compute_features(long: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """
    Por cada combinación (zona, métrica) calcula:
      L0W_VALUE, L1W_VALUE: valores recientes
      PCT_CHANGE_WOW: cambio % última semana
      PCT_CHANGE_TOTAL: cambio % vs 8 semanas atrás
      TREND_SLOPE: pendiente de regresión lineal
      IS_DECLINING_3W: deterioro 3 semanas consecutivas
      ZSCORE_VS_CITY: z-score dentro de su ciudad y métrica
    """
    records = []
    for keys, grp in long.groupby(group_cols):
        grp = grp.sort_values("WEEK_NUM")
        vals = grp["VALUE"].values

        l0w = vals[-1]
        l1w = vals[-2] if len(vals) >= 2 else np.nan
        l8w = vals[0]

        pct_wow   = (l0w - l1w) / abs(l1w) * 100 if (l1w and l1w != 0 and not np.isnan(l1w)) else np.nan
        pct_total = (l0w - l8w) / abs(l8w) * 100 if (l8w and l8w != 0 and not np.isnan(l8w)) else np.nan

        x = np.arange(len(vals))
        try:
            slope, _, r2, _, _ = stats.linregress(x, vals)
        except Exception:
            slope, r2 = np.nan, np.nan

        if len(vals) >= 3:
            last3 = vals[-3:]
            is_declining = bool(last3[0] > last3[1] > last3[2])
            is_improving = bool(last3[0] < last3[1] < last3[2])
        else:
            is_declining = is_improving = False

        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else [keys]))
        row.update({
            "L0W_VALUE":       round(float(l0w), 6) if not np.isnan(l0w) else None,
            "L1W_VALUE":       round(float(l1w), 6) if not np.isnan(l1w) else None,
            "L8W_VALUE":       round(float(l8w), 6) if not np.isnan(l8w) else None,
            "PCT_CHANGE_WOW":  round(float(pct_wow), 2)   if not np.isnan(pct_wow) else None,
            "PCT_CHANGE_TOTAL":round(float(pct_total), 2) if not np.isnan(pct_total) else None,
            "TREND_SLOPE":     round(float(slope), 6)     if not np.isnan(slope) else None,
            "IS_DECLINING_3W": is_declining,
            "IS_IMPROVING_3W": is_improving,
        })
        records.append(row)

    df_feat = pd.DataFrame(records)

    # Z-score por CITY + METRIC
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        df_feat["ZSCORE_VS_CITY"] = (
            df_feat.groupby(["CITY", "METRIC"])["L0W_VALUE"]
            .transform(lambda x: stats.zscore(x.astype(float), nan_policy="omit"))
            .round(2)
        )
    return df_feat


def _combine(metrics: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """Une métricas y órdenes en un solo DataFrame para queries cruzados."""
    m = metrics.copy()
    o = orders.copy()
    shared = ["COUNTRY", "CITY", "ZONE", "METRIC", "L0W_VALUE", "L1W_VALUE",
              "PCT_CHANGE_WOW", "PCT_CHANGE_TOTAL", "TREND_SLOPE",
              "IS_DECLINING_3W", "IS_IMPROVING_3W", "ZSCORE_VS_CITY"]
    m["SOURCE"] = "metrics"
    o["SOURCE"] = "orders"
    return pd.concat([m[shared + ["SOURCE"]], o[shared + ["SOURCE"]]], ignore_index=True)