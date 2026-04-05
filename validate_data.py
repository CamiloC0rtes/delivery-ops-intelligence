"""
validate_data.py
Valida la estructura y calidad del Excel antes de correr el sistema.
Uso: python validate_data.py
     python validate_data.py --data otra_ruta.xlsx
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# ── Esquemas esperados ────────────────────────────────────────
EXPECTED_METRICS_COLS = [
    "COUNTRY", "CITY", "ZONE", "ZONE_TYPE", "ZONE_PRIORITIZATION",
    "METRIC", "L8W_ROLL", "L7W_ROLL", "L6W_ROLL", "L5W_ROLL",
    "L4W_ROLL", "L3W_ROLL", "L2W_ROLL", "L1W_ROLL", "L0W_ROLL"
]
EXPECTED_ORDERS_COLS = [
    "COUNTRY", "CITY", "ZONE", "METRIC",
    "L8W", "L7W", "L6W", "L5W", "L4W", "L3W", "L2W", "L1W", "L0W"
]
VALID_COUNTRIES  = {"CO","PE","AR","MX","BR","CL","EC","UY","CR"}
VALID_ZONE_TYPES = {"Wealthy","Non Wealthy"}
VALID_PRIORITIES = {"High Priority","Prioritized","Not Prioritized"}
WEEK_COLS_M = ["L8W_ROLL","L7W_ROLL","L6W_ROLL","L5W_ROLL",
               "L4W_ROLL","L3W_ROLL","L2W_ROLL","L1W_ROLL","L0W_ROLL"]
WEEK_COLS_O = ["L8W","L7W","L6W","L5W","L4W","L3W","L2W","L1W","L0W"]

OK   = "  ✅"
WARN = "  ⚠️ "
ERR  = "  ❌"

issues = []
warnings = []

def ok(msg):   print(f"{OK} {msg}")
def warn(msg): print(f"{WARN} {msg}"); warnings.append(msg)
def err(msg):  print(f"{ERR} {msg}");  issues.append(msg)

def check_cols(df, expected, sheet):
    missing = [c for c in expected if c not in df.columns]
    extra   = [c for c in df.columns if c not in expected]
    if missing: err(f"[{sheet}] Columnas faltantes: {missing}")
    else:       ok(f"[{sheet}] Todas las columnas requeridas presentes")
    if extra:   warn(f"[{sheet}] Columnas extra (se ignoran): {extra}")

def check_nulls(df, week_cols, sheet):
    null_counts = df[week_cols].isnull().sum()
    total_nulls = null_counts.sum()
    if total_nulls == 0:
        ok(f"[{sheet}] Sin valores nulos en columnas de semana")
    else:
        pct = total_nulls / (len(df) * len(week_cols)) * 100
        # Orders threshold is higher (25%) — inactive zones naturally have nulls
        threshold = 25.0 if "ORDERS" in sheet.upper() else 5.0
        if pct < threshold:
            warn(f"[{sheet}] {total_nulls} nulos en semanas ({pct:.1f}%) — aceptable")
        else:
            err(f"[{sheet}] {total_nulls} nulos en semanas ({pct:.1f}%) — revisar datos")
        worst = null_counts[null_counts > 0].sort_values(ascending=False)
        for col, n in worst.head(3).items():
            print(f"       {col}: {n} nulos")

def check_numeric(df, week_cols, sheet):
    for col in week_cols:
        if col not in df.columns: continue
        non_num = df[col].dropna().apply(lambda x: not isinstance(x, (int, float, np.integer, np.floating)))
        if non_num.any():
            err(f"[{sheet}] {col} tiene {non_num.sum()} valores no numéricos")
            return
    ok(f"[{sheet}] Todos los valores de semana son numéricos")

def check_duplicates(df, keys, sheet):
    dups = df.duplicated(subset=keys, keep=False).sum()
    if dups == 0:
        ok(f"[{sheet}] Sin duplicados en {keys}")
        return
    dup_rows = df[df.duplicated(subset=keys, keep=False)]
    conflicting = sum(1 for _, g in dup_rows.groupby(keys) if len(g.drop_duplicates()) > 1)
    sample = dup_rows[keys].head(3)
    if conflicting == 0:
        warn(f"[{sheet}] {dups} filas duplicadas identicas en {keys} — resueltas automaticamente por data_loader")
    else:
        err(f"[{sheet}] {conflicting} grupos con duplicados conflictivos en {keys}")
    print(f"       Ejemplos:\n{sample.to_string(index=False)}")

def check_categories(df, col, valid_set, sheet):
    if col not in df.columns: return
    found    = set(df[col].dropna().unique())
    unknown  = found - valid_set
    if unknown:
        warn(f"[{sheet}] {col} tiene valores desconocidos: {unknown}")
    else:
        ok(f"[{sheet}] {col} — valores válidos: {sorted(found)}")

def check_monotonicity(df, week_cols, sheet, sample_n=5):
    """Detecta zonas donde todos los valores son idénticos (posible error de datos)."""
    flat = df[week_cols].std(axis=1) == 0
    flat_count = flat.sum()
    if flat_count == 0:
        ok(f"[{sheet}] Sin filas con valores completamente planos")
    else:
        pct = flat_count / len(df) * 100
        warn(f"[{sheet}] {flat_count} filas ({pct:.1f}%) con valor idéntico en las 9 semanas")

def check_outliers(df, week_cols, sheet):
    """Detecta valores extremos que podrían ser errores."""
    l0 = df[week_cols[-1]].dropna()
    q1, q99 = l0.quantile(0.01), l0.quantile(0.99)
    extreme = ((l0 < q1 * 0.01) | (l0 > q99 * 100)).sum()
    if extreme == 0:
        ok(f"[{sheet}] Sin valores extremos detectados")
    else:
        warn(f"[{sheet}] {extreme} valores extremos en L0W (posibles errores)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.xlsx")
    args = parser.parse_args()

    path = Path(args.data)
    if not path.exists():
        print(f"❌ Archivo no encontrado: {path}")
        sys.exit(1)

    print("=" * 55)
    print(f"  VALIDACIÓN DE DATOS — {path.name}")
    print("=" * 55)

    # ── Cargar hojas ──────────────────────────────────────────
    try:
        xl = pd.ExcelFile(path)
    except Exception as e:
        print(f"{ERR} No se pudo abrir el Excel: {e}")
        sys.exit(1)

    ok(f"Excel abierto — hojas: {xl.sheet_names}")

    required_sheets = {"RAW_INPUT_METRICS", "RAW_ORDERS"}
    missing_sheets  = required_sheets - set(xl.sheet_names)
    if missing_sheets:
        err(f"Hojas faltantes: {missing_sheets}")
        sys.exit(1)

    df_m = pd.read_excel(xl, sheet_name="RAW_INPUT_METRICS")
    df_o = pd.read_excel(xl, sheet_name="RAW_ORDERS")
    df_m.columns = df_m.columns.str.strip()
    df_o.columns = df_o.columns.str.strip()

    # ── Tamaño ────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print("  TAMAÑO")
    print(f"{'─'*55}")
    ok(f"RAW_INPUT_METRICS: {len(df_m):,} filas × {len(df_m.columns)} columnas")
    ok(f"RAW_ORDERS:        {len(df_o):,} filas × {len(df_o.columns)} columnas")

    # ── Columnas ──────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print("  COLUMNAS")
    print(f"{'─'*55}")
    check_cols(df_m, EXPECTED_METRICS_COLS, "RAW_INPUT_METRICS")
    check_cols(df_o, EXPECTED_ORDERS_COLS,  "RAW_ORDERS")

    # ── Tipos numéricos ───────────────────────────────────────
    print(f"\n{'─'*55}")
    print("  TIPOS DE DATOS")
    print(f"{'─'*55}")
    check_numeric(df_m, [c for c in WEEK_COLS_M if c in df_m.columns], "RAW_INPUT_METRICS")
    check_numeric(df_o, [c for c in WEEK_COLS_O if c in df_o.columns], "RAW_ORDERS")

    # ── Nulos ─────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print("  VALORES NULOS")
    print(f"{'─'*55}")
    check_nulls(df_m, [c for c in WEEK_COLS_M if c in df_m.columns], "RAW_INPUT_METRICS")
    check_nulls(df_o, [c for c in WEEK_COLS_O if c in df_o.columns], "RAW_ORDERS")

    # ── Duplicados ────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print("  DUPLICADOS")
    print(f"{'─'*55}")
    check_duplicates(df_m, ["COUNTRY","CITY","ZONE","METRIC"], "RAW_INPUT_METRICS")
    check_duplicates(df_o, ["COUNTRY","CITY","ZONE","METRIC"], "RAW_ORDERS")

    # ── Categorías ────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print("  VALORES CATEGÓRICOS")
    print(f"{'─'*55}")
    check_categories(df_m, "COUNTRY",           VALID_COUNTRIES,  "RAW_INPUT_METRICS")
    check_categories(df_m, "ZONE_TYPE",          VALID_ZONE_TYPES, "RAW_INPUT_METRICS")
    check_categories(df_m, "ZONE_PRIORITIZATION",VALID_PRIORITIES, "RAW_INPUT_METRICS")
    check_categories(df_o, "COUNTRY",            VALID_COUNTRIES,  "RAW_ORDERS")

    # ── Cobertura ─────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print("  COBERTURA")
    print(f"{'─'*55}")
    m_zones = set(zip(df_m["COUNTRY"], df_m["CITY"], df_m["ZONE"]))
    o_zones = set(zip(df_o["COUNTRY"], df_o["CITY"], df_o["ZONE"]))
    in_m_not_o = m_zones - o_zones
    in_o_not_m = o_zones - m_zones
    ok(f"Zonas únicas en métricas:  {len(m_zones):,}")
    ok(f"Zonas únicas en órdenes:   {len(o_zones):,}")
    if in_m_not_o:
        warn(f"{len(in_m_not_o)} zonas en métricas sin datos de órdenes")
    if in_o_not_m:
        warn(f"{len(in_o_not_m)} zonas en órdenes sin datos de métricas")
    ok(f"Zonas en ambas tablas:     {len(m_zones & o_zones):,}")

    # ── Métricas disponibles ──────────────────────────────────
    print(f"\n{'─'*55}")
    print("  MÉTRICAS DISPONIBLES")
    print(f"{'─'*55}")
    for metric in sorted(df_m["METRIC"].unique()):
        count = len(df_m[df_m["METRIC"] == metric])
        ok(f"{metric} ({count:,} zonas)")

    # ── Calidad de datos ──────────────────────────────────────
    print(f"\n{'─'*55}")
    print("  CALIDAD DE DATOS")
    print(f"{'─'*55}")
    check_monotonicity(df_m, [c for c in WEEK_COLS_M if c in df_m.columns], "RAW_INPUT_METRICS")
    check_monotonicity(df_o, [c for c in WEEK_COLS_O if c in df_o.columns], "RAW_ORDERS")
    check_outliers(df_m, [c for c in WEEK_COLS_M if c in df_m.columns], "RAW_INPUT_METRICS")
    check_outliers(df_o, [c for c in WEEK_COLS_O if c in df_o.columns], "RAW_ORDERS")

    # ── Resumen final ─────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  RESUMEN")
    print(f"{'='*55}")
    if not issues and not warnings:
        print("  ✅ Todo perfecto — datos listos para el sistema")
    else:
        if issues:
            print(f"  ❌ {len(issues)} error(es) crítico(s) — requieren corrección:")
            for i in issues:
                print(f"     • {i}")
        if warnings:
            print(f"  ⚠️  {len(warnings)} advertencia(s) — revisar si es posible:")
            for w in warnings:
                print(f"     • {w}")
    print(f"{'='*55}\n")
    sys.exit(1 if issues else 0)

if __name__ == "__main__":
    main()