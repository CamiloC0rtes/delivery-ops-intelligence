"""
insights.py
Motor de detección automática de insights sobre los DataFrames procesados.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class Insight:
    tipo: str          # anomaly | trend | benchmark | opportunity
    severity: str      # high | medium | low
    country: str
    city: str
    zone: str
    metric: str
    finding: str
    impact: str
    recommendation: str
    evidence_value: float


def run_all(df_metrics: pd.DataFrame, df_orders: pd.DataFrame,
            anomaly_threshold: float = 10.0) -> list[Insight]:
    """Ejecuta todos los detectores y retorna lista priorizada."""
    all_insights = []
    for df in [df_metrics, df_orders]:
        all_insights.extend(_anomalies(df, anomaly_threshold))
        all_insights.extend(_sustained_trends(df))
        all_insights.extend(_benchmarking(df))
        all_insights.extend(_opportunities(df))
    all_insights.extend(_correlations(df_metrics))

    severity_order = {"high": 0, "medium": 1, "low": 2}
    all_insights.sort(key=lambda x: (severity_order[x.severity], -abs(x.evidence_value)))
    return all_insights


def format_digest(insights: list[Insight], top_n: int = 8) -> str:
    """Genera texto resumido de los top insights para mostrar al inicio."""
    if not insights:
        return "✅ No se detectaron anomalías significativas en el período analizado."

    emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}
    lines = [f"📊 DIGEST AUTOMÁTICO — {len(insights)} insights detectados\n"]
    for i, ins in enumerate(insights[:top_n], 1):
        lines.append(
            f"{i}. {emoji[ins.severity]} [{ins.tipo.upper()}] "
            f"{ins.zone} ({ins.city}) · {ins.metric}\n"
            f"   {ins.finding}\n"
            f"   → {ins.recommendation}\n"
        )
    return "\n".join(lines)


# ── Detectores ────────────────────────────────────────────────────────────────

def _correlations(df: pd.DataFrame) -> list[Insight]:
    """Detecta pares de métricas con correlación fuerte (r >= 0.65)."""
    insights = []
    try:
        pivot = df.pivot_table(
            index=["COUNTRY","CITY","ZONE"],
            columns="METRIC", values="L0W_VALUE"
        )
        corr = pivot.corr(method="pearson")
        cols = list(corr.columns)
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                r = float(corr.iloc[i,j])
                if abs(r) < 0.65 or np.isnan(r):
                    continue
                direction = "positiva" if r > 0 else "negativa"
                insights.append(Insight(
                    tipo="benchmark",
                    severity="low",
                    country="GLOBAL", city="GLOBAL", zone="Correlacion",
                    metric=f"{cols[i][:25]} ↔ {cols[j][:25]}",
                    finding=(f"Correlacion {direction} fuerte (r={r:.2f}) entre "
                             f"'{cols[i]}' y '{cols[j]}'"),
                    impact=(f"Zonas con buen desempenio en {cols[i][:20]} "
                            f"{'tambien' if r > 0 else 'no'} tienden a tener "
                            f"buen desempenio en {cols[j][:20]}"),
                    recommendation=(f"Usar {cols[i][:20]} como indicador proxy de "
                                    f"{cols[j][:20]} en zonas sin datos completos."),
                    evidence_value=r,
                ))
    except Exception:
        pass
    return insights


def _anomalies(df: pd.DataFrame, threshold: float) -> list[Insight]:
    insights = []
    valid = df[df["PCT_CHANGE_WOW"].notna()]
    hits = valid[valid["PCT_CHANGE_WOW"].abs() > threshold]

    for _, r in hits.iterrows():
        direction = "caída" if r["PCT_CHANGE_WOW"] < 0 else "alza"
        if r["PCT_CHANGE_WOW"] < 0:
            severity = "high" if abs(r["PCT_CHANGE_WOW"]) > 20 else "medium"
        else:
            severity = "low"
        delta = r["L0W_VALUE"] - r["L1W_VALUE"] if r["L1W_VALUE"] else 0

        insights.append(Insight(
            tipo="anomaly", severity=severity,
            country=r.get("COUNTRY", ""), city=r.get("CITY", ""),
            zone=r.get("ZONE", ""), metric=r.get("METRIC", ""),
            finding=(f"{direction.capitalize()} de {abs(r['PCT_CHANGE_WOW']):.1f}% WoW "
                     f"en {r.get('METRIC','')} ({r['L1W_VALUE']:.3g} → {r['L0W_VALUE']:.3g})"),
            impact=(f"Variación de {delta:+.3g} unidades. "
                    f"Proyección acumulada 4 semanas: {delta*4:+.3g}"),
            recommendation=("Investigar causa raíz: oferta, demanda o evento externo. "
                            "Revisar logs de cancelaciones en la zona."
                            if r["PCT_CHANGE_WOW"] < 0
                            else "Identificar drivers y replicar en zonas similares."),
            evidence_value=r["PCT_CHANGE_WOW"],
        ))
    return insights


def _sustained_trends(df: pd.DataFrame) -> list[Insight]:
    insights = []
    for col, label in [
        ("IS_DECLINING_3W", "deterioro"),
        ("IS_IMPROVING_3W", "mejora"),
    ]:
        severity = "high" if label == "deterioro" else "low"
        if col not in df.columns:
            continue
        hits = df[df[col] == True].copy()
        for _, r in hits.iterrows():
            slope = r.get("TREND_SLOPE") or 0
            l0w   = r["L0W_VALUE"] or 0
            proj  = l0w + slope * 4
            insights.append(Insight(
                tipo="trend", severity=severity,
                country=r.get("COUNTRY", ""), city=r.get("CITY", ""),
                zone=r.get("ZONE", ""), metric=r.get("METRIC", ""),
                finding=(f"{label.capitalize()} continuo 3+ semanas en {r.get('METRIC','')}. "
                         f"Pendiente: {slope:+.4g}/semana | L0W: {l0w:.4g}"),
                impact=f"Proyección en 4 semanas: {proj:.4g} vs {l0w:.4g} actual",
                recommendation=("Activar plan de recuperación: revisar mix, incentivos, ETAs."
                                if label == "deterioro"
                                else "Zona en momentum positivo — considerar inversión adicional."),
                evidence_value=slope,
            ))
    return insights


def _benchmarking(df: pd.DataFrame) -> list[Insight]:
    insights = []
    if "ZSCORE_VS_CITY" not in df.columns:
        return insights
    hits = df[df["ZSCORE_VS_CITY"].notna() & (df["ZSCORE_VS_CITY"] < -1.5)]
    for _, r in hits.iterrows():
        city_avg = df[
            (df["CITY"] == r["CITY"]) & (df["METRIC"] == r.get("METRIC", ""))
        ]["L0W_VALUE"].mean()
        gap = city_avg - r["L0W_VALUE"] if city_avg and r["L0W_VALUE"] else 0

        insights.append(Insight(
            tipo="benchmark", severity="medium",
            country=r.get("COUNTRY", ""), city=r.get("CITY", ""),
            zone=r.get("ZONE", ""), metric=r.get("METRIC", ""),
            finding=(f"Performance {abs(r['ZSCORE_VS_CITY']):.1f}σ por debajo del promedio "
                     f"de {r.get('CITY','')} en {r.get('METRIC','')}. "
                     f"Valor: {r['L0W_VALUE']:.3g} vs avg {city_avg:.3g}"),
            impact=f"Brecha de {gap:.3g} unidades vs promedio ciudad.",
            recommendation="Analizar zonas de alto desempeño en la misma ciudad y replicar condiciones.",
            evidence_value=r["ZSCORE_VS_CITY"],
        ))
    return insights


def _opportunities(df: pd.DataFrame) -> list[Insight]:
    insights = []
    valid = df[df["PCT_CHANGE_WOW"].notna() & df["TREND_SLOPE"].notna()]
    hits = valid[(valid["PCT_CHANGE_WOW"] > 10) & (valid["TREND_SLOPE"] > 0)]
    for _, r in hits.iterrows():
        proj = (r["L0W_VALUE"] or 0) + r["TREND_SLOPE"] * 4
        insights.append(Insight(
            tipo="opportunity", severity="low",
            country=r.get("COUNTRY", ""), city=r.get("CITY", ""),
            zone=r.get("ZONE", ""), metric=r.get("METRIC", ""),
            finding=(f"Crecimiento {r['PCT_CHANGE_WOW']:.1f}% WoW con tendencia positiva "
                     f"(slope: +{r['TREND_SLOPE']:.4g}/sem)"),
            impact=f"Proyección 4 semanas: {proj:.3g} vs {r['L0W_VALUE']:.3g} actual.",
            recommendation="Zona en momentum — priorizar inversión: oferta, repartidores, campañas.",
            evidence_value=r["PCT_CHANGE_WOW"],
        ))
    return insights