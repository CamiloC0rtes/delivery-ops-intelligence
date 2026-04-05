"""
app.py — Rappi Ops Intelligence · FastAPI backend
Run: python app.py
"""

import os, json, uuid, logging
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import pandas as pd
from openai import OpenAI

from data_loader import load_excel
from query_engine import execute_query
from insights import run_all, Insight

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── State global ──────────────────────────────────────────────────────────────
DATA: dict = {}
SESSIONS: dict[str, dict] = {}   # session_id → {history, context}

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
DATA_FILE  = os.getenv("DATA_FILE", "data.xlsx")
MODEL      = os.getenv("OPENAI_MODEL", "gpt-4o")
PORT       = int(os.getenv("PORT", 8000))

# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not OPENAI_KEY:
        raise RuntimeError("Falta OPENAI_API_KEY en variables de entorno")
    if not Path(DATA_FILE).exists():
        raise RuntimeError(f"No se encontró el archivo de datos: {DATA_FILE}")

    log.info("Cargando datos desde %s …", DATA_FILE)
    DATA.update(load_excel(DATA_FILE))
    DATA["client"] = OpenAI(api_key=OPENAI_KEY)
    DATA["insights"] = run_all(DATA["metrics_long"], DATA["orders_long"])
    log.info("✅ Datos cargados — métricas: %d | órdenes: %d | insights: %d",
             len(DATA["metrics_long"]), len(DATA["orders_long"]), len(DATA["insights"]))
    yield
    log.info("Shutting down.")

app = FastAPI(title="Rappi Ops Intelligence", lifespan=lifespan)

# ── Prompts ───────────────────────────────────────────────────────────────────
EXTRACTION_PROMPT = """Eres un extractor de entidades para análisis de datos Rappi.
Devuelve SOLO un JSON sin markdown. Nada más.

Países: CO, PE, AR, MX, BR, CL, EC, UY, CR
Métricas exactas: Orders, Restaurants SST > SS CVR, Retail SST > SS CVR,
Gross Profit UE, Perfect Orders, Turbo Adoption, Pro Adoption (Last Week Status),
Lead Penetration, Restaurants Markdowns / GMV,
% Restaurants Sessions With Optimal Assortment,
Non-Pro PTC > OP, % PRO Users Who Breakeven, MLTV Top Verticals Adoption

REGLA CRITICA — CIUDAD vs ZONA:
CIUDADES (usar campo city): Bogota, Medellin, Cali, Barranquilla, Buenos Aires,
Lima, Ciudad De Mexico, Guadalajara, Santiago, Sao Paulo, Quito, Montevideo.
ZONAS (usar campo zone): barrios/sectores dentro de una ciudad como Chapinero,
Usaquen, Colina, Belgrano, Miraflores, Polanco, Palermo, etc.
Ejemplos: "Chapinero" -> zone="Chapinero", city=null
"Bogota" -> city="Bogota", zone=null
"Chapinero en Bogota" -> city="Bogota", zone="Chapinero"

REGLAS DE RANKING:
CAIDAS/BAJADAS -> sort_order="asc", concept=null (NUNCA concept="bajo performance")
CRECIMIENTO/SUBIDAS -> sort_order="desc", concept=null
conceptos solo si el usuario los dice literalmente: "zonas problematicas",
"alto crecimiento", "bajo performance", "deterioro sostenido", "mejora sostenida"

Si el usuario pregunta por "alto X pero bajo Y" o "X alto y Y bajo" o "tienen X pero no Y":
  intent="multivariable", metric_high=metrica alta, metric_low=metrica baja, metric=null
  IMPORTANTE: multivariable SOLO cuando hay DOS metricas en conflicto. "zonas que mas crecen" = ranking, NO multivariable.
Si pregunta por correlacion entre metricas: intent="correlation"
"crecimiento en ordenes", "mas crecen", "mayor crecimiento" = intent:"ranking", metric:"Orders", sort_order:"desc"
metric SIEMPRE string o null, NUNCA array.

is_new_topic=true si es pregunta nueva. false si es seguimiento ("y en X?", "esa zona?")

JSON de salida:
{"intent":"ranking|trend|comparison|anomaly|summary|filter|multivariable|correlation","metric":null,"metric_high":null,"metric_low":null,"city":null,"country":null,"zone":null,"concept":null,"top_n":null,"sort_order":"desc","zone_type":null,"priority":null,"is_new_topic":true}
"""

ANALYSIS_PROMPT = """Eres un analista senior de operaciones de Rappi.
Recibirás una pregunta y datos ya filtrados. Analiza y responde de forma clara.

REGLAS:
- Usa los datos proporcionados, no inventes cifras
- Si no hay datos, explica qué filtros se aplicaron y sugiere alternativas
- Sé específico con zonas, ciudades y números
- NO incluyas JSON en tu respuesta
- Responde siempre en español
- Cuando el usuario pida evolución, tendencia o gráfico de una zona específica, responde con los datos disponibles Y agrega al final: 'Para ver el gráfico completo de 8 semanas, ve a la pestaña Analisis, selecciona el país y ciudad correspondiente.'

ESTRUCTURA OBLIGATORIA (usa exactamente estos encabezados):
**Resultado**: [zonas y números concretos del dataset]
**Contexto**: [por qué importa, tendencia, comparación]
**Recomendación**: [acción específica ejecutable]

Glosario de columnas:
- PCT_CHANGE_WOW: cambio % vs semana anterior (negativo = caída)
- L0W_VALUE: valor más reciente
- IS_DECLINING_3W: True = 3+ semanas bajando consecutivamente
- ZSCORE_VS_CITY: desviaciones vs promedio de su ciudad (< -1.5 = bajo performance)
- TREND_SLOPE: pendiente de tendencia (negativo = deterioro)
"""

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_session(session_id: str) -> dict:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {"history": [], "context": {}}
    return SESSIONS[session_id]

def extract_entities(message: str, prev_questions: list[str], client: OpenAI) -> dict:
    prev = ""
    if prev_questions:
        prev = "Preguntas anteriores:\n" + "\n".join(prev_questions[-3:]) + "\n\n"
    try:
        resp = client.chat.completions.create(
            model=MODEL, max_tokens=150, temperature=0,
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user",   "content": f"{prev}Mensaje: {message}"},
            ],
        )
        raw = resp.choices[0].message.content.strip().replace("```json","").replace("```","").strip()
        return json.loads(raw)
    except Exception as e:
        log.warning("Entity extraction failed: %s", e)
        return {"intent":"summary","metric":None,"city":None,"country":None,
                "concept":None,"top_n":None,"sort_order":"desc","zone_type":None,"priority":None}

def df_to_text(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "Sin resultados para los filtros aplicados."
    preferred = ["COUNTRY","CITY","ZONE","METRIC","L0W_VALUE","L1W_VALUE",
             "PCT_CHANGE_WOW","TREND_SLOPE","IS_DECLINING_3W","ZSCORE_VS_CITY",
             "ZONE_TYPE","ZONE_PRIORITIZATION",
             "METRIC_A","METRIC_B","CORRELATION","STRENGTH","DIRECTION",
             "Lead Penetration","Perfect Orders","n_zonas","pct_declining"]
    cols = [c for c in preferred if c in df.columns]
    if not cols:
        cols = list(df.columns)
    sample = df[cols].head(max_rows).copy()
    for col in sample.select_dtypes(include="float").columns:
        sample[col] = sample[col].round(4)
    out = sample.to_string(index=False)
    if len(df) > max_rows:
        out += f"\n... ({len(df)-max_rows} registros adicionales)"
    return out

def generate_answer(message: str, data_text: str, context: dict,
                    bot_history: list, client: OpenAI) -> str:
    augmented = (
        f"Pregunta: {message}\n\n"
        f"Filtros aplicados: {json.dumps(context, ensure_ascii=False)}\n\n"
        f"Datos:\n{data_text}"
    )
    messages = ([{"role":"system","content":ANALYSIS_PROMPT}]
                + bot_history[-4:]
                + [{"role":"user","content":augmented}])
    try:
        resp = client.chat.completions.create(model=MODEL, max_tokens=900, messages=messages)
        return resp.choices[0].message.content
    except Exception as e:
        log.error("Answer generation failed: %s", e)
        raise HTTPException(status_code=502, detail=f"Error OpenAI: {e}")

def insight_to_dict(ins: Insight) -> dict:
    return {
        "tipo": ins.tipo, "severity": ins.severity,
        "country": ins.country, "city": ins.city,
        "zone": ins.zone, "metric": ins.metric,
        "finding": ins.finding, "impact": ins.impact,
        "recommendation": ins.recommendation,
        "evidence_value": round(float(ins.evidence_value), 4),
    }

# ── API Routes ────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: str = ""

@app.post("/api/chat")
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    session    = get_session(session_id)
    client     = DATA["client"]

    prev_questions = [m["content"] for m in session["history"] if m["role"] == "user"]
    entities = extract_entities(req.message, prev_questions, client)
    log.info("[ENTITIES] msg='%s' → %s", req.message[:60], json.dumps(entities, ensure_ascii=False))

    # Actualizar contexto acumulado con lógica smart
    is_new_topic = entities.pop("is_new_topic", False)

    # Claves que SIEMPRE se resetean si la nueva pregunta no las menciona explícitamente
    VOLATILE_KEYS = ("zone", "city", "concept", "zone_type", "priority")
    # Claves que se acumulan entre turnos (país puede mantenerse si el usuario lo estableció)
    STICKY_KEYS   = ("country",)

    if is_new_topic:
        # Nuevo tema: limpiar todo el contexto volátil, conservar solo lo que
        # viene explícito en la nueva pregunta
        for key in VOLATILE_KEYS:
            session["context"].pop(key, None)
        # También limpiar métrica y top_n — la nueva pregunta los redefine
        for key in ("metric", "metric_high", "metric_low", "top_n", "sort_order", "country"):
            session["context"].pop(key, None)
    else:
        # Follow-up: limpiar solo las claves volátiles que la nueva pregunta NO menciona
        for key in VOLATILE_KEYS:
            if entities.get(key) is None:
                session["context"].pop(key, None)
        # Limpiar metric_high/metric_low si el nuevo intent no es multivariable
        if entities.get("intent") != "multivariable":
            session["context"].pop("metric_high", None)
            session["context"].pop("metric_low", None)

    # Aplicar las entidades nuevas (sobreescribe lo que quedó del contexto)
    for k, v in entities.items():
        if v is not None:
            session["context"][k] = v

    # Query sobre datos reales
    log.info("[CONTEXT] after merge: %s", json.dumps(session["context"], ensure_ascii=False))
    try:
        ctx_intent = session["context"].get("intent", "")
        if ctx_intent == "multivariable":
            from query_engine import execute_multivariable
            result_df, query_desc = execute_multivariable(session["context"], DATA["metrics_long"])
        elif ctx_intent == "correlation":
            from query_engine import execute_correlation
            result_df, query_desc = execute_correlation(DATA["metrics_long"], country=session["context"].get("country",""), min_r=0.4)
        else:
            result_df, query_desc = execute_query(
                session["context"], DATA["metrics_long"], DATA["orders_long"]
            )
        result_df = result_df.reset_index(drop=True)
        log.info("[QUERY] %s → %d filas", query_desc, len(result_df))
        if not result_df.empty:
            preview_cols = [c for c in ["ZONE","CITY","ZONE_TYPE","PCT_CHANGE_WOW"] if c in result_df.columns]
            preview = result_df[preview_cols].head(3).to_string(index=False)
            log.info("[DATA PREVIEW]\n%s", preview)
        data_text  = df_to_text(result_df)
        chart_data = build_chart_data(result_df, session["context"])
    except Exception as e:
        log.error("Query failed: %s", e)
        result_df  = pd.DataFrame()
        data_text  = f"Error al consultar datos: {e}"
        chart_data = None

    bot_turns = [m for m in session["history"] if m["role"] == "assistant"]
    answer    = generate_answer(req.message, data_text, session["context"], bot_turns, client)

    session["history"].append({"role":"user",      "content": req.message})
    session["history"].append({"role":"assistant",  "content": answer})
    if len(session["history"]) > 24:
        session["history"] = session["history"][-24:]

    log.info("[%s] Q: %s | rows: %d", session_id[:8], req.message[:60], len(result_df))


    import math
    def _cj(o):
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)): return None
        if isinstance(o, dict): return {k: _cj(v) for k,v in o.items()}
        if isinstance(o, list): return [_cj(v) for v in o]
        return o
    return _cj({
        "session_id": session_id,
        "answer":     answer,
        "chart":      chart_data,
        "rows_found": len(result_df),
        "context":    session["context"],
    })

@app.delete("/api/session/{session_id}")
async def reset_session(session_id: str):
    SESSIONS.pop(session_id, None)
    return {"status": "reset"}

@app.get("/api/insights")
async def get_insights(country: str = None, tipo: str = None,
                       severity: str = None, limit: int = 20):
    insights = DATA["insights"]
    if country:
        insights = [i for i in insights if i.country == country.upper()]
    if tipo:
        insights = [i for i in insights if i.tipo == tipo]
        return {"insights": [insight_to_dict(i) for i in insights[:limit]],
                "total": len(insights)}
    if severity:
        insights = [i for i in insights if i.severity == severity]
        return {"insights": [insight_to_dict(i) for i in insights[:limit]],
                "total": len(insights)}
    # Sin filtros: devolver mix balanceado por tipo para que todos los tabs tengan datos
    buckets = {}
    for i in insights:
        buckets.setdefault(i.tipo, []).append(i)
    mixed = []
    per_tipo = max(1, limit // len(buckets)) if buckets else limit
    for tipo_key in ["anomaly", "trend", "benchmark", "opportunity"]:
        mixed.extend(buckets.get(tipo_key, [])[:per_tipo])
    # Rellenar con lo que sobre hasta el limit
    seen = set(id(i) for i in mixed)
    for i in insights:
        if len(mixed) >= limit:
            break
        if id(i) not in seen:
            mixed.append(i)
            seen.add(id(i))
    return {"insights": [insight_to_dict(i) for i in mixed],
            "total": len(insights)}

@app.get("/api/insights/summary")
async def insights_summary():
    ins = DATA["insights"]
    by_type     = {}
    by_severity = {}
    by_country  = {}
    for i in ins:
        by_type[i.tipo]         = by_type.get(i.tipo, 0) + 1
        by_severity[i.severity] = by_severity.get(i.severity, 0) + 1
        by_country[i.country]   = by_country.get(i.country, 0) + 1
    return {"total": len(ins), "by_type": by_type,
            "by_severity": by_severity, "by_country": by_country}

@app.get("/api/filters")
async def get_filters():
    m = DATA["metrics_long"]
    o = DATA["orders_long"]
    return {
        "countries": sorted(m["COUNTRY"].unique().tolist()),
        "metrics":   sorted(m["METRIC"].unique().tolist()) + ["Orders"],
        "cities":    sorted(m["CITY"].unique().tolist()),
        "zone_types": ["Wealthy", "Non Wealthy"],
        "priorities": ["High Priority", "Prioritized", "Not Prioritized"],
    }

@app.get("/api/timeseries")
async def get_timeseries(country: str, zone: str, metric: str):
    """Devuelve serie de tiempo de 9 semanas para una zona+métrica."""
    raw = DATA["raw_metrics"] if metric != "Orders" else DATA["raw_orders"]
    week_cols = (["L8W_ROLL","L7W_ROLL","L6W_ROLL","L5W_ROLL","L4W_ROLL",
                  "L3W_ROLL","L2W_ROLL","L1W_ROLL","L0W_ROLL"]
                 if metric != "Orders"
                 else ["L8W","L7W","L6W","L5W","L4W","L3W","L2W","L1W","L0W"])

    mask = ((raw["COUNTRY"] == country) &
            (raw["ZONE"].str.lower() == zone.lower()))
    if metric != "Orders":
        mask &= (raw["METRIC"] == metric)

    row = raw[mask]
    if row.empty:
        raise HTTPException(404, "Serie no encontrada")

    row = row.iloc[0]
    labels = ["L8W","L7W","L6W","L5W","L4W","L3W","L2W","L1W","L0W"]
    values = [round(float(row[c]), 4) if pd.notna(row[c]) else None for c in week_cols]
    return {"zone": zone, "metric": metric, "labels": labels, "values": values}

@app.get("/api/ranking")
async def get_ranking(country: str = None, metric: str = "Orders",
                      sort: str = "desc", limit: int = 10):
    """Top/bottom zonas por valor L0W."""
    if metric == "Orders":
        df = DATA["orders_long"].copy()
    else:
        df = DATA["metrics_long"].copy()
        df = df[df["METRIC"] == metric]

    if country:
        df = df[df["COUNTRY"] == country]

    df = df.dropna(subset=["L0W_VALUE"])
    df = (df.nlargest(limit, "L0W_VALUE") if sort == "desc"
          else df.nsmallest(limit, "L0W_VALUE"))

    return {"data": df[["COUNTRY","CITY","ZONE","METRIC",
                         "L0W_VALUE","PCT_CHANGE_WOW"]].to_dict("records")}

# ── Chart builder ─────────────────────────────────────────────────────────────

def build_chart_data(df: pd.DataFrame, context: dict) -> dict | None:
    """Decide qué gráfico generar basándose en el resultado y contexto."""
    if df.empty:
        return None

    intent  = context.get("intent", "")
    concept = context.get("concept", "") or ""
    metric  = context.get("metric", "") or ""

    # Comparison chart (Wealthy vs Non Wealthy)
    if intent == "comparison" and "ZONE_TYPE" in df.columns and "L0W_VALUE" in df.columns:
        labels = df["ZONE_TYPE"].tolist()
        values = df["L0W_VALUE"].round(4).tolist()
        colors = ["#00b4d8" if "non" in str(l).lower() else "#ff441f" for l in labels]
        return {"type":"bar","title":"Comparacion · "+metric,"labels":labels,
                "datasets":[{"label":metric,"data":values,"colors":colors}],
                "changes":df["PCT_CHANGE_WOW"].round(2).tolist() if "PCT_CHANGE_WOW" in df.columns else []}

    # Ranking chart — barras horizontales
    if intent == "ranking" or context.get("top_n") or "mejor" in concept or "peor" in concept:
        if "L0W_VALUE" not in df.columns:
            return None
        sort_order = context.get("sort_order", "desc")
        if "PCT_CHANGE_WOW" in df.columns:
            df_sorted = df.dropna(subset=["PCT_CHANGE_WOW"])
            if sort_order == "asc":
                top = df_sorted.nsmallest(10, "PCT_CHANGE_WOW")
            else:
                top = df_sorted.nlargest(10, "PCT_CHANGE_WOW")
        else:
            top = df.dropna(subset=["L0W_VALUE"]).head(10)
        def short_metric(m):
            words = str(m).split()
            return " ".join(words[:3]) if len(words) > 3 else str(m)
        # Handle comparison results (ZONE_TYPE) vs normal results (ZONE)
        if "ZONE_TYPE" in top.columns and "ZONE" not in top.columns:
            labels = (top["ZONE_TYPE"] + ("  ·  " + top["METRIC"].apply(short_metric) if "METRIC" in top.columns else "")).tolist()
        elif "ZONE" in top.columns and "METRIC" in top.columns and top["METRIC"].nunique() > 1:
            labels = (top["ZONE"] + "  ·  " + top["METRIC"].apply(short_metric)).tolist()
        elif "ZONE" in top.columns:
            labels = (top["ZONE"] + "  (" + top["CITY"].fillna("") + ")").tolist() if "CITY" in top.columns else top["ZONE"].tolist()
        else:
            labels = top.iloc[:,0].astype(str).tolist()
        if "PCT_CHANGE_WOW" in top.columns and top["PCT_CHANGE_WOW"].notna().any():
            values = top["PCT_CHANGE_WOW"].round(2).tolist()
            colors = ["#dc2626" if v < 0 else "#16a34a" for v in values]
            changes = top["L0W_VALUE"].round(0).tolist()
        else:
            values = top["L0W_VALUE"].round(4).tolist()
            colors = ["#1a1a1a"] * len(values)
            changes = []
        return {
            "type": "bar",
            "title": f"Ranking · {metric or 'metrica'}",
            "labels": labels,
            "datasets": [{"label": "Cambio WoW", "data": values, "colors": colors}],
            "changes": changes,
        }

    # Anomaly / trend chart — scatter de PCT_CHANGE_WOW
    if ("anomal" in intent or "problemáticas" in concept or
            "crecimiento" in concept or "deterioro" in concept):
        if "PCT_CHANGE_WOW" not in df.columns:
            return None
        top = df.dropna(subset=["PCT_CHANGE_WOW"]).head(15)
        def short_metric(m):
            words = str(m).split()
            return " ".join(words[:3]) if len(words) > 3 else str(m)
        if "ZONE_TYPE" in top.columns and "ZONE" not in top.columns:
            labels = (top["ZONE_TYPE"] + "  ·  " + top["METRIC"].apply(short_metric)).tolist() if "METRIC" in top.columns else top["ZONE_TYPE"].tolist()
        elif "METRIC" in top.columns and top["METRIC"].nunique() > 1 and "ZONE" in top.columns:
            labels = (top["ZONE"] + "  ·  " + top["METRIC"].apply(short_metric)).tolist()
        elif "ZONE" in top.columns:
            labels = (top["ZONE"] + "  (" + top.get("CITY", pd.Series([""] * len(top))).fillna("") + ")").tolist()
        else:
            labels = top.iloc[:,0].astype(str).tolist()
        values  = top["PCT_CHANGE_WOW"].round(2).tolist()
        colors  = ["#e24b4a" if v < 0 else "#1D9E75" for v in values]
        return {
            "type": "anomaly",
            "title": f"Cambio WoW (%) · {context.get('country','todos')}",
            "labels": labels,
            "datasets": [{"label": "% cambio WoW", "data": values, "colors": colors}],
        }

    # Default: barras de L0W_VALUE si hay suficientes filas
    if len(df) >= 3 and "L0W_VALUE" in df.columns:
        top    = df.dropna(subset=["L0W_VALUE"]).head(10)
        def short_metric(m):
            words = str(m).split()
            return " ".join(words[:3]) if len(words) > 3 else str(m)
        if "ZONE_TYPE" in top.columns and "ZONE" not in top.columns:
            labels = top["ZONE_TYPE"].astype(str).tolist()
        elif "METRIC" in top.columns and top["METRIC"].nunique() > 1 and "ZONE" in top.columns:
            labels = (top["ZONE"] + "  ·  " + top["METRIC"].apply(short_metric)).tolist()
        elif "ZONE" in top.columns:
            labels = top["ZONE"].tolist()
        else:
            labels = top.iloc[:,0].astype(str).tolist()
        values = top["L0W_VALUE"].round(4).tolist()
        return {
            "type": "bar",
            "title": f"{metric or 'Valor'} por zona",
            "labels": labels,
            "datasets": [{"label": "Valor L0W", "data": values}],
        }
    return None


@app.get("/api/analysis/country-summary")
async def country_summary():
    """Resumen por pais para el heatmap."""
    m = DATA["metrics_long"]
    o = DATA["orders_long"]

    countries = []
    for country in sorted(m["COUNTRY"].unique()):
        m_c = m[m["COUNTRY"] == country]
        o_c = o[o["COUNTRY"] == country]

        avg_wow      = round(float(m_c["PCT_CHANGE_WOW"].dropna().mean()), 2)
        pct_decline  = round(float(m_c["IS_DECLINING_3W"].mean()) * 100, 1)
        n_zones      = int(m_c["ZONE"].nunique())
        n_anomalies  = int((m_c["PCT_CHANGE_WOW"].abs() > 10).sum())
        avg_orders   = round(float(o_c["L0W_VALUE"].dropna().mean()), 0) if not o_c.empty else 0

        countries.append({
            "country": country,
            "avg_wow": avg_wow,
            "pct_decline": pct_decline,
            "n_zones": n_zones,
            "n_anomalies": n_anomalies,
            "avg_orders": avg_orders,
        })
    return {"data": countries}


@app.get("/api/analysis/city-comparison")
async def city_comparison(country: str, metric: str = "Orders"):
    """Comparacion de zonas dentro de una ciudad."""
    if metric == "Orders":
        df = DATA["orders_long"].copy()
    else:
        df = DATA["metrics_long"].copy()
        df = df[df["METRIC"] == metric]

    df = df[df["COUNTRY"] == country.upper()]

    # Top ciudades por numero de zonas
    top_cities = (df.groupby("CITY")["ZONE"]
                  .nunique()
                  .sort_values(ascending=False)
                  .head(8)
                  .index.tolist())

    result = []
    for city in top_cities:
        city_df = df[df["CITY"] == city].dropna(subset=["L0W_VALUE"])
        if city_df.empty:
            continue
        zones = city_df.nlargest(6, "L0W_VALUE")
        result.append({
            "city": city,
            "zones": zones[["ZONE", "L0W_VALUE", "PCT_CHANGE_WOW"]].to_dict("records"),
            "avg": round(float(city_df["L0W_VALUE"].mean()), 2),
        })
    import math
    def clean(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return None
        return v
    safe = []
    for row in result:
        safe_row = {"city": row["city"], "avg": clean(row["avg"]), "zones": []}
        for z in row.get("zones", []):
            safe_row["zones"].append({k: clean(v) for k, v in z.items()})
        safe.append(safe_row)
    return {"data": safe, "metric": metric, "country": country}


@app.get("/api/analysis/timeseries-multi")
async def timeseries_multi(country: str, city: str, metric: str = "Orders", top_n: int = 5):
    """Series de tiempo de top N zonas de una ciudad."""
    if metric == "Orders":
        raw = DATA["raw_orders"]
        week_cols = ["L8W","L7W","L6W","L5W","L4W","L3W","L2W","L1W","L0W"]
    else:
        raw = DATA["raw_metrics"]
        week_cols = ["L8W_ROLL","L7W_ROLL","L6W_ROLL","L5W_ROLL",
                     "L4W_ROLL","L3W_ROLL","L2W_ROLL","L1W_ROLL","L0W_ROLL"]

    import unicodedata
    def norm(s):
        return unicodedata.normalize("NFD", str(s)).encode("ascii","ignore").decode().lower()

    mask = (raw["COUNTRY"] == country.upper()) & (raw["CITY"].apply(norm) == norm(city))
    if metric != "Orders":
        mask &= (raw["METRIC"] == metric)

    df = raw[mask].copy()
    if df.empty:
        raise HTTPException(404, f"No data for {city} / {metric}")

    # Top N zonas por valor L0W
    last_col = week_cols[-1]
    df = df.dropna(subset=[last_col]).nlargest(top_n, last_col)

    labels = ["L8W","L7W","L6W","L5W","L4W","L3W","L2W","L1W","L0W"]
    series = []
    for _, row in df.iterrows():
        values = [round(float(row[c]), 4) if pd.notna(row[c]) else None for c in week_cols]
        series.append({"zone": row["ZONE"], "values": values})

    import math
    def clean_float(v):
        if v is None: return None
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return None
        return v
    series = [{"zone": s["zone"], "values": [clean_float(v) for v in s["values"]]} for s in series]
    return {"labels": labels, "series": series, "city": city, "metric": metric}


@app.get("/api/correlations")
async def get_correlations(country: str = None, min_r: float = 0.5):
    """Matriz de correlaciones entre métricas."""
    import pandas as pd
    m = DATA["metrics_long"].copy()
    if country:
        m = m[m["COUNTRY"] == country.upper()]
    pivot = m.pivot_table(
        index=["COUNTRY","CITY","ZONE"],
        columns="METRIC",
        values="L0W_VALUE"
    )
    corr = pivot.corr(method="pearson").round(3)
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            r = float(corr.iloc[i,j])
            if abs(r) >= min_r and not pd.isna(r):
                pairs.append({
                    "metric_a": cols[i],
                    "metric_b": cols[j],
                    "r": r,
                    "strength": "strong" if abs(r) >= 0.7 else "moderate",
                    "direction": "positive" if r > 0 else "negative",
                })
    pairs.sort(key=lambda x: abs(x["r"]), reverse=True)
    return {"correlations": pairs, "country": country or "global", "n_pairs": len(pairs)}


@app.get("/api/multivariable")
async def multivariable_analysis(
    metric_high: str = "Lead Penetration",
    metric_low: str = "Perfect Orders",
    country: str = None,
    threshold_pct: float = 0.25,
    limit: int = 20,
):
    """Zonas con alto valor en metric_high y bajo en metric_low."""
    import pandas as pd
    m = DATA["metrics_long"].copy()
    if country:
        m = m[m["COUNTRY"] == country.upper()]

    pivot = m.pivot_table(
        index=["COUNTRY","CITY","ZONE"],
        columns="METRIC",
        values="L0W_VALUE"
    ).reset_index()

    if metric_high not in pivot.columns or metric_low not in pivot.columns:
        raise HTTPException(400, f"Métricas no disponibles: {metric_high}, {metric_low}")

    q_high = pivot[metric_high].quantile(1 - threshold_pct)
    q_low  = pivot[metric_low].quantile(threshold_pct)

    result = pivot[
        (pivot[metric_high] >= q_high) &
        (pivot[metric_low]  <= q_low)
    ].copy()

    result = result.sort_values(metric_high, ascending=False).head(limit)
    cols = ["COUNTRY","CITY","ZONE", metric_high, metric_low]
    cols = [c for c in cols if c in result.columns]
    result = result[cols].round(4)

    return {
        "zones": result.to_dict("records"),
        "metric_high": metric_high,
        "metric_low":  metric_low,
        "threshold": f"Top {int(threshold_pct*100)}% en {metric_high}, bottom {int(threshold_pct*100)}% en {metric_low}",
        "count": len(result),
    }



@app.get("/api/zone-detail")
async def zone_detail(zone: str, country: str):
    """Todas las metricas + timeseries de ordenes para una zona especifica."""
    import math, unicodedata

    def norm(s):
        return unicodedata.normalize("NFD", str(s)).encode("ascii","ignore").decode().lower()

    def clean(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return round(v, 4) if isinstance(v, float) else v

    m = DATA["metrics_long"]
    o = DATA["orders_long"]
    raw_o = DATA["raw_orders"]

    cc = country.upper()
    mask_m = (m["COUNTRY"] == cc) & (m["ZONE"].apply(norm).str.contains(norm(zone), na=False))
    mask_o = (o["COUNTRY"] == cc) & (o["ZONE"].apply(norm).str.contains(norm(zone), na=False))

    zone_m = m[mask_m]
    zone_o = o[mask_o]

    if zone_m.empty and zone_o.empty:
        raise HTTPException(404, f"Zona no encontrada: {zone}")

    actual_zone = zone_m["ZONE"].iloc[0] if not zone_m.empty else zone_o["ZONE"].iloc[0]
    city = zone_m["CITY"].iloc[0] if not zone_m.empty else zone_o["CITY"].iloc[0]
    zone_type = zone_m["ZONE_TYPE"].iloc[0] if "ZONE_TYPE" in zone_m.columns and not zone_m.empty else None
    priority = zone_m["ZONE_PRIORITIZATION"].iloc[0] if "ZONE_PRIORITIZATION" in zone_m.columns and not zone_m.empty else None

    metrics = []
    for _, row in zone_m.iterrows():
        metrics.append({
            "metric": row["METRIC"],
            "value": clean(row["L0W_VALUE"]),
            "pct_wow": clean(row["PCT_CHANGE_WOW"]),
            "zscore": clean(row["ZSCORE_VS_CITY"]),
            "declining": bool(row["IS_DECLINING_3W"]),
            "slope": clean(row["TREND_SLOPE"]),
        })

    # Orders timeseries
    week_cols = ["L8W","L7W","L6W","L5W","L4W","L3W","L2W","L1W","L0W"]
    raw_mask = (DATA["raw_orders"]["COUNTRY"] == cc) &                (DATA["raw_orders"]["ZONE"].apply(norm).str.contains(norm(zone), na=False))
    raw_row = DATA["raw_orders"][raw_mask]
    orders_ts = []
    if not raw_row.empty:
        r = raw_row.iloc[0]
        orders_ts = [clean(float(r[c])) if pd.notna(r[c]) else None for c in week_cols]

    orders_l0w = clean(zone_o["L0W_VALUE"].iloc[0]) if not zone_o.empty else None
    orders_wow = clean(zone_o["PCT_CHANGE_WOW"].iloc[0]) if not zone_o.empty else None

    return {
        "zone": actual_zone, "city": city, "country": cc,
        "zone_type": zone_type, "priority": priority,
        "metrics": metrics,
        "orders": {"l0w": orders_l0w, "pct_wow": orders_wow, "timeseries": orders_ts},
        "week_labels": ["L8W","L7W","L6W","L5W","L4W","L3W","L2W","L1W","L0W"],
    }


@app.get("/api/search-zones")
async def search_zones(q: str = "", limit: int = 10, city: str = "", country: str = ""):
    """Busqueda rapida de zonas por nombre, o todas las zonas de una ciudad."""
    import unicodedata
    def norm(s):
        return unicodedata.normalize("NFD", str(s)).encode("ascii","ignore").decode().lower()

    m = DATA["metrics_long"]
    o = DATA["orders_long"]

    if city:
        # Devolver todas las zonas de una ciudad especifica
        matched = m.copy()
        if country:
            matched = matched[matched["COUNTRY"] == country.upper()]
        matched = matched[matched["CITY"].apply(norm) == norm(city)]
    else:
        q_norm = norm(q)
        matched = m[m["ZONE"].apply(norm).str.contains(q_norm, na=False)].copy()
    matched = matched.groupby(["COUNTRY","CITY","ZONE"]).agg(
        L0W_VALUE=("L0W_VALUE","mean"),
        PCT_CHANGE_WOW=("PCT_CHANGE_WOW","mean"),
    ).reset_index()

    # Enrich with orders
    ord_agg = o.groupby(["COUNTRY","CITY","ZONE"])["L0W_VALUE"].sum().reset_index()
    ord_agg.columns = ["COUNTRY","CITY","ZONE","orders_l0w"]
    matched = matched.merge(ord_agg, on=["COUNTRY","CITY","ZONE"], how="left")

    import math
    def clean(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return None
        return round(v, 3) if isinstance(v, float) else v

    results = []
    for _, r in matched.head(limit).iterrows():
        results.append({
            "zone": r["ZONE"], "city": r["CITY"], "country": r["COUNTRY"],
            "avg_metric": clean(r["L0W_VALUE"]),
            "pct_wow": clean(r["PCT_CHANGE_WOW"]),
            "orders": clean(r.get("orders_l0w")),
        })
    return {"results": results, "total": len(matched)}


@app.get("/api/executive-report")
async def executive_report():
    """Genera reporte ejecutivo HTML con top insights de la semana."""
    import math
    from datetime import datetime

    ins = DATA["insights"]

    def clean(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return None
        return v

    # Top 3 criticos + 2 oportunidades
    critical = [i for i in ins if i.severity == "high"][:5]
    opps     = [i for i in ins if i.tipo == "opportunity"][:5]
    top_all  = critical + opps

    # Country summary
    m = DATA["metrics_long"]
    country_summary = []
    for cc in sorted(m["COUNTRY"].unique()):
        mc = m[m["COUNTRY"]==cc]
        country_summary.append({
            "country": cc,
            "avg_wow": round(float(mc["PCT_CHANGE_WOW"].dropna().mean()), 2),
            "pct_dec": round(float(mc["IS_DECLINING_3W"].mean())*100, 1),
            "n_zones": int(mc["ZONE"].nunique()),
        })

    today = datetime.utcnow().strftime("%d %b %Y")

    rows_critical = ""
    for i in top_all:
        color = "#dc2626" if i.severity=="high" else "#16a34a"
        badge = "CRITICO" if i.severity=="high" else "OPORTUNIDAD"
        rows_critical += f"""
        <tr>
          <td style="padding:12px 16px;border-bottom:1px solid #f0ede8;">
            <span style="background:{'#fef2f2' if i.severity=='high' else '#f0fdf4'};color:{color};font-size:9px;font-weight:700;padding:2px 6px;border-radius:3px;letter-spacing:.8px">{badge}</span>
            <div style="font-weight:600;font-size:13px;margin-top:4px;color:#1a1a1a">{i.zone} &mdash; {i.city}</div>
            <div style="font-size:11px;color:#888;margin-top:2px;font-family:monospace">{i.metric}</div>
          </td>
          <td style="padding:12px 16px;border-bottom:1px solid #f0ede8;font-size:12px;color:#333;line-height:1.5">{i.finding}</td>
          <td style="padding:12px 16px;border-bottom:1px solid #f0ede8;font-size:12px;color:#555;line-height:1.5">{i.recommendation}</td>
        </tr>"""

    rows_countries = ""
    for c in country_summary:
        color = "#16a34a" if c["avg_wow"] >= 0 else "#dc2626"
        arrow = "&#8679;" if c["avg_wow"] >= 0 else "&#8681;"
        rows_countries += f"""
        <tr>
          <td style="padding:10px 16px;border-bottom:1px solid #f0ede8;font-weight:600;color:#1a1a1a">{c['country']}</td>
          <td style="padding:10px 16px;border-bottom:1px solid #f0ede8;font-family:monospace;color:{color}">{arrow} {'+' if c['avg_wow']>=0 else ''}{c['avg_wow']}%</td>
          <td style="padding:10px 16px;border-bottom:1px solid #f0ede8;font-family:monospace;color:#888">{c['pct_dec']}%</td>
          <td style="padding:10px 16px;border-bottom:1px solid #f0ede8;font-family:monospace;color:#555">{c['n_zones']}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="es">
<head><meta charset="UTF-8">
<title>Rappi Ops — Reporte Ejecutivo {today}</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:-apple-system,BlinkMacSystemFont,'Inter',sans-serif;background:#f9f8f5;color:#1a1a1a;padding:40px}}
  .page{{max-width:960px;margin:0 auto;background:#fff;border-radius:8px;overflow:hidden;border:1px solid #e8e6e0}}
  .hdr{{background:#ff441f;padding:28px 36px;display:flex;justify-content:space-between;align-items:flex-end}}
  .hdr-title{{color:#fff;font-size:22px;font-weight:700;letter-spacing:-.3px}}
  .hdr-sub{{color:rgba(255,255,255,.75);font-size:13px;margin-top:4px}}
  .hdr-date{{color:rgba(255,255,255,.85);font-size:12px;font-family:monospace}}
  .section{{padding:28px 36px;border-bottom:1px solid #f0ede8}}
  .section-title{{font-size:11px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;color:#bbb;margin-bottom:16px}}
  table{{width:100%;border-collapse:collapse}}
  th{{padding:10px 16px;text-align:left;font-size:10px;font-weight:700;letter-spacing:.8px;text-transform:uppercase;color:#aaa;background:#fafaf8;border-bottom:2px solid #f0ede8}}
  .footer{{padding:20px 36px;font-size:11px;color:#aaa;font-family:monospace;display:flex;justify-content:space-between}}
</style>
</head>
<body>
<div class="page">
  <div class="hdr">
    <div>
      <div class="hdr-title">Rappi Ops Intelligence</div>
      <div class="hdr-sub">Reporte ejecutivo semanal</div>
    </div>
    <div class="hdr-date">{today}</div>
  </div>

  <div class="section">
    <div class="section-title">Hallazgos criticos + oportunidades</div>
    <table>
      <tr>
        <th style="width:200px">Zona</th>
        <th>Hallazgo</th>
        <th>Recomendacion</th>
      </tr>
      {rows_critical}
    </table>
  </div>

  <div class="section">
    <div class="section-title">Resumen por pais</div>
    <table>
      <tr>
        <th>Pais</th>
        <th>Cambio WoW promedio</th>
        <th>% zonas en deterioro</th>
        <th>Total zonas</th>
      </tr>
      {rows_countries}
    </table>
  </div>

  <div class="footer">
    <span>Generado por Rappi Ops Intelligence</span>
    <span>{len(ins)} insights analizados esta semana</span>
  </div>
</div>
</body>
</html>"""

    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html, headers={
        "Content-Disposition": f"attachment; filename=rappi-ops-report-{datetime.utcnow().strftime('%Y%m%d')}.html"
    })


@app.get("/api/cities")
async def get_cities(country: str):
    """Todas las ciudades de un pais, ordenadas alfabeticamente."""
    import math
    m = DATA["metrics_long"]
    o = DATA["orders_long"]
    cc = country.upper()
    cities_m = set(m[m["COUNTRY"]==cc]["CITY"].dropna().unique())
    cities_o = set(o[o["COUNTRY"]==cc]["CITY"].dropna().unique())
    all_cities = sorted(cities_m | cities_o)
    return {"cities": all_cities, "total": len(all_cities)}

# ── Static files + SPA ────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=False)