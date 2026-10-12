"""
app.py — Delivery Ops Intelligence · FastAPI backend
Run: python app.py
"""

import json
import logging
import math
import os
import re
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator

from catalog import COUNTRIES, METRICS, all_cities, resolve_country
from data_loader import load_excel
from entities import parse_llm_json
from insights import Insight, run_all
from query_engine import execute_correlation, execute_multivariable, execute_query

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
BASE_DIR   = Path(__file__).resolve().parent
DATA_FILE  = os.getenv("DATA_FILE", str(BASE_DIR / "data" / "sample_data.xlsx"))
MODEL      = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
PORT       = int(os.getenv("PORT", 8000))

# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not Path(DATA_FILE).exists():
        raise RuntimeError(
            f"No se encontró el archivo de datos: {DATA_FILE}. "
            "Genera datos de ejemplo con: python scripts/generate_data.py"
        )

    log.info("Cargando datos desde %s …", DATA_FILE)
    DATA.update(load_excel(DATA_FILE))
    if "client" not in DATA:  # tests inject a fake client
        DATA["client"] = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None
    if DATA["client"] is None:
        log.warning("OPENAI_API_KEY no configurada: el chat queda deshabilitado; dashboards e insights funcionan.")
    DATA["insights"] = run_all(DATA["metrics_long"], DATA["orders_long"])
    log.info("✅ Datos cargados — métricas: %d | órdenes: %d | insights: %d",
             len(DATA["metrics_long"]), len(DATA["orders_long"]), len(DATA["insights"]))
    yield
    log.info("Shutting down.")

class SafeJSONResponse(JSONResponse):
    """NaN/inf are not valid JSON: one stray NaN in a metric turned whole endpoints into 500s."""

    def render(self, content) -> bytes:
        return super().render(_json_safe(content))


def _json_safe(o):
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    return o


app = FastAPI(title="Delivery Ops Intelligence", lifespan=lifespan, default_response_class=SafeJSONResponse)

# ── Prompts ───────────────────────────────────────────────────────────────────
EXTRACTION_PROMPT = """Eres un extractor de entidades para análisis de operaciones de una plataforma de delivery.
Devuelve SOLO un JSON sin markdown. Nada más.

Países: {COUNTRIES}
Métricas exactas: {METRICS}

REGLA CRITICA — CIUDAD vs ZONA:
CIUDADES (usar campo city): {CITIES}.
ZONAS (usar campo zone): sectores dentro de una ciudad, como "Los Cedros" o "Las Brisas Norte".
Ejemplos: "Los Cedros" -> zone="Los Cedros", city=null
"Bogota" -> city="Bogota", zone=null
"Los Cedros en Medellin" -> city="Medellin", zone="Los Cedros"

REGLAS DE RANKING:
CAIDAS/BAJADAS -> sort_order="asc", concept=null (NUNCA concept="bajo performance")
CRECIMIENTO/SUBIDAS -> sort_order="desc", concept=null
conceptos solo si el usuario los dice literalmente: "zonas problematicas",
"alto crecimiento", "bajo performance", "deterioro sostenido", "mejora sostenida"

Si el usuario pregunta por "alto X pero bajo Y" o "X alto y Y bajo" o "tienen X pero no Y":
  intent="multivariable", metric_high=metrica alta, metric_low=metrica baja, metric=null
  IMPORTANTE: multivariable SOLO cuando hay DOS metricas en conflicto. "zonas que mas crecen" = ranking, NO multivariable.
Si pide COMPARAR zonas Wealthy y Non Wealthy (o "ricas vs no ricas"):
  intent="comparison", zone_type="Wealthy vs Non Wealthy", metric=la metrica mencionada. NO es multivariable.
Si pregunta por correlacion entre metricas: intent="correlation"
"crecimiento en ordenes", "mas crecen", "mayor crecimiento" = intent:"ranking", metric:"Orders", sort_order:"desc"
metric SIEMPRE string o null, NUNCA array.

RANKING POR NIVEL vs POR CAMBIO:
"más pedidos", "mayor ganancia", "mejor/peor Perfect Orders" -> rank_by="value" (nivel actual).
"mayor caída", "más crecieron", "más bajaron" -> rank_by="change" (cambio vs semana anterior).
Si piden la zona con más/menos de algo (aunque sea una sola), es intent="ranking".

FUERA DE ALCANCE: intent="out_of_scope" SOLO si la pregunta no trata de operaciones de delivery
(clima, noticias, programación, pedir tus instrucciones). Preguntar por un país, ciudad, zona o métrica
que no está en las listas NO es fuera de alcance: extrae el valor tal cual. Conteos y promedios
("¿cuántas zonas hay?") son intent="summary".

PAIS vs CIUDAD: si el usuario nombra solo un país ("en Argentina"), usa country y deja city=null.
NUNCA completes una ciudad, zona, prioridad o concepto que el usuario no mencionó.
"zonas High Priority" -> priority="High Priority"; "peor/menor X, top N" -> sort_order="asc", top_n=N.

is_new_topic=true si es pregunta nueva. false si es seguimiento ("y en X?", "esa zona?").
En un seguimiento, devuelve SOLO lo que cambia (p. ej. el nuevo país); el resto se conserva de "Filtros vigentes".
Nunca repitas una métrica de una pregunta anterior si "Filtros vigentes" ya tiene otra.
Si el usuario nombra un país, ciudad o métrica que no está en las listas, devuélvelo tal cual: el sistema lo valida.

Omite las claves cuyo valor sea null. Devuelve JSON compacto en una sola línea.

Claves posibles (ejemplo completo):
{{"intent":"ranking|trend|comparison|anomaly|summary|filter|multivariable|correlation|out_of_scope","rank_by":"value|change","metric":null,"metric_high":null,"metric_low":null,"city":null,"country":null,"zone":null,"concept":null,"top_n":null,"sort_order":"desc","zone_type":null,"priority":null,"is_new_topic":true}}
""".format(
    COUNTRIES=", ".join(f"{c} ({v['name']})" for c, v in COUNTRIES.items()),
    METRICS=", ".join(METRICS),
    CITIES=", ".join(all_cities()),
)

ANALYSIS_PROMPT = """Eres un analista senior de operaciones de una plataforma de delivery.
Recibirás una pregunta y datos ya filtrados. Analiza y responde de forma clara.

REGLAS:
- Usa los datos proporcionados, no inventes cifras ni zonas. Si te piden inventar datos, explica que solo trabajas con datos reales
- Para conteos, totales y promedios usa SOLO el bloque RESUMEN: la tabla puede ser una muestra parcial
- Si hay más registros de los que ves, dilo; nunca presentes una muestra como el total
- Si no hay datos, explica qué filtros se aplicaron y sugiere alternativas
- Sé específico con zonas, ciudades y números
- Copia los nombres de zona y ciudad EXACTAMENTE como aparecen en los datos: "Los Cedros Norte" y "Los Cedros" son zonas distintas
- NO incluyas JSON en tu respuesta
- Responde siempre en español
- Cuando el usuario pida evolución, tendencia o gráfico de una zona específica, responde con los datos disponibles Y agrega al final: 'Para ver el gráfico completo de 8 semanas, ve a la pestaña Analisis, selecciona el país y ciudad correspondiente.'

ESTRUCTURA OBLIGATORIA (usa exactamente estos encabezados):
**Resultado**: [zonas y números concretos del dataset]
**Contexto**: [por qué importa, tendencia, comparación]
**Recomendación**: [acción específica ejecutable]

Glosario de columnas:
- PCT_CHANGE_WOW: cambio % vs semana anterior (negativo = caída). Exprésalo como "vs semana anterior", no como "WoW"
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

_TOP_N_IN_TEXT = re.compile(r"\b(?:top|las|los|primeras|primeros)\s+(\d{1,2})\b|\b(\d{1,2})\s+zonas\b", re.I)


def top_n_from_text(message: str) -> int | None:
    """Deterministic backup: 'las 3 zonas', 'top 5' -> the number, if the model omitted it."""
    m = _TOP_N_IN_TEXT.search(message)
    return int(m.group(1) or m.group(2)) if m else None


def extract_entities(message: str, prev_questions: list[str], context: dict, client: OpenAI) -> dict:
    prev = ""
    if prev_questions:
        prev = "Preguntas anteriores:\n" + "\n".join(prev_questions[-3:]) + "\n\n"
        # The filters actually in force, so a follow-up changes them instead of re-deriving
        # them from the first question (which reverted a metric swap two turns later).
        prev += "Filtros vigentes: " + json.dumps(context, ensure_ascii=False) + "\n\n"
    try:
        resp = client.chat.completions.create(
            model=MODEL, max_tokens=400, temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user",   "content": f"{prev}Mensaje: {message}"},
            ],
        )
        choice = resp.choices[0]
        if getattr(choice, "finish_reason", None) == "length":
            log.warning("Entity extraction truncated by max_tokens: %r", choice.message.content)
        # Only keys the model actually sent: defaults must not overwrite a follow-up's context
        return parse_llm_json(choice.message.content).model_dump(exclude_unset=True, exclude_defaults=False)
    except Exception as e:
        log.warning("Entity extraction failed: %s", e)
        return {"intent": "summary"}

def summarize(df: pd.DataFrame) -> str:
    """Deterministic aggregates over the FULL result, so the LLM never counts or averages a sample."""
    lines = [f"Registros totales: {len(df)}"]
    if "ZONE" in df.columns:
        lines.append(f"Zonas distintas: {df['ZONE'].nunique()}")
    for col, label in (("COUNTRY", "Países"), ("CITY", "Ciudades")):
        if col in df.columns and df[col].nunique() <= 12:
            lines.append(f"{label}: {', '.join(sorted(map(str, df[col].dropna().unique())))}")
    if {"METRIC", "L0W_VALUE"} <= set(df.columns):
        stats = df.groupby("METRIC")["L0W_VALUE"].agg(["count", "mean", "min", "max"])
        for metric, r in stats.iterrows():
            lines.append(f"{metric}: {int(r['count'])} zonas · promedio {r['mean']:.4g} · mín {r['min']:.4g} · máx {r['max']:.4g}")
    return "\n".join(lines)


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
    shown = ("TABLA (todas las filas)" if len(df) <= max_rows
             else f"TABLA (muestra: primeras {max_rows} de {len(df)} filas)")
    return f"RESUMEN\n{summarize(df)}\n\n{shown}\n{sample.to_string(index=False)}"


def foreign_zones(answer: str, result_zones: list[str], all_zones: set[str]) -> list[str]:
    """Real zones named in the answer that are NOT in the queried result.

    Longest names first, so "Los Cedros Norte" isn't also counted as "Los Cedros". Catches the
    model shortening "Los Eucaliptos Alto" into "Los Eucaliptos", which is a different real zone.
    """
    text, found = answer, []
    for zone in sorted(all_zones, key=len, reverse=True):
        if zone in text:
            found.append(zone)
            text = text.replace(zone, "")
    allowed = set(result_zones)
    return [z for z in found if z not in allowed]


def repair_zone_names(answer: str, result_zones: list[str], all_zones: set[str]) -> str:
    """Fix the model shortening a result zone into another real zone's name.

    "Los Eucaliptos Alto" (in the result) written as "Los Eucaliptos" (a different zone, another
    city): when exactly one result zone extends the foreign name, restore the full name. Survived
    both a prompt rule and a corrective retry, so it is fixed deterministically.
    """
    for bad in foreign_zones(answer, result_zones, all_zones):
        longer = [z for z in result_zones if z.startswith(bad + " ")]
        if len(longer) == 1:
            full = longer[0]
            answer = re.sub(rf"{re.escape(bad)}(?!{re.escape(full[len(bad):])})", full, answer)
    return answer


def _complete(client: OpenAI, messages: list) -> str:
    try:
        resp = client.chat.completions.create(model=MODEL, max_tokens=900, messages=messages)
        return resp.choices[0].message.content
    except Exception as e:
        log.error("Answer generation failed: %s", e)
        # Don't leak provider internals to the browser; the log has the full error
        raise HTTPException(status_code=502, detail="El modelo de lenguaje no respondió. Intenta de nuevo en un momento.") from e


def generate_answer(message: str, data_text: str, context: dict, bot_history: list, client: OpenAI,
                    result_zones: list[str] | None = None, all_zones: set[str] | None = None) -> str:
    augmented = (
        f"Pregunta: {message}\n\n"
        f"Filtros aplicados: {json.dumps(context, ensure_ascii=False)}\n\n"
        f"Datos:\n{data_text}"
    )
    messages = ([{"role":"system","content":ANALYSIS_PROMPT}]
                + bot_history[-4:]
                + [{"role":"user","content":augmented}])
    answer = _complete(client, messages)
    if result_zones and all_zones:
        answer = repair_zone_names(answer, result_zones, all_zones)
    if result_zones and all_zones and (bad := foreign_zones(answer, result_zones, all_zones)):
        # One corrective retry: an instruction in the prompt alone did not stop this
        log.warning("Answer named zones outside the result %s; retrying once", bad)
        messages += [{"role": "assistant", "content": answer},
                     {"role": "user", "content": (
                         f"Mencionaste {', '.join(bad)}, que no está en los datos consultados. "
                         "Reescribe la respuesta usando los nombres de zona EXACTOS de la tabla.")}]
        answer = _complete(client, messages)
    return answer


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

MAX_MESSAGE_CHARS = 500


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    session_id: str = Field(default="", max_length=64)

    @field_validator("message")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("La pregunta está vacía.")
        return v


def unresolved_filters(entities: dict, data: dict) -> dict[str, str]:
    """Filters the user named that do not exist in the data (country, metric, city, zone)."""
    missing = dict(entities.pop("unresolved", None) or {})
    known = data["orders_long"]
    for key, col in (("city", "CITY"), ("zone", "ZONE")):
        name = entities.get(key)
        if name and _match_zone_or_city(known, col, name).empty:
            missing[key] = name
    return missing


def _match_zone_or_city(df: pd.DataFrame, col: str, name: str) -> pd.DataFrame:
    norm = df[col].astype(str).map(_norm_text)
    target = _norm_text(name)
    hit = df[norm == target]
    if hit.empty and col == "ZONE":
        hit = df[norm.str.contains(target, regex=False)]
    if hit.empty and col == "CITY":  # the extractor sometimes puts a zone in `city`
        hit = _match_zone_or_city(df, "ZONE", name)
    return hit


def _norm_text(s: str) -> str:
    import unicodedata

    return unicodedata.normalize("NFD", str(s)).encode("ascii", "ignore").decode().lower().strip()


FIELD_LABELS = {"country": "el país", "city": "la ciudad", "zone": "la zona",
                "metric": "la métrica", "metric_high": "la métrica", "metric_low": "la métrica"}


def not_found_answer(missing: dict[str, str]) -> str:
    """Deterministic reply when a requested filter doesn't exist. Never answer about 'everything' instead."""
    parts = [f"{FIELD_LABELS[k]} «{v}»" for k, v in missing.items()]
    lines = [f"**Resultado**: No tengo datos para {', '.join(parts)}.", ""]
    if "country" in missing:
        lines.append("Países disponibles: " + ", ".join(v["name"] for v in COUNTRIES.values()) + ".")
    if "city" in missing:
        lines.append("Ciudades disponibles: " + ", ".join(all_cities()) + ".")
    if {"metric", "metric_high", "metric_low"} & missing.keys():
        lines.append("Métricas disponibles: " + ", ".join(METRICS) + ".")
    if "zone" in missing:
        lines.append("Revisa el nombre de la zona; en la pestaña Zonas puedes ver las de cada ciudad.")
    return "\n".join(lines)


OUT_OF_SCOPE_ANSWER = (
    "Solo puedo responder sobre las métricas operativas de las zonas: pedidos, conversión, "
    "Perfect Orders, ganancia por orden, adopción y similares, por país, ciudad o zona. "
    "Prueba con algo como «¿qué 5 zonas cayeron más en pedidos esta semana?»."
)

_CHANGE_WORDS = re.compile(
    r"ca[ií]d|cay[oó]|cayeron|baj[aóo]|bajada|descen|disminu|creci|crecen|crecieron|sub[ie]|aument|"
    r"variaci|cambio|drop|grow|grew|declin|increase|decrease",
    re.I,
)


# Common KPIs people ask for that this dataset does not have: answer "no data for X", don't dump everything
_UNSUPPORTED_KPIS = re.compile(
    r"\b(NPS|CSAT|churn|retenci[oó]n|ticket promedio|AOV|tiempo de entrega|ETA|cancelaci[oó]n(?:es)?|"
    r"rating|calificaci[oó]n|satisfacci[oó]n)\b",
    re.I,
)


def unsupported_kpi_in_text(message: str) -> str | None:
    m = _UNSUPPORTED_KPIS.search(message)
    return m.group(1) if m else None


_OTHER_COUNTRIES = ["Venezuela", "Ecuador", "Uruguay", "Paraguay", "Bolivia", "Panamá", "Costa Rica",
                    "Guatemala", "Honduras", "El Salvador", "Nicaragua", "República Dominicana", "Cuba",
                    "Puerto Rico", "España", "Estados Unidos", "Canadá", "Portugal", "Francia", "Italia"]


def unknown_country_in_text(message: str) -> str | None:
    """Countries people ask about that are not in the dataset; the extractor sometimes drops them."""
    text = _norm_text(message)
    return next((c for c in _OTHER_COUNTRIES if re.search(rf"\b{re.escape(_norm_text(c))}\b", text)), None)


def city_in_text(message: str) -> str | None:
    text = _norm_text(message)
    hits = [c for c in all_cities() if re.search(rf"\b{re.escape(_norm_text(c))}\b", text)]
    return max(hits, key=len) if len(hits) == 1 else None


_OPERATIONS_WORDS = re.compile(
    r"\bzonas?\b|pedido|[oó]rden|ordenes|m[eé]trica|kpi|promedio|ranking|ca[ií]da|crec|conversi|"
    r"perfect|ganancia|profit|adopci|penetraci|markdown|surtido|assortment|\bnps\b|ventas?\b|"
    r"\bzones?\b|orders?\b|metric",
    re.I,
)


def mentions_operations(message: str) -> bool:
    """Words that make a question about the operations data, even if it names something we lack."""
    return bool(_OPERATIONS_WORDS.search(message))


_RANK_WORDS = re.compile(r"\bm[aá]s\b|\bmenos\b|mayor|menor|mejor|peor|top|ranking|primer|[uú]ltim|"
                         r"ca[ií]d|cay|baj|crec|sub|aument|disminu|most|least|best|worst", re.I)


def rank_by_from_text(message: str) -> str:
    """'más pedidos' ranks by level; 'mayor caída/crecimiento' ranks by week-over-week change."""
    return "change" if _CHANGE_WORDS.search(message) else "value"


def finish(session: dict, session_id: str, question: str, answer: str, result_df: pd.DataFrame | None = None,
           chart: dict | None = None) -> dict:
    session["history"].append({"role": "user", "content": question})
    session["history"].append({"role": "assistant", "content": answer})
    session["history"] = session["history"][-24:]
    return {"session_id": session_id, "answer": answer, "chart": chart,
            "rows_found": 0 if result_df is None else len(result_df), "context": session["context"]}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    client = DATA.get("client")
    if client is None:
        raise HTTPException(status_code=503, detail="Chat deshabilitado: configura OPENAI_API_KEY.")
    session_id = req.session_id or str(uuid.uuid4())
    session    = get_session(session_id)
    message    = req.message

    prev_questions = [m["content"] for m in session["history"] if m["role"] == "user"]
    entities = extract_entities(message, prev_questions, session["context"], client)
    log.info("[ENTITIES] msg='%s' → %s", message[:60], json.dumps(entities, ensure_ascii=False))

    if entities.get("intent") == "out_of_scope" and mentions_operations(message):
        # The model over-applies out_of_scope to "NPS de Bogotá" or "caídas en Venezuela":
        # those are about the data, and deserve "no tengo esa métrica/país", not a brush-off.
        log.info("out_of_scope overridden: message is about operations; re-extracting")
        entities = extract_entities(
            message + "\n(Nota del sistema: la pregunta SÍ es sobre operaciones. Extrae los filtros, "
            "incluyendo países, métricas o zonas que no estén en las listas; no uses out_of_scope.)",
            prev_questions, session["context"], client)
        if entities.get("intent") in (None, "out_of_scope"):
            entities["intent"] = "summary"
    if entities.get("intent") == "out_of_scope":
        return finish(session, session_id, message, OUT_OF_SCOPE_ANSWER)

    # Deterministic backstops for what the extractor drops (seen with "¿NPS de las zonas de Bogotá?")
    if not any(entities.get(k) for k in ("city", "zone", "country")) and (city := city_in_text(message)):
        entities["city"] = city
    if (kpi := unsupported_kpi_in_text(message)) and not entities.get("metric"):
        entities.setdefault("unresolved", {})["metric"] = kpi
    if (other := unknown_country_in_text(message)) and not entities.get("country"):
        entities.setdefault("unresolved", {})["country"] = other
    missing = unresolved_filters(entities, DATA)
    if missing:
        return finish(session, session_id, message, not_found_answer(missing))

    # Deterministic backups for what the extractor tends to omit
    if not entities.get("top_n") and (n := top_n_from_text(message)):
        entities["top_n"] = n
    is_new_topic = entities.pop("is_new_topic", not prev_questions)
    if not is_new_topic and "rank_by" in entities and not _RANK_WORDS.search(message):
        entities.pop("rank_by")  # "¿y en Perfect Orders?" changes the metric, not how we rank
    inherited = None if is_new_topic else session["context"].get("rank_by")
    if not entities.get("rank_by") and not inherited and (entities.get("top_n") or entities.get("intent") == "ranking"):
        entities["rank_by"] = rank_by_from_text(message)

    # Claves que SIEMPRE se resetean si la nueva pregunta no las menciona explícitamente
    VOLATILE_KEYS = ("zone", "city", "concept", "zone_type", "priority")

    if is_new_topic:
        for key in VOLATILE_KEYS + ("metric", "metric_high", "metric_low", "top_n", "sort_order",
                                    "country", "rank_by", "intent"):
            session["context"].pop(key, None)
    else:
        # Follow-up: limpiar solo las claves volátiles que la nueva pregunta NO menciona
        for key in VOLATILE_KEYS:
            if entities.get(key) is None:
                session["context"].pop(key, None)
        if entities.get("intent", session["context"].get("intent")) != "multivariable":
            session["context"].pop("metric_high", None)
            session["context"].pop("metric_low", None)

    for k, v in entities.items():
        if v is not None:
            session["context"][k] = v

    log.info("[CONTEXT] after merge: %s", json.dumps(session["context"], ensure_ascii=False))
    try:
        ctx_intent = session["context"].get("intent", "")
        if ctx_intent == "multivariable":
            result_df, query_desc = execute_multivariable(session["context"], DATA["metrics_long"])
        elif ctx_intent == "correlation":
            result_df, query_desc = execute_correlation(DATA["metrics_long"], country=session["context"].get("country",""), min_r=0.4)
        else:
            result_df, query_desc = execute_query(
                session["context"], DATA["metrics_long"], DATA["orders_long"]
            )
        result_df = result_df.reset_index(drop=True)
        log.info("[QUERY] %s → %d filas", query_desc, len(result_df))
        data_text  = df_to_text(result_df)
        chart_data = build_chart_data(result_df, session["context"])
    except Exception as e:
        log.error("Query failed: %s", e)
        result_df  = pd.DataFrame()
        data_text  = "Error al consultar datos."
        chart_data = None

    bot_turns = [m for m in session["history"] if m["role"] == "assistant"]
    result_zones = list(result_df["ZONE"]) if "ZONE" in result_df.columns else []
    answer    = generate_answer(message, data_text, session["context"], bot_turns, client,
                                result_zones, set(DATA["orders_long"]["ZONE"]))
    return finish(session, session_id, message, answer, result_df, chart_data)

@app.delete("/api/session/{session_id}")
async def reset_session(session_id: str):
    SESSIONS.pop(session_id, None)
    return {"status": "reset"}

@app.get("/api/insights")
async def get_insights(country: str = None, tipo: str = None,
                       severity: str = None, limit: int = Query(20, ge=1, le=200)):
    insights = DATA["insights"]
    if country:
        cc = resolve_country(country) or country.upper()
        insights = [i for i in insights if i.country == cc]
    if tipo:
        insights = [i for i in insights if i.tipo == tipo]
    if severity:
        insights = [i for i in insights if i.severity == severity]
    if tipo or severity:
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
                      sort: str = "desc", limit: int = Query(10, ge=1, le=100)):
    """Top/bottom zonas por valor L0W."""
    if metric == "Orders":
        df = DATA["orders_long"].copy()
    else:
        df = DATA["metrics_long"].copy()
        df = df[df["METRIC"] == metric]

    if country:
        df = df[df["COUNTRY"] == (resolve_country(country) or country.upper())]

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
async def timeseries_multi(country: str, city: str, metric: str = "Orders", top_n: int = Query(5, ge=1, le=12)):
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
async def get_correlations(country: str = None, min_r: float = Query(0.5, ge=0, le=1)):
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
    import math
    import unicodedata

    def norm(s):
        return unicodedata.normalize("NFD", str(s)).encode("ascii","ignore").decode().lower()

    def clean(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return round(v, 4) if isinstance(v, float) else v

    m = DATA["metrics_long"]
    o = DATA["orders_long"]

    cc = resolve_country(country) or country.upper()
    candidates = sorted(o.loc[o["COUNTRY"] == cc, "ZONE"].unique())
    exact = [z for z in candidates if norm(z) == norm(zone)]
    partial = [z for z in candidates if norm(zone) in norm(z)]
    if not exact and len(partial) > 1:
        raise HTTPException(409, {"message": f"«{zone}» coincide con varias zonas", "candidates": partial[:20]})
    target = (exact or partial or [None])[0]
    mask_m = (m["COUNTRY"] == cc) & (m["ZONE"] == target)
    mask_o = (o["COUNTRY"] == cc) & (o["ZONE"] == target)

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
    raw_mask = (DATA["raw_orders"]["COUNTRY"] == cc) & (DATA["raw_orders"]["ZONE"] == target)
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
async def search_zones(q: str = Query("", max_length=60), limit: int = Query(10, ge=1, le=50),
                       city: str = "", country: str = ""):
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
            matched = matched[matched["COUNTRY"] == (resolve_country(country) or country.upper())]
        matched = matched[matched["CITY"].apply(norm) == norm(city)]
    else:
        q_norm = norm(q)
        matched = m[m["ZONE"].apply(norm).str.contains(q_norm, regex=False, na=False)].copy()
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

    today = datetime.now(UTC).strftime("%d %b %Y")

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
<title>Delivery Ops — Reporte Ejecutivo {today}</title>
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
      <div class="hdr-title">Delivery Ops Intelligence</div>
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
    <span>Generado por Delivery Ops Intelligence</span>
    <span>{len(ins)} insights analizados esta semana</span>
  </div>
</div>
</body>
</html>"""

    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html, headers={
        "Content-Disposition": f"attachment; filename=delivery-ops-report-{datetime.now(UTC).strftime('%Y%m%d')}.html"
    })


@app.get("/api/cities")
async def get_cities(country: str):
    """Todas las ciudades de un pais, ordenadas alfabeticamente."""
    m = DATA["metrics_long"]
    o = DATA["orders_long"]
    cc = resolve_country(country) or country.upper()
    cities_m = set(m[m["COUNTRY"]==cc]["CITY"].dropna().unique())
    cities_o = set(o[o["COUNTRY"]==cc]["CITY"].dropna().unique())
    all_cities = sorted(cities_m | cities_o)
    return {"cities": all_cities, "total": len(all_cities)}

# ── Static files + SPA ────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

@app.get("/")
async def root():
    return FileResponse(BASE_DIR / "static" / "index.html")

@app.get("/health")
async def health():
    return {"status": "ok", "chat_enabled": DATA.get("client") is not None,
            "timestamp": datetime.now(UTC).isoformat()}

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=False)