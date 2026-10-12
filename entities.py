"""
entities.py
Validates what the LLM extracts before it touches the data.

The LLM returns free-form JSON. Anything that doesn't map to the catalog is never guessed:
it is recorded in `unresolved` so the app can tell the user "I have no data for X" instead
of silently dropping the filter and answering about everything. Numbers are coerced and
clamped, and enums fall back to safe defaults.
"""

import json
import re
from typing import Literal

from pydantic import BaseModel, field_validator

from catalog import CONCEPTS, INTENTS, PRIORITIES, resolve_country, resolve_metric

Intent = Literal["ranking", "trend", "comparison", "anomaly", "summary", "filter", "multivariable", "correlation",
                 "out_of_scope"]
assert set(INTENTS) == set(Intent.__args__)

MAX_TOP_N = 50
_WORD_NUMBERS = {"uno": 1, "one": 1, "tres": 3, "three": 3, "cinco": 5, "five": 5, "diez": 10, "ten": 10}


class Entities(BaseModel):
    intent: Intent = "summary"
    metric: str | None = None
    metric_high: str | None = None
    metric_low: str | None = None
    city: str | None = None
    country: str | None = None
    zone: str | None = None
    concept: str | None = None
    top_n: int | None = None
    sort_order: Literal["asc", "desc"] = "desc"
    rank_by: Literal["value", "change"] | None = None  # rank by current level or by week-over-week change
    zone_type: Literal["Wealthy", "Non Wealthy", "comparison"] | None = None
    priority: str | None = None
    is_new_topic: bool = True
    unresolved: dict[str, str] = {}  # field -> raw value that did not map to the catalog

    @field_validator("intent", mode="before")
    @classmethod
    def _intent(cls, v):
        return v if v in INTENTS else "summary"

    @field_validator("metric", "metric_high", "metric_low", mode="before")
    @classmethod
    def _metric(cls, v):
        if isinstance(v, list):  # LLMs sometimes return ["Orders"] or ["Perfect Orders", "deterioro sostenido"]
            return next((m for m in map(resolve_metric, v) if m), None)
        return resolve_metric(v) if v else None

    @field_validator("country", mode="before")
    @classmethod
    def _country(cls, v):
        return resolve_country(v) if v else None

    @field_validator("city", "zone", mode="before")
    @classmethod
    def _text(cls, v):
        if not v or not isinstance(v, str):
            return None
        v = v.strip()[:60]
        return v or None

    @field_validator("concept", mode="before")
    @classmethod
    def _concept(cls, v):
        v = (v or "").strip().lower() if isinstance(v, str) else ""
        return v if v in CONCEPTS else None

    @field_validator("top_n", mode="before")
    @classmethod
    def _top_n(cls, v):
        if v is None or v == "":
            return None
        if isinstance(v, str):
            v = _WORD_NUMBERS.get(v.strip().lower(), v)
            m = re.search(r"\d+", str(v))
            if not m:
                return None
            v = m.group()
        try:
            n = int(float(v))
        except (TypeError, ValueError):
            return None
        return max(1, min(n, MAX_TOP_N)) if n > 0 else None

    @field_validator("rank_by", mode="before")
    @classmethod
    def _rank_by(cls, v):
        return v if v in ("value", "change") else None

    @field_validator("sort_order", mode="before")
    @classmethod
    def _sort(cls, v):
        return "asc" if str(v).lower().startswith("asc") else "desc"

    @field_validator("zone_type", mode="before")
    @classmethod
    def _zone_type(cls, v):
        if not v or not isinstance(v, str):
            return None
        s = v.lower()
        if "vs" in s or ("non" in s and s.count("wealthy") > 1):
            return "comparison"
        if "non" in s:
            return "Non Wealthy"
        if "wealthy" in s:
            return "Wealthy"
        return None

    @field_validator("priority", mode="before")
    @classmethod
    def _priority(cls, v):
        if not v or not isinstance(v, str):
            return None
        s = v.lower()
        if "high" in s or "alta" in s:
            return "High Priority"
        if "not" in s or "no " in s:
            return "Not Prioritized"
        if "prior" in s:
            return "Prioritized"
        return v if v in PRIORITIES else None

    def as_context(self) -> dict:
        """Dict form for the session context (None values dropped)."""
        return {k: v for k, v in self.model_dump().items()
                if v is not None and k not in ("is_new_topic", "unresolved")}


def _move_concepts_out_of_metrics(fields: dict) -> dict:
    """The extractor sometimes puts a concept ("deterioro sostenido") in a metric field.

    That is a filter, not an unknown metric: move it to `concept` instead of answering
    "no tengo datos para la métrica «deterioro sostenido»".
    """
    fields = dict(fields)
    for key in ("metric", "metric_high", "metric_low"):
        value = fields.get(key)
        items = value if isinstance(value, list) else [value]
        concepts = [v for v in items if isinstance(v, str) and v.strip().lower() in CONCEPTS]
        if not concepts:
            continue
        fields.setdefault("concept", concepts[0].strip().lower())
        rest = [v for v in items if v not in concepts]
        fields[key] = (rest if isinstance(value, list) else rest[0]) if rest else None
    return fields


def parse_llm_json(raw: str) -> Entities:
    """Parse the extractor's output. Malformed JSON -> safe defaults, never an exception."""
    text = (raw or "").strip().replace("```json", "").replace("```", "").strip()
    match = re.search(r"\{.*\}", text, re.S)
    try:
        data = json.loads(match.group() if match else text)
        if not isinstance(data, dict):
            raise ValueError("not an object")
    except (ValueError, json.JSONDecodeError):
        return Entities(intent="summary")
    fields = {k: v for k, v in data.items() if k in Entities.model_fields and k != "unresolved"}
    fields = _move_concepts_out_of_metrics(fields)
    if fields.get("intent") != "multivariable":
        # metric_high/metric_low only mean something in "alto X pero bajo Y" questions
        fields.pop("metric_high", None)
        fields.pop("metric_low", None)
    entities = Entities(**fields)
    unresolved = {
        k: str(fields[k]).strip() for k in ("metric", "metric_high", "metric_low", "country")
        if fields.get(k) not in (None, "", []) and getattr(entities, k) is None
    }
    if unresolved:
        entities.unresolved = unresolved
    return entities
