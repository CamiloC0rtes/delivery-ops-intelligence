"""
entities.py
Validates what the LLM extracts before it touches the data.

The LLM returns free-form JSON. Anything that doesn't map to the catalog is dropped
(not guessed), numbers are coerced and clamped, and enums fall back to safe defaults.
The query engine only ever sees a clean `Entities` object.
"""

import json
import re
from typing import Literal

from pydantic import BaseModel, field_validator

from catalog import CONCEPTS, INTENTS, PRIORITIES, resolve_country, resolve_metric

Intent = Literal["ranking", "trend", "comparison", "anomaly", "summary", "filter", "multivariable", "correlation"]
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
    zone_type: Literal["Wealthy", "Non Wealthy", "comparison"] | None = None
    priority: str | None = None
    is_new_topic: bool = True

    @field_validator("intent", mode="before")
    @classmethod
    def _intent(cls, v):
        return v if v in INTENTS else "summary"

    @field_validator("metric", "metric_high", "metric_low", mode="before")
    @classmethod
    def _metric(cls, v):
        if isinstance(v, list):  # LLMs sometimes return ["Orders"]
            v = v[0] if v else None
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
        return {k: v for k, v in self.model_dump().items() if v is not None and k != "is_new_topic"}


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
    return Entities(**{k: v for k, v in data.items() if k in Entities.model_fields})
