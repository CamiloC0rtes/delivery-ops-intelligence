"""
catalog.py
Single source of truth for metrics, countries and cities.
Used by the data generator, the LLM prompt, entity validation and the query engine.
"""

ORDERS = "Orders"

# name -> (kind, aliases).  kind: "ratio" (0-1), "currency" (per-order value) or "count".
METRICS: dict[str, dict] = {
    ORDERS: {"kind": "count", "aliases": ["órdenes", "ordenes", "pedidos", "orders", "volumen"]},
    "Restaurant Conversion Rate": {
        "kind": "ratio",
        "aliases": ["cvr restaurantes", "conversión restaurantes", "conversion restaurantes",
                    "restaurant conversion", "restaurant cvr"],
    },
    "Retail Conversion Rate": {
        "kind": "ratio",
        "aliases": ["cvr retail", "conversión retail", "conversion retail", "retail conversion", "retail cvr"],
    },
    "Gross Profit per Order": {
        "kind": "currency",
        "aliases": ["gross profit", "ganancia", "margen", "profit", "utilidad", "rentabilidad"],
    },
    "Perfect Orders": {
        "kind": "ratio",
        "aliases": ["perfect orders", "órdenes perfectas", "ordenes perfectas", "pedidos perfectos", "calidad"],
    },
    "Express Delivery Adoption": {
        "kind": "ratio",
        "aliases": ["express", "entrega express", "express adoption", "delivery express"],
    },
    "Subscription Adoption": {
        "kind": "ratio",
        "aliases": ["suscripción", "suscripcion", "subscription", "membresía", "membresia"],
    },
    "Lead Penetration": {
        "kind": "ratio",
        "aliases": ["lead penetration", "penetración", "penetracion", "leads"],
    },
    "Markdowns / GMV": {
        "kind": "ratio",
        "aliases": ["markdowns", "descuentos", "promociones", "markdown"],
    },
    "Optimal Assortment Sessions": {
        "kind": "ratio",
        "aliases": ["assortment", "surtido", "surtido óptimo", "surtido optimo"],
    },
}

RATE_METRICS = [m for m, v in METRICS.items() if m != ORDERS]

# code -> (name, aliases, cities)
COUNTRIES: dict[str, dict] = {
    "CO": {"name": "Colombia", "aliases": ["colombia"], "cities": ["Bogota", "Medellin", "Cali"]},
    "MX": {"name": "Mexico", "aliases": ["mexico", "méxico"], "cities": ["Ciudad De Mexico", "Guadalajara", "Monterrey"]},
    "BR": {"name": "Brasil", "aliases": ["brasil", "brazil"], "cities": ["Sao Paulo", "Rio De Janeiro"]},
    "AR": {"name": "Argentina", "aliases": ["argentina"], "cities": ["Buenos Aires", "Cordoba"]},
    "PE": {"name": "Peru", "aliases": ["peru", "perú"], "cities": ["Lima"]},
    "CL": {"name": "Chile", "aliases": ["chile"], "cities": ["Santiago"]},
}

ZONE_TYPES = ["Wealthy", "Non Wealthy"]
PRIORITIES = ["High Priority", "Prioritized", "Not Prioritized"]

INTENTS = ["ranking", "trend", "comparison", "anomaly", "summary", "filter", "multivariable", "correlation"]
CONCEPTS = ["zonas problemáticas", "alto crecimiento", "bajo performance", "deterioro sostenido",
            "mejora sostenida", "mejor zona", "peor zona", "outliers"]


def all_cities() -> list[str]:
    return [c for v in COUNTRIES.values() for c in v["cities"]]


def resolve_country(value: str | None) -> str | None:
    """'Colombia', 'co', 'CO', 'Bogota' -> 'CO'. Unknown -> None."""
    if not value:
        return None
    v = _norm(value)
    for code, info in COUNTRIES.items():
        if v == code.lower() or v in [_norm(a) for a in info["aliases"]] or v in [_norm(c) for c in info["cities"]]:
            return code
    return None


def resolve_metric(value: str | None) -> str | None:
    """Exact name, exact alias, or a long-enough alias contained in the text. Unknown -> None."""
    if not value:
        return None
    v = _norm(value)
    for name in METRICS:
        if v == _norm(name):
            return name
    for name, info in METRICS.items():
        if v in (_norm(a) for a in info["aliases"]):
            return name
    for name, info in METRICS.items():
        for alias in info["aliases"]:
            a = _norm(alias)
            if len(a) >= 6 and a in v:
                return name
    return None


def _norm(s: str) -> str:
    import unicodedata

    return unicodedata.normalize("NFD", str(s)).encode("ascii", "ignore").decode().lower().strip()
