"""
Generate a synthetic weekly operations dataset with the same structure the app expects.

Every zone name and number is invented. A few patterns are planted on purpose
(sustained declines, sharp week-over-week drops, strong growth, Wealthy zones
converting better) so the insight engine and the eval have something real to find.

Usage:
    python scripts/generate_data.py                       # -> data/sample_data.xlsx
    python scripts/generate_data.py --seed 7 --out x.xlsx
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from catalog import COUNTRIES, METRICS, ORDERS, PRIORITIES  # noqa: E402

WEEKS = 9  # L8W (oldest) ... L0W (latest)
MASC = ["Almendros", "Cedros", "Robles", "Pinos", "Olivos", "Nogales", "Sauces", "Laureles",
        "Cipreses", "Naranjos", "Arrayanes", "Alamos", "Guaduales", "Cerezos", "Manzanos", "Eucaliptos"]
FEM = ["Acacias", "Palmas", "Margaritas", "Violetas", "Orquideas", "Gardenias", "Magnolias", "Azucenas",
       "Camelias", "Hortensias", "Lomas", "Praderas", "Colinas", "Brisas", "Fuentes", "Terrazas"]
SUFFIXES = ["", " Norte", " Sur", " Alto"]

# metric -> (center, spread, wealthy_bonus) for ratio/currency metrics
PROFILE = {
    "Restaurant Conversion Rate": (0.88, 0.05, 0.03),
    "Retail Conversion Rate": (0.86, 0.07, 0.03),
    "Gross Profit per Order": (1.4, 0.9, 0.6),
    "Perfect Orders": (0.88, 0.04, 0.02),
    "Express Delivery Adoption": (0.20, 0.08, 0.05),
    "Subscription Adoption": (0.30, 0.10, 0.08),
    "Lead Penetration": (0.15, 0.06, 0.02),
    "Markdowns / GMV": (0.12, 0.04, -0.01),
    "Optimal Assortment Sessions": (0.72, 0.10, 0.05),
}


# Metrics driven by a shared per-zone "engagement" factor, so they correlate (loading ~0.8)
CORRELATED = {"Subscription Adoption", "Express Delivery Adoption", "Gross Profit per Order"}


def zone_names(rng, n):
    pool = [f"Los {m}{s}" for m in MASC for s in SUFFIXES] + [f"Las {f}{s}" for f in FEM for s in SUFFIXES]
    rng.shuffle(pool)
    return pool[:n]


def series(rng, start, drift, noise, n=WEEKS):
    steps = rng.normal(drift, noise, n - 1)
    return np.concatenate([[start], start + np.cumsum(steps)])


def generate(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    zones = []
    for code, info in COUNTRIES.items():
        for city in info["cities"]:
            for _ in range(int(rng.integers(7, 12))):
                zones.append({"COUNTRY": code, "CITY": city})
    for z, name in zip(zones, zone_names(rng, len(zones)), strict=True):
        z["ZONE"] = name
        z["ZONE_TYPE"] = "Wealthy" if rng.random() < 0.3 else "Non Wealthy"
        z["ZONE_PRIORITIZATION"] = rng.choice(PRIORITIES, p=[0.15, 0.45, 0.40])

    # Planted patterns, chosen at random but reproducible
    idx = rng.permutation(len(zones))
    declining = set(idx[:8])        # Perfect Orders falls 3+ weeks in a row
    sharp_drop = set(idx[8:14])     # Orders drop ~30% in the last week
    growth = set(idx[14:20])        # Orders grow ~25% in the last week

    metric_rows, order_rows = [], []
    for i, z in enumerate(zones):
        wealthy = z["ZONE_TYPE"] == "Wealthy"
        engagement = rng.normal()

        # Orders
        base = float(rng.lognormal(7.0 if wealthy else 6.3, 0.9))
        vals = series(rng, base, base * 0.01, base * 0.04)
        if i in sharp_drop:
            vals[-1] = vals[-2] * rng.uniform(0.62, 0.75)
        if i in growth:
            vals[-1] = vals[-2] * rng.uniform(1.22, 1.35)
        vals = np.maximum(vals, 1).round(0)
        order_rows.append({**{k: z[k] for k in ("COUNTRY", "CITY", "ZONE")}, "METRIC": ORDERS,
                           **{f"L{WEEKS - 1 - w}W": vals[w] for w in range(WEEKS)}})

        # Rate / currency metrics
        for metric, (center, spread, bonus) in PROFILE.items():
            shock = 0.8 * engagement + 0.6 * rng.normal() if metric in CORRELATED else rng.normal()
            start = center + (bonus if wealthy else 0) + spread * shock
            vals = series(rng, start, 0, spread * 0.06)
            if metric == "Perfect Orders" and i in declining:
                drops = rng.uniform(0.015, 0.03, 3)
                for k in range(3):
                    vals[WEEKS - 3 + k] = vals[WEEKS - 4 + k] - drops[k]
            if METRICS[metric]["kind"] == "ratio":
                vals = np.clip(vals, 0.0, 1.0)
            vals = vals.round(4)
            if rng.random() < 0.02:  # a little missing data, like the real world
                vals[int(rng.integers(0, WEEKS - 1))] = np.nan
            metric_rows.append({**{k: z[k] for k in ("COUNTRY", "CITY", "ZONE", "ZONE_TYPE",
                                                     "ZONE_PRIORITIZATION")},
                                "METRIC": metric,
                                **{f"L{WEEKS - 1 - w}W_ROLL": vals[w] for w in range(WEEKS)}})

    return pd.DataFrame(metric_rows), pd.DataFrame(order_rows)


DICTIONARY = pd.DataFrame(
    [
        ("COUNTRY", "ISO-like country code"),
        ("CITY", "City"),
        ("ZONE", "Delivery zone inside a city (synthetic name)"),
        ("ZONE_TYPE", "Wealthy / Non Wealthy"),
        ("ZONE_PRIORITIZATION", "High Priority / Prioritized / Not Prioritized"),
        ("METRIC", "Metric name, see catalog.py"),
        ("L8W..L0W", "Weekly values, L0W = most recent week (_ROLL = rolling for rate metrics)"),
    ],
    columns=["Column", "Description"],
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/sample_data.xlsx")
    args = parser.parse_args()

    metrics, orders = generate(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out) as xl:
        metrics.to_excel(xl, sheet_name="RAW_INPUT_METRICS", index=False)
        orders.to_excel(xl, sheet_name="RAW_ORDERS", index=False)
        DICTIONARY.to_excel(xl, sheet_name="RAW_SUMMARY", index=False)
    print(f"Wrote {out}: {orders.shape[0]} zones, {metrics.shape[0]} metric rows (seed={args.seed})")


if __name__ == "__main__":
    main()
