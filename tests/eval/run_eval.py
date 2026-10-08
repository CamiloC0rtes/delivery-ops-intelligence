"""Accuracy eval for the NL-to-query assistant against the real model.

For each question in golden.jsonl:
  1. Extraction  - did the LLM produce the expected filters (metric, country, top_n, ...)?
  2. Grounding   - does the answer mention the top zones that pandas returns for those filters?
  3. Fabrication - does the answer name dataset zones that are NOT in the ground-truth result?

Ground truth is computed with the same deterministic query engine, from the expected filters,
so a wrong extraction shows up as both an extraction miss and a grounding miss.

Usage:  OPENAI_API_KEY=... python -m tests.eval.run_eval [--min-pass 0.8]
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DATA_FILE", str(ROOT / "data" / "sample_data.xlsx"))
if __name__ == "__main__" and not os.getenv("OPENAI_API_KEY"):
    raise SystemExit("OPENAI_API_KEY is required for the eval.")

from fastapi.testclient import TestClient  # noqa: E402

import app as app_module  # noqa: E402
from query_engine import execute_correlation, execute_multivariable, execute_query  # noqa: E402

HERE = Path(__file__).parent


def ground_truth(expect: dict, data: dict):
    intent = expect.get("intent")
    if intent == "multivariable":
        df, _ = execute_multivariable(expect, data["metrics_long"])
    elif intent == "correlation":
        df, _ = execute_correlation(data["metrics_long"], country=expect.get("country", ""), min_r=0.4)
    else:
        df, _ = execute_query(expect, data["metrics_long"], data["orders_long"])
    return df


def score_case(case: dict, client: TestClient, data: dict, all_zones: set[str]) -> dict:
    session = ""
    if case.get("after"):
        session = client.post("/api/chat", json={"message": case["after"]}).json()["session_id"]
    start = time.perf_counter()
    r = client.post("/api/chat", json={"message": case["question"], "session_id": session}).json()
    latency = time.perf_counter() - start

    ctx, answer = r["context"], r["answer"]
    failures = []
    for key, want in case["expect"].items():
        got = ctx.get(key)
        if isinstance(want, str) and isinstance(got, str):
            ok = want.lower() == got.lower()
        else:
            ok = want == got
        if not ok:
            failures.append(f"extraction {key}: want {want!r}, got {got!r}")

    truth = ground_truth(case["expect"], data)
    truth_zones = list(truth["ZONE"]) if "ZONE" in truth.columns else []
    for zone in truth_zones[: case.get("mention_top", 0)]:
        if zone not in answer:
            failures.append(f"missing top zone {zone!r}")

    # Longest names first so "Los Cedros Norte" is not also counted as "Los Cedros"
    mentioned, text = [], answer
    for zone in sorted(all_zones, key=len, reverse=True):
        if zone in text:
            mentioned.append(zone)
            text = text.replace(zone, "")
    fabricated = [z for z in mentioned if z not in truth_zones]
    if truth_zones and fabricated:
        failures.append(f"zones not in result: {fabricated}")

    return {"id": case["id"], "passed": not failures, "failures": failures,
            "latency_s": round(latency, 2), "answer": answer}


def write_report(results, out: Path):
    passed = sum(r["passed"] for r in results)
    lat = sorted(r["latency_s"] for r in results)
    p95 = lat[min(int(len(lat) * 0.95), len(lat) - 1)]
    lines = ["# NL-to-query accuracy eval", "",
             f"**{passed}/{len(results)} passed** · p95 latency {p95:.2f}s · median {statistics.median(lat):.2f}s", "",
             "| Case | Result | Notes |", "|---|---|---|"]
    lines += [f"| {r['id']} | {'✅' if r['passed'] else '❌'} | {'; '.join(r['failures'])} |" for r in results]
    failed = [r for r in results if not r["passed"]]
    if failed:
        lines += ["", "## Failed answers", ""]
        for r in failed:
            lines += [f"**{r['id']}**", "", "> " + r["answer"].replace("\n", " ")[:700], ""]
    (out / "eval_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "eval_report.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return passed / len(results), p95


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-pass", type=float, default=0.8)
    parser.add_argument("--out", default=".")
    args = parser.parse_args()

    cases = [json.loads(line) for line in (HERE / "golden.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    with TestClient(app_module.app) as client:
        data = app_module.DATA
        all_zones = set(data["orders_long"]["ZONE"])
        results = []
        for case in cases:
            r = score_case(case, client, data, all_zones)
            results.append(r)
            print(f"{'✅' if r['passed'] else '❌'} {r['id']:<22} {'; '.join(r['failures'])}")
    rate, p95 = write_report(results, Path(args.out))
    print(f"\npass rate {rate:.0%} · p95 {p95:.2f}s")
    if rate < args.min_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
