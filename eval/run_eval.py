"""Measure response continuity and latency, with and without long-term memory.

For each persona the assistant is first told some facts in one conversation.
Then, in a brand-new conversation (so short-term memory can't help), it is asked
questions that need those facts. A probe passes if the reply mentions one of the
expected keywords.

    python -m eval.run_eval                 # uses your .env settings
    python -m eval.run_eval --runs 3        # repeat for steadier numbers

Results are written to eval/results.json and eval/results.md.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from dataclasses import replace
from pathlib import Path

from app.config import settings
from app.logs import InteractionLog
from app.service import Assistant

from .metrics import contains_any, latency_summary

HERE = Path(__file__).parent


def run(runs: int, scenario_path: Path) -> dict:
    scenarios = json.loads(scenario_path.read_text(encoding="utf-8"))
    eval_settings = replace(settings, chroma_collection=f"{settings.chroma_collection}_eval")
    assistant = Assistant(eval_settings, log=InteractionLog(":memory:"))

    cases = []
    chat_ms = {"memory": [], "baseline": []}
    recall_ms = []

    for run_no in range(runs):
        for sc in scenarios:
            user_id = f"eval-{sc['persona']}-{uuid.uuid4().hex[:6]}"
            for msg in sc["teach"]:
                assistant.chat(user_id, "teach", msg)
                assistant.remember(user_id, msg)

            for probe in sc["probes"]:
                t0 = time.perf_counter()
                assistant.memory.recall(user_id, probe["question"])
                recall_ms.append((time.perf_counter() - t0) * 1000)

                with_mem = assistant.chat(user_id, "probe", probe["question"], use_memory=True)
                without = assistant.chat(user_id, "baseline", probe["question"], use_memory=False)
                chat_ms["memory"].append(with_mem.latency_ms)
                chat_ms["baseline"].append(without.latency_ms)
                cases.append(
                    {
                        "run": run_no + 1,
                        "persona": sc["persona"],
                        "question": probe["question"],
                        "expect": probe["expect"],
                        "with_memory": {"pass": contains_any(with_mem.reply, probe["expect"]), "reply": with_mem.reply},
                        "baseline": {"pass": contains_any(without.reply, probe["expect"]), "reply": without.reply},
                        "recalled": [m["text"] for m in with_mem.recalled],
                    }
                )

    n = len(cases)
    mem_rate = sum(c["with_memory"]["pass"] for c in cases) / n if n else 0.0
    base_rate = sum(c["baseline"]["pass"] for c in cases) / n if n else 0.0
    return {
        "config": {
            "llm": f"{settings.llm_provider}:{settings.llm_model}",
            "embeddings": f"{settings.embedding_provider}:{settings.embedding_model}",
            "vector_backend": settings.vector_backend,
            "runs": runs,
            "probes": n,
        },
        "continuity": {
            "with_memory": round(mem_rate * 100, 1),
            "without_memory": round(base_rate * 100, 1),
            "improvement_points": round((mem_rate - base_rate) * 100, 1),
        },
        "latency_ms": {
            "recall_only": latency_summary(recall_ms),
            "chat_with_memory": latency_summary(chat_ms["memory"]),
            "chat_without_memory": latency_summary(chat_ms["baseline"]),
        },
        "cases": cases,
    }


def to_markdown(r: dict) -> str:
    c, lat, cfg = r["continuity"], r["latency_ms"], r["config"]
    lines = [
        "# Evaluation results",
        "",
        f"LLM `{cfg['llm']}` · embeddings `{cfg['embeddings']}` · {cfg['probes']} probes over {cfg['runs']} run(s)",
        "",
        "| Metric | With memory | Without memory |",
        "|---|---|---|",
        f"| Continuity (probes answered from earlier sessions) | {c['with_memory']}% | {c['without_memory']}% |",
        f"| Chat latency, mean | {lat['chat_with_memory']['mean']} ms | {lat['chat_without_memory']['mean']} ms |",
        f"| Chat latency, p95 | {lat['chat_with_memory']['p95']} ms | {lat['chat_without_memory']['p95']} ms |",
        "",
        f"Memory recall step alone: mean {lat['recall_only']['mean']} ms, p95 {lat['recall_only']['p95']} ms.",
        "",
        "## Failed probes (with memory)",
        "",
    ]
    failed = [x for x in r["cases"] if not x["with_memory"]["pass"]]
    if not failed:
        lines.append("None.")
    for x in failed:
        lines.append(f"- **{x['persona']}**: {x['question']} (expected one of {', '.join(x['expect'])})")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--scenarios", type=Path, default=HERE / "scenarios.json")
    args = parser.parse_args()

    results = run(args.runs, args.scenarios)
    (HERE / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (HERE / "results.md").write_text(to_markdown(results), encoding="utf-8")
    c = results["continuity"]
    print(f"Continuity: {c['with_memory']}% with memory vs {c['without_memory']}% without")
    print(f"Chat latency (with memory): {results['latency_ms']['chat_with_memory']}")
    print(f"Recall step only: {results['latency_ms']['recall_only']}")
    print("Saved eval/results.md and eval/results.json")


if __name__ == "__main__":
    main()
