"""Ceiling probe: baseline per-position acceptance profile + decay analysis.

Measures WHERE suffix decay bites in the 1st-order Markov head, then combines
with DPC ceiling to size the prize for higher-order corrections.

Output: per-position acceptance rates, decay curve, and gap-to-ceiling analysis.
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29600")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from deepspec.eval.dspark import Qwen3DSparkEvaluator


DATASETS = [
    ("gsm8k", 200),
    ("math500", 200),
    ("alpaca", 200),
    ("mt-bench", 80),
]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "ceiling_probe")


class EvalArgs:
    target_name_or_path = "Qwen/Qwen3-4B"
    draft_name_or_path = "deepseek-ai/dspark_qwen3_4b_block7"
    max_new_tokens = 256
    temperature = 1.0
    confidence_threshold = 0.0
    tensorboard_dir = None
    step = None
    seed = 42
    tasks = DATASETS


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    t0 = time.time()
    print(f"[ceiling-probe] Starting baseline eval on {len(DATASETS)} datasets", flush=True)
    print(f"[ceiling-probe] Args: {json.dumps(vars(EvalArgs()), default=str, indent=2)}", flush=True)

    evaluator = Qwen3DSparkEvaluator(0, EvalArgs())
    evaluator.evaluate()
    evaluator.clean_up()

    results = {}
    for row in evaluator.metrics_rows:
        ds = row["dataset"]
        results[ds] = {
            "acceptance_length": row["acceptance_length"],
            "verify_rate": row["verify_rate"],
            "draft_tokens_per_proposal": row["draft_tokens_per_proposal"],
            "accept_rates_by_position": row["accept_rates_by_position"],
        }

    results["_meta"] = {
        "target": EvalArgs.target_name_or_path,
        "draft": EvalArgs.draft_name_or_path,
        "temperature": EvalArgs.temperature,
        "seed": EvalArgs.seed,
        "elapsed_s": time.time() - t0,
        "block_size": 7,
    }

    out_path = os.path.join(OUTPUT_DIR, "baseline_profile.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[ceiling-probe] Saved to {out_path}", flush=True)

    print("\n=== CEILING PROBE: Per-Position Acceptance Profile ===", flush=True)
    print(f"{'Dataset':<12} {'AccLen':>7} {'Pos0':>7} {'Pos1':>7} {'Pos2':>7} {'Pos3':>7} {'Pos4':>7} {'Pos5':>7} {'Pos6':>7}", flush=True)
    print("-" * 80, flush=True)
    for ds in ["gsm8k", "math500", "alpaca", "mt-bench"]:
        if ds not in results:
            continue
        r = results[ds]
        rates = r["accept_rates_by_position"]
        vals = [f"{v:.4f}" if v is not None else "  -   " for v in rates]
        print(f"{ds:<12} {r['acceptance_length']:>7.2f} {'  '.join(vals)}", flush=True)

    print("\n=== DECAY ANALYSIS ===", flush=True)
    for ds in ["gsm8k", "math500"]:
        if ds not in results:
            continue
        rates = results[ds]["accept_rates_by_position"]
        if rates[0] is not None and rates[0] > 0:
            for i in range(1, len(rates)):
                if rates[i] is not None and rates[i-1] is not None and rates[i-1] > 0:
                    decay = 1.0 - rates[i] / rates[i-1]
                    print(f"  {ds} pos{i-1}→{i}: decay={decay:.1%} (α_{i-1}={rates[i-1]:.4f} → α_{i}={rates[i]:.4f})", flush=True)

    print(f"\n[ceiling-probe] Done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
