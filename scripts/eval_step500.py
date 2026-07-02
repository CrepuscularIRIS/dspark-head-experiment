"""Eval step-500 checkpoints: RSMH vs Vanilla retrain vs released baseline."""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", os.environ.get("EVAL_PORT", "29800"))
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from deepspec.eval.dspark import Qwen3DSparkEvaluator

DRAFT_CKPT = os.environ.get("DRAFT_CKPT", "deepseek-ai/dspark_qwen3_4b_block7")
EVAL_NAME = os.environ.get("EVAL_NAME", "baseline")
EVAL_PROMPTS = int(os.environ.get("EVAL_PROMPTS", "100"))

DATASETS = [
    ("gsm8k", EVAL_PROMPTS),
    ("alpaca", EVAL_PROMPTS),
]


class EvalArgs:
    target_name_or_path = "Qwen/Qwen3-4B"
    draft_name_or_path = DRAFT_CKPT
    max_new_tokens = 256
    temperature = 1.0
    confidence_threshold = 0.0
    tensorboard_dir = None
    step = None
    seed = 42
    tasks = DATASETS


def main():
    out_dir = os.path.join("outputs", "eval_step500")
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    print(f"[eval] {EVAL_NAME}: draft={DRAFT_CKPT}, datasets={len(DATASETS)}, prompts={EVAL_PROMPTS}", flush=True)

    evaluator = Qwen3DSparkEvaluator(0, EvalArgs())
    evaluator.evaluate()
    evaluator.clean_up()

    results = {}
    for row in evaluator.metrics_rows:
        ds = row["dataset"]
        results[ds] = {
            "acceptance_length": row["acceptance_length"],
            "verify_rate": row["verify_rate"],
            "accept_rates_by_position": row["accept_rates_by_position"],
        }
    results["_meta"] = {
        "name": EVAL_NAME,
        "draft": DRAFT_CKPT,
        "elapsed_s": time.time() - t0,
    }

    out_path = os.path.join(out_dir, f"{EVAL_NAME}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] Saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
