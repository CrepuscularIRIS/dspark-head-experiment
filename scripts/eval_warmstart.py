"""Eval warm-started head: load released baseline, swap head, load trained additions."""
import os
import sys
import json
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", os.environ.get("EVAL_PORT", "29800"))
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

BASELINE_CKPT = "deepseek-ai/dspark_qwen3_4b_block7"
HEAD_TYPE = os.environ.get("HEAD_TYPE", "rsmh")
ADDITIONS_PATH = os.environ.get("ADDITIONS_PATH", "")
EVAL_NAME = os.environ.get("EVAL_NAME", "warmstart")
EVAL_PROMPTS = int(os.environ.get("EVAL_PROMPTS", "100"))

DATASETS = [
    ("gsm8k", EVAL_PROMPTS),
    ("alpaca", EVAL_PROMPTS),
    ("mt-bench", 80),
]


def main():
    from deepspec.eval.dspark import Qwen3DSparkEvaluator
    from deepspec.modeling.dspark.qwen3.modeling import Qwen3DSparkModel

    class EvalArgs:
        target_name_or_path = "Qwen/Qwen3-4B"
        draft_name_or_path = BASELINE_CKPT
        max_new_tokens = 256
        temperature = 1.0
        confidence_threshold = 0.0
        tensorboard_dir = None
        step = None
        seed = 42
        tasks = DATASETS

    t0 = time.time()
    print(f"[eval] {EVAL_NAME}: head={HEAD_TYPE}, additions={ADDITIONS_PATH}", flush=True)

    evaluator = Qwen3DSparkEvaluator(0, EvalArgs())

    if ADDITIONS_PATH:
        old_head = evaluator.draft_model.markov_head
        if HEAD_TYPE == "rsmh":
            from deepspec.modeling.dspark.rsmh_head import RSMHHead
            new_head = RSMHHead(
                vocab_size=old_head.vocab_size,
                markov_rank=old_head.markov_rank,
                hidden_size=evaluator.draft_model.config.hidden_size,
            ).to(evaluator.device).to(torch.bfloat16)
        elif HEAD_TYPE == "fir":
            from deepspec.modeling.dspark.fir_head import FIRMarkovHead
            new_head = FIRMarkovHead(
                vocab_size=old_head.vocab_size,
                markov_rank=old_head.markov_rank,
                hidden_size=evaluator.draft_model.config.hidden_size,
            ).to(evaluator.device).to(torch.bfloat16)
        else:
            raise ValueError(f"Unknown head type: {HEAD_TYPE}")

        new_head.markov_w1.weight.data.copy_(old_head.markov_w1.weight.data)
        new_head.markov_w2.weight.data.copy_(old_head.markov_w2.weight.data)

        additions = torch.load(ADDITIONS_PATH, map_location=evaluator.device, weights_only=True)
        missing = new_head.load_state_dict(additions, strict=False)
        print(f"[eval] Loaded additions: {len(additions)} tensors, missing={missing.missing_keys[:3]}...", flush=True)

        evaluator.draft_model.markov_head = new_head

    evaluator.evaluate()

    out_dir = os.path.join("outputs", "eval_warmstart")
    os.makedirs(out_dir, exist_ok=True)
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
        "head_type": HEAD_TYPE,
        "additions": ADDITIONS_PATH,
        "elapsed_s": time.time() - t0,
    }
    out_path = os.path.join(out_dir, f"{EVAL_NAME}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] Saved to {out_path}", flush=True)

    evaluator.clean_up()


if __name__ == "__main__":
    main()
