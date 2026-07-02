"""Warm-start RSMH training: load released DSpark baseline, swap head, freeze backbone.

Only trains the 2.4M RSMH additions (update_proj, gate_proj, residual_proj, pos_residual).
Uses DeepSpec's native CacheDataset + loss, and the model's own forward (FlexAttention).

Usage:
  CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29700 RANK=0 WORLD_SIZE=1 \
  PYTHONPATH=. python train_rsmh_warmstart.py
"""
import os
import sys
import time
import json
import math

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from deepspec.modeling.dspark.qwen3.modeling import Qwen3DSparkModel
from deepspec.modeling.dspark.rsmh_head import RSMHHead
from deepspec.modeling.dspark.loss import compute_dspark_loss
from deepspec.data.target_cache_dataset import CacheDataset, CacheCollator
from deepspec.utils import init_dist, seed_all

BASELINE_CKPT = "deepseek-ai/dspark_qwen3_4b_block7"
CACHE_DIR = "/data/deepspec_cache/qwen3_4b_perfectblend_5k"
LR = 3e-4
WARMUP_RATIO = 0.04
EPOCHS = 5
LOCAL_BATCH_SIZE = 1
GRAD_ACCUM = 16
MAX_GRAD_NORM = 1.0
LOG_EVERY = 10
CKPT_EVERY = 200
CKPT_DIR = os.path.expanduser("~/checkpoints/deepspec/rsmh_warmstart")
SEED = 42


def main(local_rank):
    device, global_rank, world_size = init_dist(local_rank)
    seed_all(SEED)
    is_main = (global_rank == 0)

    if is_main:
        print(f"[load] Released baseline: {BASELINE_CKPT}", flush=True)
    model = Qwen3DSparkModel.from_pretrained(
        BASELINE_CKPT, dtype=torch.bfloat16,
        attn_implementation="flex_attention",
    ).to(device)

    old_head = model.markov_head
    rsmh = RSMHHead(
        vocab_size=old_head.vocab_size,
        markov_rank=old_head.markov_rank,
        hidden_size=model.config.hidden_size,
    ).to(device).to(torch.bfloat16)
    rsmh.markov_w1.weight.data.copy_(old_head.markov_w1.weight.data)
    rsmh.markov_w2.weight.data.copy_(old_head.markov_w2.weight.data)
    model.markov_head = rsmh

    for name, param in model.named_parameters():
        param.requires_grad = False
    trainable_names = []
    for name, param in model.markov_head.named_parameters():
        if any(k in name for k in ["update_proj", "gate_proj", "residual_proj", "pos_residual"]):
            param.requires_grad = True
            trainable_names.append(name)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f"[load] Trainable: {n_train/1e6:.2f}M / {n_total/1e6:.1f}M total", flush=True)
        print(f"[load] Trainable params: {trainable_names}", flush=True)

    if is_main:
        print("[compile] torch.compile with dynamic=True...", flush=True)
    model = torch.compile(model, dynamic=True)

    dataset = CacheDataset(CACHE_DIR)
    collator = CacheCollator()
    loader = DataLoader(
        dataset, batch_size=LOCAL_BATCH_SIZE, shuffle=True,
        num_workers=4, collate_fn=collator, pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01,
    )

    steps_per_epoch = math.ceil(len(dataset) / (LOCAL_BATCH_SIZE * GRAD_ACCUM * world_size))
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    def get_lr(step):
        if step < warmup_steps:
            return LR * step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return LR * 0.5 * (1 + math.cos(math.pi * progress))

    if is_main:
        print(f"[train] {len(dataset)} samples, {steps_per_epoch} steps/epoch, {total_steps} total steps", flush=True)
        print(f"[train] LR={LR}, warmup={warmup_steps}, grad_accum={GRAD_ACCUM}", flush=True)
        os.makedirs(CKPT_DIR, exist_ok=True)

    model.train()
    global_step = 0
    micro_step = 0
    running_loss = 0.0
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            batch.pop("attention_mask", None)
            if "input_ids" in batch:
                batch["input_ids"] = batch["input_ids"].long()
            if "loss_mask" in batch:
                batch["loss_mask"] = batch["loss_mask"].long()

            outputs = model(**batch)
            batch_loss = compute_dspark_loss(
                outputs=outputs,
                loss_decay_gamma=4.0,
                ce_loss_alpha=0.1,
                l1_loss_alpha=0.9,
                confidence_head_alpha=1.0,
            )
            loss = batch_loss / GRAD_ACCUM
            loss.backward()
            running_loss += float(batch_loss)
            micro_step += 1

            if micro_step % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    MAX_GRAD_NORM,
                )
                lr = get_lr(global_step)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if is_main and global_step % LOG_EVERY == 0:
                    avg_loss = running_loss / LOG_EVERY
                    elapsed = (time.time() - t0) / 60
                    remaining = elapsed / global_step * (total_steps - global_step)
                    print(
                        f"epoch={epoch} step={global_step}/{total_steps} "
                        f"loss={avg_loss:.4f} lr={lr:.2e} | "
                        f"elapsed={elapsed:.1f}min | remaining={remaining:.1f}min",
                        flush=True,
                    )
                    running_loss = 0.0

                if is_main and global_step % CKPT_EVERY == 0:
                    ckpt_path = os.path.join(CKPT_DIR, f"step_{global_step}")
                    os.makedirs(ckpt_path, exist_ok=True)
                    state = {}
                    for name, param in model.markov_head.named_parameters():
                        if param.requires_grad:
                            state[name] = param.data.cpu()
                    torch.save(state, os.path.join(ckpt_path, "rsmh_additions.pt"))
                    with open(os.path.join(ckpt_path, "train_state.json"), "w") as f:
                        json.dump({"step": global_step, "epoch": epoch, "lr": lr}, f)
                    print(f"[ckpt] Saved to {ckpt_path}", flush=True)

                if global_step >= total_steps:
                    break
        if global_step >= total_steps:
            break

    if is_main:
        final_path = os.path.join(CKPT_DIR, "final")
        os.makedirs(final_path, exist_ok=True)
        state = {}
        for name, param in model.markov_head.named_parameters():
            if param.requires_grad:
                state[name] = param.data.cpu()
        torch.save(state, os.path.join(final_path, "rsmh_additions.pt"))
        print(f"[done] Training complete. {global_step} steps in {(time.time()-t0)/60:.1f}min", flush=True)
        print(f"[done] Final checkpoint: {final_path}", flush=True)

    # Quick eval
    print("[eval] Running quick eval...", flush=True)
    model.requires_grad_(False)

    from deepspec.eval.dspark import Qwen3DSparkEvaluator

    class EvalArgs:
        target_name_or_path = "Qwen/Qwen3-4B"
        draft_name_or_path = BASELINE_CKPT
        max_new_tokens = 256
        temperature = 1.0
        confidence_threshold = 0.0
        tensorboard_dir = None
        step = None
        seed = 42
        tasks = [("gsm8k", 100), ("alpaca", 100)]

    evaluator = Qwen3DSparkEvaluator(local_rank, EvalArgs())
    evaluator.draft_model.markov_head = model.markov_head.to(evaluator.device)
    evaluator.evaluate()
    evaluator.clean_up()
    print("[eval] Compare vs baseline: gsm8k=6.16, alpaca=3.46", flush=True)


if __name__ == "__main__":
    if torch.cuda.device_count() == 1:
        main(0)
    else:
        import torch.multiprocessing
        torch.multiprocessing.spawn(main, nprocs=torch.cuda.device_count())
