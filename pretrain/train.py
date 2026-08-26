"""Fresh causal-LM pretraining for SHADOW 250M on compressed Dolma shards."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

for key, value in {
    "SHADOW_D": "1536",
    "SHADOW_NL": "10",
    "SHADOW_NH": "24",
    "SHADOW_NKV": "2",
    "SHADOW_HD": "64",
    "SHADOW_FFNH": "4224",
    "SHADOW_FAST_ATTN": "1",
    "SHADOW_KV_BITS": "1",
    "SHADOW_KV_TWO_TIER": "1",
}.items():
    os.environ.setdefault(key, value)

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT / "finetune"),str(ROOT / "finetune" / "modeling")]

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import common  # noqa: E402
import arm_qat  # noqa: E402
from model_250m import Shadow250M  # noqa: E402
from pretrain.data import DolmaPacker, split_shards  # noqa: E402
from pretrain.diagnostics import gradient_participation, stability_diagnostics  # noqa: E402


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("train", "benchmark", "scan"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("pretrain_runs/dolma-8b"))
    parser.add_argument("--table", type=Path, default=ROOT / "deployment/fp131072.npy")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizer/tokenizer.model")
    parser.add_argument("--remap", type=Path, default=ROOT / "tokenizer/new2old.u32")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device",choices=("auto","cuda","cpu"),default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--ctx", type=int, default=2048)
    parser.add_argument("--micro-batch", type=int, default=12)
    parser.add_argument("--accum", type=int, default=16)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--chunk-docs", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=8_000_000_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-frac", type=float, default=0.01)
    parser.add_argument("--amp-dtype",choices=("bf16","fp16","fp32"),default="bf16")
    parser.add_argument("--ffn-weight-dtype",choices=("ternary","int4_row"),default="ternary")
    parser.add_argument("--ffn-act-qat",action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument("--ffn-act-warmup-tokens",type=int,default=100_000_000)
    parser.add_argument("--mtp-horizon",type=int,choices=(1,2),default=2,
                        help="total prediction horizon including the normal next-token head")
    parser.add_argument("--mtp-loss-weight",type=float,default=0.3,
                        help="weight for each auxiliary future-token loss")
    parser.add_argument("--val-every", type=int, default=100_000_000)
    parser.add_argument("--checkpoint-every", type=int, default=500_000_000)
    parser.add_argument("--val-batches", type=int, default=8)
    parser.add_argument("--benchmark-steps", type=int, default=50)
    parser.add_argument("--scan-max-docs", type=int)
    parser.add_argument(
        "--diagnostics-every", type=int, default=10,
        help="write compact gradient/QAT diagnostics every N updates; 0 disables",
    )
    return parser.parse_args()


def qat_config(args):
    return {**arm_qat.contract(args.ffn_act_qat,args.ffn_weight_dtype),
            "ffn_act_warmup_tokens":args.ffn_act_warmup_tokens,
            "training_amp_dtype":args.amp_dtype,"parameter_dtype":"float32",
            "mtp_horizon":args.mtp_horizon,"mtp_loss_weight":args.mtp_loss_weight}


def training_contract(args):
    return {"ctx":args.ctx,"micro_batch":args.micro_batch,"accum":args.accum,
            "seed":args.seed,"qat":arm_qat.checkpoint_contract(qat_config(args)),
            "ffn_act_warmup_tokens":args.ffn_act_warmup_tokens,"amp_dtype":args.amp_dtype,
            "mtp_horizon":args.mtp_horizon,"mtp_loss_weight":args.mtp_loss_weight}


def corpus_files(directory):
    files = sorted(directory.glob("*.json.gz")) + sorted(directory.glob("*.jsonl.gz"))
    files = sorted(set(files))
    if len(files) < 2:
        raise SystemExit(f"need at least two compressed Dolma shards in {directory}")
    return files


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def corpus_identity(files, tokenizer, remap, seed):
    payload = {
        "seed": seed,
        "files": [(path.name, path.stat().st_size) for path in files],
        "tokenizer_sha256": file_hash(tokenizer),
        "remap_sha256": file_hash(remap),
    }
    encoded = json.dumps(payload, sort_keys=True).encode()
    payload["sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def make_packer(args, shards, seed=None, workers=None):
    return DolmaPacker(
        shards, args.tokenizer, args.remap, context=args.ctx,
        workers=args.workers if workers is None else workers,
        chunk_docs=args.chunk_docs, seed=args.seed if seed is None else seed,
    )


def make_model(args, device):
    strength=arm_qat.activation_qat_strength(0,args.ffn_act_warmup_tokens,args.ffn_act_qat)
    arm_qat.configure(args.ffn_act_qat,args.ffn_weight_dtype,strength)
    packed = np.unpackbits(np.load(args.table), axis=1)[:, :512]
    centroids = torch.tensor(packed.astype(np.float32) * 2 - 1, device=device)
    torch.manual_seed(args.seed)
    model = Shadow250M(
        centroids,F.normalize(centroids,dim=-1),centroids.shape[0],mtp_horizon=args.mtp_horizon
    ).to(device)
    common.requant(model)
    return model


def causal_loss(model,inputs,targets,mtp_loss_weight=0.0):
    hidden, _ = model.trunk(inputs)
    return model.language_model_loss(hidden,targets,mtp_loss_weight,chunk=2048,
                                     conditioning_ids=inputs[:,1:])


def learning_rate(args, consumed):
    progress = min(1.0, consumed / max(1, args.max_tokens))
    if progress < args.warmup_frac:
        return args.lr * progress / max(args.warmup_frac, 1e-12)
    decay = (progress - args.warmup_frac) / (1 - args.warmup_frac)
    return args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * decay))


def atomic_torch_save(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def atomic_hardlink(source, destination):
    """Give an existing checkpoint a final name without duplicating its disk blocks."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    os.link(source, temporary)
    temporary.replace(destination)


def checkpoint(args,model,optimizer,scaler,packer,identity,update,consumed,path):
    device=next(model.parameters()).device
    atomic_torch_save({
        "version": 1,
        "cfg": {"V":model.V,**qat_config(args)},
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "data": packer.state_dict(),
        "corpus_sha256": identity["sha256"],
        "update": update,
        "consumed_tokens": consumed,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if device.type=="cuda" else [],
        "args": vars(args),
        "training_geometry": training_contract(args),
    }, path)


@torch.no_grad()
def validate(args, model, shards, device):
    packer = make_packer(args, shards, seed=args.seed + 1, workers=max(1, args.workers // 2))
    old_strength=common.FFN_ACT_QAT_STRENGTH
    common.set_ffn_activation_qat_strength(1.0 if args.ffn_act_qat else 0.0); model.eval()
    total = 0.0
    count = 0
    dtype,enabled,_=arm_qat.autocast_and_scaler(device,args.amp_dtype)
    try:
        for _ in range(args.val_batches):
            inputs, targets = packer.next_batch(args.micro_batch)
            with torch.autocast(device.type,dtype=dtype,enabled=enabled):
                total += float(causal_loss(model,inputs.to(device),targets.to(device),args.mtp_loss_weight))
            count += 1
    except StopIteration:
        pass
    finally:
        packer.close()
        model.train(); common.set_ffn_activation_qat_strength(old_strength)
    return total / max(1, count)


def run_scan(args, files, identity):
    train, validation = split_shards(files, args.seed)
    stats = {"identity": identity, "splits": {"train": [], "validation": []}}
    remaining = args.scan_max_docs
    for name, shards in (("train", train), ("validation", validation)):
        for shard in shards:
            packer = make_packer(args, [shard], workers=args.workers)
            # A duplicated path lets the reusable packer accept a one-shard scan.
            packer.shards = [str(shard)]
            try:
                while remaining is None or packer.documents < remaining:
                    packer.next_window()
            except StopIteration:
                pass
            entry = {
                "file": shard.name, "documents": packer.documents,
                "bad_records": packer.bad_records,
                "packed_tokens": packer.consumed_tokens,
                "pending_tokens": len(packer.tokens),
            }
            stats["splits"][name].append(entry)
            print(json.dumps(entry), flush=True)
            packer.close()
            if remaining is not None:
                remaining -= entry["documents"]
                if remaining <= 0:
                    break
        if remaining is not None and remaining <= 0:
            break
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "corpus-scan.json").write_text(json.dumps(stats, indent=2) + "\n")


def run_benchmark(args, train_shards, device):
    packer = make_packer(args, train_shards)
    model = make_model(args, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
    )
    samples = []
    amp_dtype,amp_enabled,scaler=arm_qat.autocast_and_scaler(device,args.amp_dtype)
    for step in range(args.benchmark_steps + 3):
        input_start = time.perf_counter()
        inputs, targets = packer.next_batch(args.micro_batch)
        input_seconds = time.perf_counter() - input_start
        if device.type=="cuda": torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        common.set_ffn_activation_qat_strength(1.0 if args.ffn_act_qat else 0.0)
        with torch.autocast(device.type,dtype=amp_dtype,enabled=amp_enabled):
            loss=causal_loss(model,inputs.to(device),targets.to(device),args.mtp_loss_weight)
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()
        common.requant(model)
        if device.type=="cuda": torch.cuda.synchronize()
        compute_seconds = time.perf_counter() - start
        if step >= 3:
            samples.append((input_seconds, compute_seconds))
        print(
            f"step={step} input={input_seconds:.3f}s compute={compute_seconds:.3f}s "
            f"loss={loss.item():.4f}", flush=True,
        )
    packer.close()
    input_total = sum(item[0] for item in samples)
    compute_total = sum(item[1] for item in samples)
    tokens = len(samples) * args.micro_batch * args.ctx
    result = {
        "steps": len(samples),
        "tokens": tokens,
        "input_tokens_per_second": tokens / input_total,
        "compute_tokens_per_second": tokens / compute_total,
        "end_to_end_tokens_per_second": tokens / (input_total + compute_total),
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30 if device.type=="cuda" else 0.0,
    }
    result["estimated_days_for_max_tokens"] = (
        args.max_tokens / result["end_to_end_tokens_per_second"] / 86400
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def run_train(args, train_shards, val_shards, identity, device):
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "corpus.json").write_text(json.dumps(identity, indent=2) + "\n")
    packer = make_packer(args, train_shards)
    model = make_model(args, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
    )
    update = consumed = 0
    amp_dtype,amp_enabled,scaler=arm_qat.autocast_and_scaler(device,args.amp_dtype)
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        if saved["corpus_sha256"] != identity["sha256"]:
            raise SystemExit("checkpoint corpus identity does not match current data")
        expected_geometry = training_contract(args)
        if saved.get("training_geometry") != expected_geometry:
            raise SystemExit(
                f"checkpoint training geometry does not match: "
                f"{saved.get('training_geometry')} != {expected_geometry}"
            )
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved.get("scaler",{}))
        # RVQ's reconstructed weights are an ephemeral cache, not part of state_dict.
        common.requant(model)
        packer.load_state_dict(saved["data"])
        update, consumed = saved["update"], saved["consumed_tokens"]
        random.setstate(saved["python_rng"]); np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        if device.type=="cuda" and saved["cuda_rng"]:
            torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])
    next_val = (consumed // args.val_every + 1) * args.val_every
    next_checkpoint = (consumed // args.checkpoint_every + 1) * args.checkpoint_every
    last_checkpoint = None
    last_checkpoint_tokens = None
    started = time.time()
    log_path = args.out / "metrics.jsonl"
    diagnostics_path = args.out / "diagnostics.jsonl"
    model.train()
    while consumed < args.max_tokens:
        common.set_ffn_activation_qat_strength(arm_qat.activation_qat_strength(
            consumed,args.ffn_act_warmup_tokens,args.ffn_act_qat))
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        actual_accum = 0
        try:
            for _ in range(args.accum):
                if consumed + (actual_accum + 1) * args.micro_batch * args.ctx > args.max_tokens:
                    break
                inputs, targets = packer.next_batch(args.micro_batch)
                with torch.autocast(device.type,dtype=amp_dtype,enabled=amp_enabled):
                    loss=causal_loss(model,inputs.to(device),targets.to(device),args.mtp_loss_weight)/args.accum
                scaler.scale(loss).backward(); loss_sum += float(loss.detach()) * args.accum; actual_accum += 1
        except StopIteration:
            if not actual_accum:
                break
        if not actual_accum:
            break
        if actual_accum != args.accum:
            # Preserve the mean gradient for the final partial update.
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(args.accum / actual_accum)
        batch_tokens = actual_accum * args.micro_batch * args.ctx
        consumed += batch_tokens; update += 1
        lr = learning_rate(args, consumed)
        for group in optimizer.param_groups:
            group["lr"] = lr
        collect_diagnostics = args.diagnostics_every > 0 and update % args.diagnostics_every == 0
        diagnostic_started = time.perf_counter() if collect_diagnostics else None
        participation = gradient_participation(model) if collect_diagnostics else None
        scaler.unscale_(optimizer)
        total_grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        total_grad_norm = float(total_grad_norm)
        scaler.step(optimizer); scaler.update(); common.requant(model)
        if collect_diagnostics:
            diagnostic = stability_diagnostics(
                model, common.RVQ, update, consumed, gradients=participation,
                started_at=diagnostic_started,weight_dtype=args.ffn_weight_dtype,
            )
            with diagnostics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(diagnostic) + "\n")
        elapsed = time.time() - started
        metrics = {
            "update": update, "tokens": consumed, "loss": loss_sum / actual_accum,
            "lr": lr, "tokens_per_second": max(0, consumed - (0 if not args.resume else saved["consumed_tokens"])) / elapsed,
            "elapsed_seconds": elapsed,
            "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30 if device.type=="cuda" else 0.0,
            "grad_norm_pre_clip": total_grad_norm,
            "grad_clip_coefficient": min(1.0, 1.0 / max(total_grad_norm, 1e-30)),
        }
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics) + "\n")
        print(json.dumps(metrics), flush=True)
        if consumed >= next_val:
            print(json.dumps({"tokens": consumed, "validation_loss": validate(args, model, val_shards, device)}), flush=True)
            next_val += args.val_every
        if consumed >= next_checkpoint:
            path = args.out / "checkpoints" / f"tokens-{consumed:012d}.pt"
            checkpoint(args,model,optimizer,scaler,packer,identity,update,consumed,path)
            last_checkpoint = path
            last_checkpoint_tokens=consumed
            checkpoints = sorted(path.parent.glob("tokens-*.pt"))
            for old in checkpoints[:-3]:
                old.unlink()
            next_checkpoint += args.checkpoint_every
    final = args.out / "checkpoints/final.pt"
    if last_checkpoint is not None and last_checkpoint_tokens==consumed:
        atomic_hardlink(last_checkpoint, final)
    else:
        checkpoint(args,model,optimizer,scaler,packer,identity,update,consumed,final)
    packer.close()
    print(f"complete: {consumed:,} tokens; checkpoint {final}")


def main():
    args = arguments()
    if args.ffn_act_warmup_tokens<0: raise SystemExit("--ffn-act-warmup-tokens must be nonnegative")
    if args.mtp_loss_weight<0: raise SystemExit("--mtp-loss-weight must be nonnegative")
    if args.device=="cuda" and not torch.cuda.is_available(): raise SystemExit("CUDA is unavailable")
    files = corpus_files(args.data)
    identity = corpus_identity(files, args.tokenizer, args.remap, args.seed)
    train_shards, val_shards = split_shards(files, args.seed)
    if args.mode == "scan":
        run_scan(args, files, identity)
        return
    device=torch.device("cuda" if args.device=="cuda" or
                        (args.device=="auto" and torch.cuda.is_available()) else "cpu")
    if args.mode == "benchmark":
        run_benchmark(args, train_shards, device)
    else:
        run_train(args, train_shards, val_shards, identity, device)


if __name__ == "__main__":
    main()
