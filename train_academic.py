"""
DocROI-Former: Protocol-Aligned Academic Trainer (DocTamper, academic branch)
=============================================================================
Trains on DocTamperV1-TrainingSet ONLY, with the degradation family of the official DTD test
protocol (grayscale + 1-3 JPEG rounds), at native 512x512 resolution, with a quality
curriculum (minq 95 -> 75). Qualities come from our own RNG (src/dataset/doctamper_protocol.py).

Integrity guarantees enforced in code:
  * Only the TrainingSet LMDB is opened. TestingSet / FCD / SCD are never touched here.
  * DTD's official per-image quality tables (pks/*.pk) are never read here.
  * Model selection (best checkpoint) uses the 2,500-image held-out validation split only,
    scored with DTD's metric formulas. Threshold fixed at 0.5.
  * Final test numbers come from evaluate_dtd_protocol.py, run once on the selected checkpoint.

Live progress in the terminal + runs/<run_name>/train.log, metrics.jsonl, checkpoints.

    python train_academic.py --run_name p2_data_fix --init checkpoints/best_docroi_former_pguoctv3.pth
    python train_academic.py --run_name smoke --max_steps_per_epoch 50 --val_samples 200 --epochs 1

Throughput / memory guards (none of them change the model, loss, data or degradations):
  * Train workers are persistent and start from a clean forkserver process, so they are created
    once, not re-forked from the big trainer process at every epoch boundary.
  * Watchdog: if the recent average exceeds --stall_s s/step, or no batch arrives within
    --loader_timeout s, it saves a step-exact checkpoint and rebuilds the workers in-process; if
    that does not help (--max_rebuilds), it exits with code 75 and the launcher restarts and resumes.
  * Step-exact resume: the shuffle is a permutation seeded by (seed, epoch), and mid-epoch
    checkpoints record the next step, so a resume continues where it stopped instead of replaying.
"""
import argparse
import collections
import gc
import json
import logging
import math
import multiprocessing
import os
import sys
import time
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from configs.docroi_config import DocROIConfig  # noqa: E402
from src.dataset.doctamper_protocol import (DocTamperTrainDataset, DocTamperValDataset,  # noqa: E402
                                            heldout_split)
from src.losses.combined_loss import DocROICombinedLoss  # noqa: E402
from src.models.blocks.bayar_conv import BayarConv2d  # noqa: E402
from src.models.docroi_former import DocROIFormer  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", default=datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    ap.add_argument("--train_lmdb", default=os.path.join(ROOT, "data", "Doctamper", "DocTamperV1-TrainingSet"))
    ap.add_argument("--init", default="imagenet", help="'imagenet' or a checkpoint path to fine-tune from")
    ap.add_argument("--resume", default="", help="runs/<name>/latest_docroi_former_<name>.pth to continue an interrupted run")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=6e-5, help="backbone / encoders")
    ap.add_argument("--head_lr", type=float, default=3e-4, help="fusion, decoder, heads")
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--warmup_steps", type=int, default=1000)
    ap.add_argument("--minq_start", type=int, default=95)
    ap.add_argument("--minq_end", type=int, default=75)
    ap.add_argument("--curriculum_frac", type=float, default=0.5, help="fraction of epochs to reach minq_end")
    ap.add_argument("--gray_prob", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_steps_per_epoch", type=int, default=0, help="0 = full epoch (for smoke tests)")
    ap.add_argument("--val_samples", type=int, default=2500)
    ap.add_argument("--save_every_steps", type=int, default=2000)
    ap.add_argument("--log_every", type=int, default=200, help="log loss and s/step every N steps")
    ap.add_argument("--amp", choices=["bf16", "fp16"], default="bf16", help="mixed precision dtype (bf16 is native on Blackwell)")
    ap.add_argument("--channels_last", type=int, default=1, help="1 = channels_last memory format for conv speed")
    ap.add_argument("--compile", type=int, default=0, help="1 = torch.compile the model forward")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dct", choices=["float", "quantized"], default="float",
                    help="frequency DCT stream: 'float' (original) or 'quantized' (final JPEG's own "
                         "coefficients + quantization table); the ONLY difference between the two ablation rows")
    ap.add_argument("--qk_norm", type=int, default=0,
                    help="1 = LayerNorm on per-head queries/keys in the cross-attention (training stability; "
                         "use the same value for every row of an ablation)")
    # ---- throughput / memory guards (no effect on what is learned) ----
    ap.add_argument("--prefetch", type=int, default=2, help="batches prefetched per worker (lower = less host RAM)")
    ap.add_argument("--worker_start", choices=["forkserver", "fork", "spawn"],
                    default="forkserver" if sys.platform.startswith("linux") else "spawn",
                    help="how DataLoader workers are started; forkserver keeps them out of the trainer's memory")
    ap.add_argument("--stall_s", type=float, default=1.0,
                    help="self-heal when the recent average exceeds this many s/step (healthy is ~0.17); 0 = off")
    ap.add_argument("--stall_window", type=int, default=20, help="steps averaged by the slowdown watchdog")
    ap.add_argument("--loader_timeout", type=int, default=300, help="seconds to wait for one batch before self-healing")
    # default 0: in the 2026-11-17 run, rebuilding workers never cured a host-memory freeze; only a full
    # process restart did, so go straight to it (exit 75 -> launcher resumes)
    ap.add_argument("--max_rebuilds", type=int, default=0,
                    help="in-process worker rebuilds per epoch before exiting with code 75 for a full restart")
    ap.add_argument("--auto_resume", type=int, default=0,
                    help="1 = resume from this run's latest_ checkpoint if it exists (used by the launcher's restart loop)")
    return ap.parse_args()


class EpochPermutationSampler(Sampler):
    """Shuffle = permutation seeded by (seed, epoch), so any position in any epoch can be re-created
    exactly; `start` skips the samples a resumed run has already trained on."""

    def __init__(self, n, seed):
        self.n, self.seed, self.epoch, self.start = n, seed, 0, 0

    def set_epoch(self, epoch, start=0):
        self.epoch, self.start = epoch, start

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed * 100003 + self.epoch)
        return iter(torch.randperm(self.n, generator=g)[self.start:].tolist())

    def __len__(self):
        return self.n - self.start


def worker_context(args):
    if args.workers <= 0:
        return None
    ctx = multiprocessing.get_context(args.worker_start)
    if args.worker_start == "forkserver":
        # The forkserver imports these once (plus this script's sys.path); workers fork from it,
        # not from the trainer, so they never inherit the CUDA / torch.compile memory.
        ctx.set_forkserver_preload(["numpy", "torch", "cv2", "lmdb", "PIL.Image", "src.dataset.doctamper_protocol"])
    return ctx


def loader_failure(e):
    """DataLoader errors the watchdog recovers from: batch timeout, or a worker that died / was killed."""
    msg = str(e)
    return "timed out" in msg or "DataLoader worker" in msg


def close_loader(loader):
    """Stop a (persistent) DataLoader's worker processes now, not whenever it is garbage-collected."""
    it = getattr(loader, "_iterator", None)
    if it is not None and hasattr(it, "_shutdown_workers"):
        it._shutdown_workers()
    loader._iterator = None


def mem_report():
    """Host-RAM snapshot for the log (Linux): available RAM, swap in use, RSS of trainer + descendants
    (RSS counts shared pages once per process, so the children figure is an upper bound)."""
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                key, val = line.split(":", 1)
                info[key] = int(val.split()[0]) / 2**20  # kB -> GiB
        parent, rss = {}, {}
        for pid in filter(str.isdigit, os.listdir("/proc")):
            try:
                with open(f"/proc/{pid}/status") as f:
                    st = f.read()
            except OSError:
                continue
            parent[int(pid)] = int(st.split("PPid:")[1].split()[0])
            rss[int(pid)] = int(st.split("VmRSS:")[1].split()[0]) / 2**20 if "VmRSS:" in st else 0.0
        me, kids, frontier = os.getpid(), set(), {os.getpid()}
        while frontier:
            frontier = {p for p, pp in parent.items() if pp in frontier and p not in kids}
            kids |= frontier
        return (f"RAM available {info['MemAvailable']:.1f}/{info['MemTotal']:.1f} GiB | swap used "
                f"{info['SwapTotal'] - info['SwapFree']:.1f} GiB | trainer RSS {rss.get(me, 0):.1f} GiB + "
                f"{len(kids)} child processes {sum(rss.get(p, 0) for p in kids):.1f} GiB")
    except Exception:
        return "memory n/a"


def jpeg_kwargs(b, device):
    """Final-JPEG inputs for the quantized-DCT stream, if the batch carries them."""
    if "jpeg_y" not in b:
        return {}
    return {"jpeg_y": b["jpeg_y"].to(device, non_blocking=True), "qtable": b["qtable"].to(device, non_blocking=True)}


def minq_for_epoch(epoch, args):
    reach = max(1, int(round(args.epochs * args.curriculum_frac)))
    if epoch >= reach:
        return args.minq_end
    return int(round(args.minq_start + (args.minq_end - args.minq_start) * epoch / reach))


@torch.no_grad()
def validate(model, loader, device, amp_dtype=torch.bfloat16, channels_last=False):
    """DTD metric formulas: P, R averaged per image; F = 2PR/(P+R); IoU accumulated over pixels."""
    model.eval()
    p_sum = r_sum = 0.0
    tp_all = fp_all = fn_all = 0
    n = 0
    for b in loader:
        x = b["image"].to(device, non_blocking=True)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        pr = b["entity_prior"].to(device, non_blocking=True)
        gt = b["mask"].to(device)[:, 0].bool()
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device == "cuda"):
            out = model(x, entity_prior=pr, **jpeg_kwargs(b, device))
        prob = torch.sigmoid(out["mask_logits"].float())
        if prob.shape[-2:] != gt.shape[-2:]:
            prob = F.interpolate(prob, size=gt.shape[-2:], mode="bilinear", align_corners=False)
        pred = prob[:, 0] > 0.5
        tp = (pred & gt).sum((1, 2)).double()
        fp = (pred & ~gt).sum((1, 2)).double()
        fn = (~pred & gt).sum((1, 2)).double()
        p_sum += (tp / (tp + fp + 1e-8)).sum().item()
        r_sum += (tp / (tp + fn).clamp(min=1)).sum().item()
        tp_all += int(tp.sum()); fp_all += int(fp.sum()); fn_all += int(fn.sum())
        n += x.size(0)
    P, R = p_sum / max(n, 1), r_sum / max(n, 1)
    return {"val_P": round(100 * P, 2), "val_R": round(100 * R, 2),
            "val_F": round(100 * 2 * P * R / (P + R + 1e-8), 2),
            "val_IoU": round(100 * tp_all / max(tp_all + fp_all + fn_all, 1), 2), "val_n": n}


def write_provenance(run_dir, tag=""):
    """Audit trail: exact code (hashes + snapshot), git state, command line and environment.
    Resumes write their own tagged copies, so the original launch's record is never overwritten."""
    import glob, hashlib, platform, subprocess, tarfile
    files = sorted([os.path.join(ROOT, "train_academic.py"), os.path.join(ROOT, "evaluate_dtd_protocol.py")]
                   + glob.glob(os.path.join(ROOT, "src", "**", "*.py"), recursive=True)
                   + glob.glob(os.path.join(ROOT, "configs", "*.py")))
    hashes = {os.path.relpath(f, ROOT).replace("\\", "/"): hashlib.sha256(open(f, "rb").read()).hexdigest()
              for f in files if os.path.isfile(f)}
    with tarfile.open(os.path.join(run_dir, f"code_snapshot{tag}.tar.gz"), "w:gz") as tar:
        for rel in hashes:
            tar.add(os.path.join(ROOT, rel), arcname=rel)

    def git(*a):
        try:
            return subprocess.run(["git", "-c", f"safe.directory={ROOT}", "-C", ROOT, *a],
                                  capture_output=True, text=True, timeout=60).stdout.strip()
        except Exception as e:  # git missing or not a repo: record the failure, never block training
            return f"unavailable: {e}"
    prov = {"command": " ".join([sys.executable] + sys.argv), "started": datetime.now().isoformat(),
            "host": platform.node(), "platform": platform.platform(), "python": sys.version,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"), "git_head": git("rev-parse", "HEAD"),
            "git_status": git("status", "--porcelain"), "source_sha256": hashes}
    json.dump(prov, open(os.path.join(run_dir, f"provenance{tag}.json"), "w"), indent=2)


def main():
    args = parse_args()
    run_dir = os.path.join(ROOT, "runs", args.run_name)
    # same naming as the production checkpoints/ folder, but kept under runs/ (academic models only)
    # the quantized-DCT model gets its own name so its checkpoints can never be confused with the
    # float-DCT baseline's (baseline names are unchanged, so existing runs still resume)
    model_tag = "docroi_former_qdct" if args.dct == "quantized" else "docroi_former"
    best_path = os.path.join(run_dir, f"best_{model_tag}_{args.run_name}.pth")
    last_path = os.path.join(run_dir, f"latest_{model_tag}_{args.run_name}.pth")
    if args.auto_resume:
        import glob
        # this run's newest latest_ checkpoint, whatever model tag it was saved under (a wrong-type
        # checkpoint cannot slip through: the state_dict load below is strict)
        cands = sorted(glob.glob(os.path.join(run_dir, f"latest_*_{args.run_name}.pth")), key=os.path.getmtime)
        if cands:
            args.resume = cands[-1]
        elif os.path.exists(os.path.join(run_dir, "train.log")):
            sys.exit(f"--auto_resume: no latest_*_{args.run_name}.pth checkpoint in {run_dir}; nothing to resume.")
    if os.path.exists(os.path.join(run_dir, "train.log")) and not args.resume:
        sys.exit(f"Run folder {run_dir} already has a train.log. Use --resume <latest_...pth> to continue it, "
                 f"or pick a new --run_name (audit: one folder = one run).")
    os.makedirs(run_dir, exist_ok=True)
    tag = "_resume_" + datetime.now().strftime("%Y%m%d_%H%M%S") if args.resume else ""
    write_provenance(run_dir, tag)
    log = logging.getLogger("train"); log.setLevel(logging.INFO); log.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.FileHandler(os.path.join(run_dir, "train.log"), encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt); log.addHandler(h)
    json.dump(vars(args), open(os.path.join(run_dir, f"args{tag}.json"), "w"), indent=2)

    assert "TrainingSet" in os.path.basename(os.path.normpath(args.train_lmdb)), \
        "Integrity guard: the academic trainer may only read DocTamperV1-TrainingSet"
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Blackwell / Ampere+ throughput settings (do not change the model's mathematics)
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    log.info(f"Model: {'DocROI-Former-QDCT (quantized JPEG-DCT stream)' if args.dct == 'quantized' else 'DocROI-Former (float DCT stream)'}")
    log.info(f"Run {run_dir} | device {device} {torch.cuda.get_device_name(0) if device == 'cuda' else ''}")
    log.info(f"Args {vars(args)}")

    # ---- data -------------------------------------------------------------------------
    import lmdb
    env = lmdb.open(args.train_lmdb, readonly=True, lock=False)
    with env.begin() as t:
        n_total = int(t.get(b"num-samples"))
    env.close()
    train_idx, val_idx = heldout_split(n_total)
    use_q = args.dct == "quantized"
    train_ds = DocTamperTrainDataset(args.train_lmdb, train_idx, args.resolution, minq=args.minq_start,
                                     gray_prob=args.gray_prob, seed=args.seed, jpeg_info=use_q)
    val_ds = DocTamperValDataset(args.train_lmdb, val_idx[:args.val_samples], args.resolution, jpeg_info=use_q)
    json.dump({str(k): v for k, v in val_ds.table.items()}, open(os.path.join(run_dir, "val_quality_table.json"), "w"))
    log.info(f"TrainingSet {n_total} -> train {len(train_ds)} | held-out val {len(val_ds)} (seed-42 split, own quality table)")
    val_bs = max(16, args.batch_size)
    mp_ctx = worker_context(args)
    val_loader = DataLoader(val_ds, batch_size=val_bs, shuffle=False, num_workers=min(4, args.workers),
                            pin_memory=True, multiprocessing_context=mp_ctx if args.workers > 0 else None)

    # ---- model ------------------------------------------------------------------------
    import dataclasses
    cfg = DocROIConfig()
    cfg = dataclasses.replace(cfg, spatial=dataclasses.replace(
        cfg.spatial, input_resolution=(args.resolution, args.resolution), pretrained=True),
        frequency=dataclasses.replace(cfg.frequency, dct_mode=args.dct),
        fusion=dataclasses.replace(cfg.fusion, qk_norm=bool(args.qk_norm)))
    model = DocROIFormer(config=cfg).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    if args.init != "imagenet" and not args.resume:
        ck = torch.load(args.init, map_location=device, weights_only=False)
        msg = model.load_state_dict(ck.get("model_state_dict", ck), strict=False)
        log.info(f"Init from {args.init}: missing {len(msg.missing_keys)} unexpected {len(msg.unexpected_keys)}")
    else:
        log.info("Init: ImageNet-pretrained PVTv2-B2 backbone, other modules random")

    # Bayar constrained convs are kept on their constraint set by projection after every update
    # (Bayar & Stamm); without it the normaliser drifts to ~0 and gradients explode (run p2_..._s0
    # went NaN at epoch 2). Biases, norm parameters and Bayar kernels get no weight decay.
    bayar_mods = [m for m in model.modules() if isinstance(m, BayarConv2d)]
    for m in bayar_mods:
        m.project_()
    no_decay_ids = {id(m.weight) for m in bayar_mods}
    no_decay_ids |= {id(p) for p in model.parameters() if p.ndim <= 1}
    enc_ids = {id(p) for p in list(model.spatial_encoder.parameters()) + list(model.frequency_encoder.parameters())}
    groups = []
    for is_enc, lr in ((True, args.lr), (False, args.head_lr)):
        ps = [p for p in model.parameters() if (id(p) in enc_ids) == is_enc]
        groups.append({"params": [p for p in ps if id(p) not in no_decay_ids], "lr": lr, "weight_decay": args.weight_decay})
        groups.append({"params": [p for p in ps if id(p) in no_decay_ids], "lr": lr, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, fused=(device == "cuda"))
    log.info(f"Bayar modules projected: {len(bayar_mods)} | no-decay params: {len(no_decay_ids)}")
    steps_per_epoch = math.ceil(len(train_ds) / args.batch_size)
    if args.max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, args.max_steps_per_epoch)
    total_updates = max(1, args.epochs * steps_per_epoch // args.grad_accum)

    def lr_lambda(u):
        if u < args.warmup_steps:
            return (u + 1) / args.warmup_steps
        prog = (u - args.warmup_steps) / max(1, total_updates - args.warmup_steps)
        return max(0.02, 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and args.amp == "fp16"))  # bf16 needs no scaling
    fwd = torch.compile(model) if args.compile else model  # state_dicts are always saved from `model`
    criterion = DocROICombinedLoss(config=cfg.loss).to(device)

    upe = steps_per_epoch // args.grad_accum           # optimizer updates per epoch
    save_every = (math.ceil(args.save_every_steps / args.grad_accum) * args.grad_accum
                  if args.save_every_steps else 0)    # always on an update boundary
    start_epoch, start_step, best_F, update, resumed_stats = 0, 0, -1.0, 0, None
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"]); opt.load_state_dict(ck["optimizer_state_dict"])
        sched.load_state_dict(ck["scheduler_state_dict"]); scaler.load_state_dict(ck["scaler_state_dict"])
        start_epoch, best_F, update = ck["epoch"], ck["best_F"], ck["update"]
        if "step" in ck:
            start_step = int(ck["step"])
        else:  # checkpoint from the earlier trainer: infer the position from the update count
            start_step = max(0, update - start_epoch * upe) * args.grad_accum
            if start_step:
                log.warning(f"Old-format mid-epoch checkpoint: continuing epoch {start_epoch + 1} at step {start_step} "
                            f"(from {update} updates); the rest of this epoch uses the new seeded shuffle.")
        if ck.get("rng_cpu") is not None:
            torch.set_rng_state(ck["rng_cpu"].cpu())
        if device == "cuda" and ck.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in ck["rng_cuda"]])
        resumed_stats = ck.get("run_stats") if start_step else None
        del ck
        log.info(f"Resumed {args.resume} at epoch {start_epoch} step {start_step} (best val F {best_F})")

    def save(path, epoch, step=0, stats=None, extra=None):
        """epoch/step = where training continues from. Written to a temp file and renamed, so a crash
        or OOM-kill during the write can never leave a corrupt latest_/best_ checkpoint."""
        bad = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
        if bad:  # never overwrite a good checkpoint with a diverged model
            log.error(f"Non-finite weights ({len(bad)} tensors, e.g. {bad[:3]}); NOT saving {path}. Stopping.")
            raise SystemExit(2)
        torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict(),
                    "scheduler_state_dict": sched.state_dict(), "scaler_state_dict": scaler.state_dict(),
                    "epoch": epoch, "step": step, "best_F": best_F, "update": update, "args": vars(args),
                    "sampler": "epoch_perm_v1", "rng_cpu": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state_all() if device == "cuda" else None,
                    "run_stats": stats, **(extra or {})}, path + ".tmp")
        os.replace(path + ".tmp", path)

    # ---- data loader: persistent workers + seeded, resumable shuffle -------------------
    sampler = EpochPermutationSampler(len(train_ds), args.seed)

    def make_loader():
        return DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
                          pin_memory=True, drop_last=True, persistent_workers=args.workers > 0,
                          prefetch_factor=args.prefetch if args.workers > 0 else None,
                          multiprocessing_context=mp_ctx,
                          timeout=args.loader_timeout if args.workers > 0 else 0)
    loader = make_loader()
    if args.worker_start == "fork":
        gc.collect(); gc.freeze()  # forked workers then do not copy the parent's objects via GC writes

    def self_heal(reason, epoch, loader_step, ckpt_step, stats):
        """Save a step-exact checkpoint, then replace the DataLoader workers (what a manual Ctrl+C +
        --resume did). After --max_rebuilds in one epoch, exit 75 so the launcher restarts the process."""
        nonlocal loader, rebuilds
        rebuilds += 1
        log.warning(f"  SLOWDOWN at ep{epoch + 1} step {loader_step}: {reason} | {mem_report()}")
        save(last_path, epoch, ckpt_step, stats)
        if rebuilds > args.max_rebuilds:
            log.error(f"  Still slow after {args.max_rebuilds} worker rebuilds this epoch. Exiting with code 75; "
                      f"the launcher resumes from {last_path} at step {ckpt_step}.")
            raise SystemExit(75)
        close_loader(loader)
        gc.collect()
        loader = make_loader()
        sampler.set_epoch(epoch, loader_step * args.batch_size)
        log.info(f"  Rebuilt DataLoader workers ({rebuilds}/{args.max_rebuilds}); continuing at step {loader_step}")
        return iter(loader)

    # ---- loop -------------------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        first = start_step if epoch == start_epoch else 0
        train_ds.minq, train_ds.epoch = minq_for_epoch(epoch, args), epoch   # shared with live workers
        sampler.set_epoch(epoch, first * args.batch_size)
        log.info(f"=== Epoch {epoch + 1}/{args.epochs} | train minq {train_ds.minq} | steps {steps_per_epoch}"
                 + (f" | resuming at step {first}" if first else "") + " ===")
        log.info(f"  {mem_report()}")
        model.train(); opt.zero_grad(set_to_none=True)
        st = resumed_stats or {"run": {"total": 0.0, "dice": 0.0, "focal": 0.0}, "k": 0, "skipped": 0,
                               "gn_sum": 0.0, "gn_n": 0}
        resumed_stats = None
        run, k, skipped, gn_sum, gn_n = st["run"], st["k"], st["skipped"], st["gn_sum"], st["gn_n"]
        n_local, wait_sum, t0 = 0, 0.0, time.time()   # timing covers this process only
        window_bad, consec_bad = False, 0
        rebuilds, since_start, win_start = 0, 0, first
        recent = collections.deque(maxlen=max(1, args.stall_window))
        bar = tqdm(total=steps_per_epoch, initial=first, desc=f"ep{epoch + 1}", dynamic_ncols=True, file=sys.stdout)
        step = first
        try:
            it = iter(loader)
        except RuntimeError as e:
            if not loader_failure(e):
                raise
            it = self_heal(f"workers did not restart: {e}", epoch, step, step, st)
        while step < steps_per_epoch:
            t_step = time.time()
            try:
                b = next(it)
            except StopIteration:
                break
            except RuntimeError as e:  # stuck (timeout) or a worker died (e.g. OOM-killed)
                if not loader_failure(e):
                    raise
                stats = {"run": dict(run), "k": k, "skipped": skipped, "gn_sum": gn_sum, "gn_n": gn_n}
                # partial accumulation-window gradients stay in .grad and are kept in-process;
                # the checkpoint points at the window start so a restart redoes that window
                it = self_heal(str(e), epoch, step, win_start, stats)
                since_start = 0; recent.clear()
                continue
            wait_sum += time.time() - t_step
            x = b["image"].to(device, non_blocking=True)
            if args.channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            tg = {"mask": b["mask"].to(device, non_blocking=True), "label": b["label"].to(device, non_blocking=True)}
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device == "cuda"):
                out = fwd(x, entity_prior=b["entity_prior"].to(device, non_blocking=True), **jpeg_kwargs(b, device))
            ld = criterion({kk: v.float() if torch.is_tensor(v) else v for kk, v in out.items()}, tg)
            loss_val = ld["loss_total"].item()
            if math.isfinite(loss_val):
                scaler.scale(ld["loss_total"] / args.grad_accum).backward()
            else:
                window_bad = True
            boundary = (step + 1) % args.grad_accum == 0
            if boundary:
                scaler.unscale_(opt)
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if window_bad or not torch.isfinite(gnorm):
                    # skip the whole accumulation window; never let a NaN reach the weights
                    opt.zero_grad(set_to_none=True); scaler.update()
                    skipped += 1; consec_bad += 1
                    log.warning(f"  skipped update at ep{epoch + 1} step {step}: loss {loss_val} grad-norm {gnorm.item()} "
                                f"(consecutive {consec_bad}, total {skipped})")
                    if consec_bad >= 20:
                        log.error("20 consecutive non-finite updates: training diverged. Stopping; "
                                  "the latest_/best_ checkpoints keep the last good weights.")
                        raise SystemExit(3)
                else:
                    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step(); update += 1
                    for m in bayar_mods:
                        m.project_()
                    consec_bad = 0; gn_sum += gnorm.item(); gn_n += 1
                window_bad = False
            if math.isfinite(loss_val):
                for kk, src in (("total", "loss_total"), ("dice", "loss_dice"), ("focal", "loss_focal")):
                    run[kk] += ld[src].item()
                k += 1
            n_local += 1
            bar.update(1)
            if step % 20 == 0:
                bar.set_postfix(loss=f"{run['total'] / max(k, 1):.4f}", dice=f"{run['dice'] / max(k, 1):.4f}",
                                lr=f"{opt.param_groups[2]['lr']:.2e}", skip=skipped)
            if step % args.log_every == 0:
                log.info(f"  ep{epoch + 1} step {step}/{steps_per_epoch} | loss {run['total'] / max(k, 1):.4f} "
                         f"dice {run['dice'] / max(k, 1):.4f} focal {run['focal'] / max(k, 1):.4f} | "
                         f"grad-norm(pre-clip, avg) {gn_sum / max(gn_n, 1):.3f} | skipped {skipped} | "
                         f"{(time.time() - t0) / n_local:.3f}s/step (data wait {wait_sum / n_local:.3f}) | peak GPU mem "
                         f"{(torch.cuda.max_memory_allocated() / 2**30) if device == 'cuda' else 0:.1f} GiB")
            step += 1
            recent.append(time.time() - t_step); since_start += 1
            if boundary:
                win_start = step
                stats = {"run": dict(run), "k": k, "skipped": skipped, "gn_sum": gn_sum, "gn_n": gn_n}
                if save_every and step % save_every == 0 and step < steps_per_epoch:
                    save(last_path, epoch, step, stats)
                avg = sum(recent) / len(recent)
                # warm-up of 2 windows after any (re)start covers worker start-up and the first compiled steps
                if (args.stall_s > 0 and step < steps_per_epoch and since_start >= 2 * recent.maxlen
                        and avg > args.stall_s):
                    it = self_heal(f"average {avg:.2f}s/step over the last {len(recent)} steps "
                                   f"(threshold {args.stall_s})", epoch, step, step, stats)
                    since_start = 0; recent.clear()
        bar.close()

        vm = validate(fwd, val_loader, device, amp_dtype, bool(args.channels_last))
        model.train()
        rec = {"epoch": epoch + 1, "minq": train_ds.minq, "train_loss": round(run["total"] / max(k, 1), 4),
               "epoch_time_s": round(time.time() - t0, 1), **vm}
        if first:
            rec["resumed_at_step"] = first   # epoch_time_s then covers only this process's part
        with open(os.path.join(run_dir, "metrics.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        log.info(f"*** Epoch {epoch + 1} | val P {vm['val_P']} R {vm['val_R']} F {vm['val_F']} IoU {vm['val_IoU']} "
                 f"| train loss {rec['train_loss']} | {rec['epoch_time_s']}s | {mem_report()}")
        if vm["val_F"] > best_F:
            best_F = vm["val_F"]
            save(best_path, epoch + 1, 0, extra={"val": vm})
            log.info(f"    new best val F {best_F} -> {best_path}")
        save(last_path, epoch + 1, 0)
    close_loader(loader)
    log.info(f"Done. Best held-out val F {best_F}. Test once with: "
             f"python evaluate_dtd_protocol.py --checkpoint {best_path} --infer_size 0")


if __name__ == "__main__":
    main()
