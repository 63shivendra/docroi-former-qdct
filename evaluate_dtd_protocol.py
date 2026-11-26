"""
DocROI-Former: Official DTD-Protocol Evaluation (TestingSet / FCD / SCD)
========================================================================
Re-implements the evaluation protocol of DTD (Qu et al., CVPR 2023, official repo
models/eval_dtd.py), which FFDN (ECCV 2024) and ADCD-Net (ICCV 2025) also follow:

  * Every test image is converted to grayscale ("L") and then JPEG-compressed 1-3 times
    with the per-image quality factors from DTD's public `pks/DocTamperV1-<split>_<minq>.pk`.
  * Metrics are computed at the native image resolution (512x512 for DocTamper):
      P   = mean over images of |pred & gt| / |pred|
      R   = mean over images of |pred & gt| / |gt|
      F   = 2PR / (P + R)                       (DTD's F, computed from the averaged P and R)
      IoU = tampered-class IoU accumulated over all pixels of the split (DTD's iu[1])
    Also logged: mIoU (mean of background and tampered IoU, the mmseg style used by FFDN),
    and the legacy per-image mean F1 / IoU used by evaluate.py.

The model is run at its training resolution (--infer_size, default 384) and the predicted
probability map is bilinearly upsampled back to native resolution before scoring.

Progress is shown live in the terminal and everything is written to
reports/dtd_protocol/<run_id>/ : eval.log, <split>_per_sample.csv, summary.json.

Usage:
    python evaluate_dtd_protocol.py                       # all three splits that exist
    python evaluate_dtd_protocol.py --datasets TestingSet,SCD --max_samples 200   # smoke test
    python evaluate_dtd_protocol.py --protocol raw        # no compression (old setting), same metrics
"""

import argparse
import csv
import hashlib
import io
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime

import cv2
import lmdb
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from configs.docroi_config import DocROIConfig  # noqa: E402
from src.dataset.doctamper_protocol import jpeg_luma_info  # noqa: E402
from src.dataset.entity_tokenizer import OtsuEntityTokenizer  # noqa: E402
from src.models.docroi_former import DocROIFormer  # noqa: E402

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
SPLITS = ("TestingSet", "FCD", "SCD")


class _NumpyOnlyUnpickler(pickle.Unpickler):
    """DTD's .pk files only hold numpy arrays; refuse anything else (no arbitrary code)."""
    _ALLOWED = {("numpy", "ndarray"), ("numpy", "dtype"),
                ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
                ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar")}

    def find_class(self, module, name):
        if (module, name) in self._ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Blocked global in quality table: {module}.{name}")


def load_quality_table(path):
    with open(path, "rb") as f:
        return _NumpyOnlyUnpickler(f).load()


class DocTamperDTDDataset(Dataset):
    def __init__(self, lmdb_path, quality_table=None, infer_size=384, protocol="dtd", jpeg_info=False):
        self.lmdb_path = lmdb_path
        self.jpeg_info = jpeg_info  # final JPEG's luma + quantization table, for quantized-DCT models
        self.qt = quality_table
        self.infer_size = infer_size
        self.protocol = protocol
        self.env = None
        env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin() as txn:
            self.n = int(txn.get(b"num-samples"))
        env.close()
        self.tokenizer = OtsuEntityTokenizer()

    def __len__(self):
        return self.n

    def _env(self):
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False,
                                 meminit=False, max_readers=64)
        return self.env

    def _dtd_compress(self, im, record):
        # Mirrors eval_dtd.py exactly: convert("L"), then q2 -> q1 -> q (oldest to newest).
        qs = [int(v) for v in np.atleast_1d(record)]
        order = []
        if len(qs) - 1 > 1:
            order.append(qs[-3])
        if len(qs) - 1 > 0:
            order.append(qs[-2])
        order.append(qs[-1])
        im = im.convert("L")
        for q in order:
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=q)
            buf.seek(0)
            im = Image.open(buf)
            im.load()
        return im, order

    def __getitem__(self, idx):
        with self._env().begin(write=False) as txn:
            img_buf = txn.get(b"image-%09d" % idx)
            lbl_buf = txn.get(b"label-%09d" % idx)
        im = Image.open(io.BytesIO(img_buf))
        gt = (cv2.imdecode(np.frombuffer(lbl_buf, np.uint8), 0) != 0).astype(np.uint8)

        if self.protocol == "dtd":
            im, qs = self._dtd_compress(im, self.qt[idx])
        else:
            qs = []
        jpeg = jpeg_luma_info(im) if self.jpeg_info else None   # read before any conversion
        rgb = np.asarray(im.convert("RGB"))
        if rgb.shape[:2] != gt.shape:
            rgb = cv2.resize(rgb, (gt.shape[1], gt.shape[0]))

        x = cv2.resize(rgb, (self.infer_size, self.infer_size), interpolation=cv2.INTER_LINEAR) \
            if self.infer_size > 0 else rgb
        prior = self.tokenizer.extract_entity_mask(x)
        x = (x.astype(np.float32) / 255.0 - _MEAN) / _STD
        extra = {}
        if jpeg is not None:
            if jpeg[0].shape != x.shape[:2]:
                raise ValueError("quantized-DCT model needs native-resolution input (--infer_size 0)")
            extra = {"jpeg_y": torch.from_numpy(np.array(jpeg[0], dtype=np.uint8))[None],
                     "qtable": torch.from_numpy(jpeg[1])}
        return {
            **extra,
            "image": torch.from_numpy(x.transpose(2, 0, 1)).float(),
            "entity_prior": torch.from_numpy(prior).unsqueeze(0).float(),
            "gt": torch.from_numpy(gt),
            "idx": idx,
            "qualities": "-".join(map(str, qs)),
        }


def sha256_file(path, chunk=1 << 22):
    """Fingerprint of a file (checkpoint / quality table) so reported numbers are tied to exact bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def setup_logger(path):
    logger = logging.getLogger("dtd_eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def find_split(data_dir, split):
    for c in (os.path.join(data_dir, "Doctamper", f"DocTamperV1-{split}"),
              os.path.join(data_dir, f"DocTamperV1-{split}")):
        if os.path.isfile(os.path.join(c, "data.mdb")):
            return c
    return None


def summarise(acc, n):
    P = acc["p_sum"] / max(n, 1)
    R = acc["r_sum"] / max(n, 1)
    Fs = 2 * P * R / (P + R + 1e-8)
    iou_t = acc["tp"] / max(acc["tp"] + acc["fp"] + acc["fn"], 1)
    iou_b = acc["tn"] / max(acc["tn"] + acc["fp"] + acc["fn"], 1)
    return {
        "samples": n,
        "precision": round(P * 100, 2),
        "recall": round(R * 100, 2),
        "f1_dtd": round(Fs * 100, 2),
        "iou_tampered": round(iou_t * 100, 2),
        "miou_2class": round((iou_t + iou_b) / 2 * 100, 2),
        "legacy_mean_image_f1": round(acc["f1_img_sum"] / max(n, 1) * 100, 2),
        "legacy_mean_image_iou": round(acc["iou_img_sum"] / max(n, 1) * 100, 2),
    }


@torch.no_grad()
def evaluate_split(model, ds, split, args, device, out_dir, log):
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=(device == "cuda"), persistent_workers=False)
    total = len(ds) if args.max_samples <= 0 else min(len(ds), args.max_samples)
    acc = dict(p_sum=0.0, r_sum=0.0, tp=0, fp=0, fn=0, tn=0, f1_img_sum=0.0, iou_img_sum=0.0)
    n = 0
    gpu_ms = 0.0
    csv_path = os.path.join(out_dir, f"{split}_per_sample.csv")
    fcsv = open(csv_path, "w", newline="", encoding="utf-8")
    w = csv.writer(fcsv)
    w.writerow(["index", "jpeg_qualities", "gt_pixels", "pred_pixels", "tp", "precision", "recall",
                "f1", "iou", "global_prob"])

    log.info(f"===== {split}: {total} samples | protocol={args.protocol} minq={args.minq} "
             f"| infer_size={args.infer_size} | batch={args.batch_size} =====")
    t0 = time.time()
    bar = tqdm(total=total, desc=f"{split:>10}", unit="img", dynamic_ncols=True, file=sys.stdout)
    for bi, b in enumerate(loader):
        if n >= total:
            break
        x = b["image"].to(device, non_blocking=True)
        pr = b["entity_prior"].to(device, non_blocking=True)
        gt = b["gt"].to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        tg = time.perf_counter()
        # same mixed precision as training (bf16); fp16 has a narrower range than the model was trained with
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            kw = ({"jpeg_y": b["jpeg_y"].to(device, non_blocking=True),
                   "qtable": b["qtable"].to(device, non_blocking=True)} if "jpeg_y" in b else {})
            out = model(x, entity_prior=pr, **kw)
        prob = torch.sigmoid(out["mask_logits"].float())
        if prob.shape[-2:] != gt.shape[-2:]:
            prob = F.interpolate(prob, size=gt.shape[-2:], mode="bilinear", align_corners=False)
        if device == "cuda":
            torch.cuda.synchronize()
        gpu_ms += (time.perf_counter() - tg) * 1000

        pred = (prob[:, 0] > args.threshold)
        g = gt.bool()
        tp = (pred & g).sum((1, 2)).double()
        fp = (pred & ~g).sum((1, 2)).double()
        fn = (~pred & g).sum((1, 2)).double()
        tn = (~pred & ~g).sum((1, 2)).double()
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn).clamp(min=1)
        f1 = 2 * tp / (2 * tp + fp + fn).clamp(min=1)
        iou = tp / (tp + fp + fn).clamp(min=1)
        gprob = out["global_prob"].float().view(-1).cpu().numpy()

        k = min(x.size(0), total - n)
        acc["p_sum"] += p[:k].sum().item()
        acc["r_sum"] += r[:k].sum().item()
        acc["f1_img_sum"] += f1[:k].sum().item()
        acc["iou_img_sum"] += iou[:k].sum().item()
        for key, v in (("tp", tp), ("fp", fp), ("fn", fn), ("tn", tn)):
            acc[key] += int(v[:k].sum().item())
        for i in range(k):
            w.writerow([int(b["idx"][i]), b["qualities"][i], int(tp[i] + fn[i]), int(tp[i] + fp[i]),
                        int(tp[i]), round(p[i].item(), 4), round(r[i].item(), 4), round(f1[i].item(), 4),
                        round(iou[i].item(), 4), round(float(gprob[i]), 4)])
        n += k
        bar.update(k)
        s = summarise(acc, n)
        bar.set_postfix(P=s["precision"], R=s["recall"], F=s["f1_dtd"], IoU=s["iou_tampered"])
        if bi % 100 == 0:
            fcsv.flush()
            log.info(f"  [{split}] {n}/{total} | P {s['precision']:.2f} R {s['recall']:.2f} "
                     f"F {s['f1_dtd']:.2f} IoU {s['iou_tampered']:.2f} | {time.time()-t0:.0f}s elapsed")
    bar.close()
    fcsv.close()

    s = summarise(acc, n)
    s.update(split=split, wall_time_s=round(time.time() - t0, 1),
             gpu_ms_per_image=round(gpu_ms / max(n, 1), 2), per_sample_csv=csv_path)
    log.info(f"----- {split} RESULT ({n} samples, {s['wall_time_s']}s) -----")
    log.info(f"  Precision {s['precision']:.2f} | Recall {s['recall']:.2f} | F1 (DTD) {s['f1_dtd']:.2f} "
             f"| IoU (tampered) {s['iou_tampered']:.2f} | mIoU {s['miou_2class']:.2f}")
    log.info(f"  Legacy per-image mean: F1 {s['legacy_mean_image_f1']:.2f} | IoU {s['legacy_mean_image_iou']:.2f} "
             f"| GPU {s['gpu_ms_per_image']:.2f} ms/img (batch {args.batch_size})")
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=os.path.join(ROOT, "checkpoints", "best_docroi_former_pguoctv3.pth"))
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--pks_dir", default=os.path.join(ROOT, "data", "Doctamper", "dtd_official", "pks"))
    ap.add_argument("--datasets", default=",".join(SPLITS))
    ap.add_argument("--protocol", choices=["dtd", "raw"], default="dtd")
    ap.add_argument("--minq", type=int, default=75, choices=[75, 80, 85, 90])
    ap.add_argument("--infer_size", type=int, default=384, help="0 = feed native resolution")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max_samples", type=int, default=0, help="0 = full split")
    ap.add_argument("--out_root", default=os.path.join(ROOT, "reports", "dtd_protocol"))
    args = ap.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.protocol}_q{args.minq}"
    out_dir = os.path.join(args.out_root, run_id)
    os.makedirs(out_dir, exist_ok=True)
    log = setup_logger(os.path.join(out_dir, "eval.log"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Run dir: {out_dir}")
    log.info(f"Device: {device} {torch.cuda.get_device_name(0) if device == 'cuda' else ''} | torch {torch.__version__}")
    log.info(f"Args: {vars(args)}")

    cfg = DocROIConfig()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_res = (ckpt.get("args") or {}).get("resolution") if isinstance(ckpt.get("args"), dict) else None
    if train_res:
        import dataclasses
        cfg = dataclasses.replace(cfg, spatial=dataclasses.replace(cfg.spatial, input_resolution=(train_res, train_res)))
        if args.infer_size not in (0, train_res):
            log.info(f"  NOTE: checkpoint trained at {train_res}; overriding --infer_size {args.infer_size} -> {train_res}")
            args.infer_size = train_res
    dct_mode = (ckpt.get("args") or {}).get("dct", "float") if isinstance(ckpt.get("args"), dict) else "float"
    if dct_mode != "float":
        import dataclasses
        cfg = dataclasses.replace(cfg, frequency=dataclasses.replace(cfg.frequency, dct_mode=dct_mode))
    qk_norm = bool((ckpt.get("args") or {}).get("qk_norm", 0)) if isinstance(ckpt.get("args"), dict) else False
    if qk_norm:
        import dataclasses
        cfg = dataclasses.replace(cfg, fusion=dataclasses.replace(cfg.fusion, qk_norm=True))
    log.info(f"Model resolution: {cfg.spatial.input_resolution} | DCT stream: {dct_mode} | QK-norm: {qk_norm}")
    model = DocROIFormer(config=cfg).to(device)
    # strict for quantized-DCT checkpoints (new code path: any key mismatch is a bug, not legacy drift)
    msg = model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=(dct_mode != "float" or qk_norm))
    log.info(f"Checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch', '?')}) | "
             f"missing keys {len(msg.missing_keys)} | unexpected keys {len(msg.unexpected_keys)}")
    if msg.missing_keys:
        log.info(f"  WARNING missing keys (random init): {msg.missing_keys[:10]}")
    model.eval()

    ckpt_sha = sha256_file(args.checkpoint)
    log.info(f"Checkpoint SHA-256: {ckpt_sha}")
    results = {"run_id": run_id, "args": vars(args), "checkpoint": args.checkpoint,
               "checkpoint_sha256": ckpt_sha, "checkpoint_epoch": ckpt.get("epoch"),
               "torch": torch.__version__, "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
               "splits": {}}
    for split in [s.strip() for s in args.datasets.split(",") if s.strip()]:
        path = find_split(args.data_dir, split)
        if path is None:
            log.info(f"SKIP {split}: no data.mdb found under {args.data_dir} (download DocTamperV1-{split}).")
            results["splits"][split] = {"status": "missing"}
            continue
        qt = None
        if args.protocol == "dtd":
            qt_path = os.path.join(args.pks_dir, f"DocTamperV1-{split}_{args.minq}.pk")
            qt = load_quality_table(qt_path)
            qt_sha = sha256_file(qt_path)
            log.info(f"Quality table: {qt_path} ({len(qt)} entries) | SHA-256 {qt_sha}")
        ds = DocTamperDTDDataset(path, qt, infer_size=args.infer_size, protocol=args.protocol,
                                 jpeg_info=(dct_mode == "quantized"))
        if qt is not None and len(qt) != len(ds):
            log.info(f"  WARNING quality table has {len(qt)} entries but LMDB has {len(ds)} samples")
        results["splits"][split] = evaluate_split(model, ds, split, args, device, out_dir, log)
        results["splits"][split]["quality_table_sha256"] = qt_sha if qt is not None else ""
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    log.info("=" * 96)
    log.info(f"{'Split':<12}{'N':>7}{'P':>8}{'R':>8}{'F1':>8}{'IoU':>8}{'mIoU':>8}{'imgF1':>8}{'imgIoU':>8}")
    for split, s in results["splits"].items():
        if s.get("status") == "missing":
            log.info(f"{split:<12}{'missing':>7}")
            continue
        log.info(f"{split:<12}{s['samples']:>7}{s['precision']:>8.2f}{s['recall']:>8.2f}{s['f1_dtd']:>8.2f}"
                 f"{s['iou_tampered']:>8.2f}{s['miou_2class']:>8.2f}{s['legacy_mean_image_f1']:>8.2f}"
                 f"{s['legacy_mean_image_iou']:>8.2f}")
    log.info("=" * 96)
    # one-row-per-split CSV with everything needed to tie the numbers to the exact weights and protocol
    summary_csv = os.path.join(out_dir, "summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "split", "samples", "precision", "recall", "f1_dtd", "iou_tampered", "miou_2class",
                    "img_f1", "img_iou", "protocol", "minq", "threshold", "infer_size", "gpu_ms_per_image",
                    "checkpoint", "checkpoint_epoch", "checkpoint_sha256", "quality_table_sha256",
                    "device", "torch"])
        for split, s in results["splits"].items():
            if s.get("status") == "missing":
                w.writerow([run_id, split, "missing"]); continue
            w.writerow([run_id, split, s["samples"], s["precision"], s["recall"], s["f1_dtd"], s["iou_tampered"],
                        s["miou_2class"], s["legacy_mean_image_f1"], s["legacy_mean_image_iou"], args.protocol,
                        args.minq, args.threshold, args.infer_size, s["gpu_ms_per_image"], args.checkpoint,
                        results["checkpoint_epoch"], ckpt_sha, s.get("quality_table_sha256", ""),
                        results["device"], results["torch"]])
    log.info(f"Saved: {os.path.join(out_dir, 'summary.json')}")
    log.info(f"Saved: {summary_csv}")
    log.info(f"Per-image CSVs: {', '.join(os.path.join(out_dir, sp + '_per_sample.csv') for sp in results['splits'] if results['splits'][sp].get('status') != 'missing')}")


if __name__ == "__main__":
    main()
