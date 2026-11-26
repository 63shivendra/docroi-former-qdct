"""
DocTamper protocol-aligned data pipeline (academic branch)
==========================================================
Single source of truth for how DocTamper images are degraded, so that training,
validation and the official test evaluation all use the *same* code path.

Official test protocol (DTD, Qu et al. CVPR'23, repo models/eval_dtd.py; followed by
FFDN ECCV'24 and ADCD-Net ICCV'25):
    image -> convert("L") -> JPEG x k rounds (k in {1,2,3}), qualities from the public
    per-image table pks/DocTamperV1-<split>_<minq>.pk -> convert("RGB") -> network.

Training (this module) applies the *same family* of degradations, but every quality factor
is drawn from our own seeded RNG. The official test tables are never read here.
    k ~ U{1,2,3},  q_i ~ U{minq, ..., 100},  minq follows a curriculum 95 -> 75.

Validation = 2,500 images held out from the DocTamper *training* LMDB (seed-42 permutation,
same split as train.py), degraded once with a fixed table generated from our own seed.
"""
import io
import multiprocessing as mp
import os
import random
from typing import Dict, List, Optional, Sequence

import cv2
import lmdb
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.dataset.entity_tokenizer import OtsuEntityTokenizer

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def dtd_order(record: Sequence[int]) -> List[int]:
    """Qualities in the order DTD applies them (record[-3], record[-2], record[-1])."""
    qs = [int(v) for v in np.atleast_1d(record)]
    order = []
    if len(qs) - 1 > 1:
        order.append(qs[-3])
    if len(qs) - 1 > 0:
        order.append(qs[-2])
    order.append(qs[-1])
    return order


def degrade(im: Image.Image, qualities: Sequence[int], gray: bool = True) -> Image.Image:
    """Grayscale (optional) + sequential JPEG re-compression, applied in the given order."""
    im = im.convert("L") if gray else im.convert("RGB")
    for q in qualities:
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=int(q))
        buf.seek(0)
        im = Image.open(buf)
        im.load()
    return im.convert("RGB")


# IJG standard luminance quantization table (JPEG Annex K), natural row-major order. libjpeg
# (used by Pillow) scales it by the quality factor; at quality 50 it is used unscaled.
_IJG_LUMA = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
                      14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
                      18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
                      49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], dtype=np.float32)
_ZIGZAG = np.array([0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48,
                    41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22,
                    15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55,
                    62, 63])  # _ZIGZAG[k] = natural index of the k-th coefficient in zigzag order
_PIL_QT_ORDER = None


def _pil_qtable_order() -> str:
    """Pillow has returned quantization tables in zigzag or natural order depending on version;
    detect it once by encoding at quality 50 (table == IJG base) and comparing."""
    global _PIL_QT_ORDER
    if _PIL_QT_ORDER is None:
        buf = io.BytesIO()
        Image.fromarray(np.full((16, 16), 128, np.uint8)).save(buf, "JPEG", quality=50)
        q = np.asarray(Image.open(buf).quantization[0], dtype=np.float32)
        if np.array_equal(q, _IJG_LUMA):
            _PIL_QT_ORDER = "natural"
        elif np.array_equal(q, _IJG_LUMA[_ZIGZAG]):
            _PIL_QT_ORDER = "zigzag"
        else:
            raise RuntimeError("Could not determine Pillow's JPEG quantization-table order")
    return _PIL_QT_ORDER


def jpeg_luma_info(im: Image.Image):
    """Luma plane (uint8 HxW) and luma quantization table (float32[64], natural row-major order,
    index = 8*vertical_freq + horizontal_freq) of a decoded JPEG, read from the file itself.
    Non-JPEG input gets an all-ones table (= no quantization)."""
    y = np.asarray(im if im.mode == "L" else im.convert("YCbCr").getchannel(0), dtype=np.uint8)
    qt = getattr(im, "quantization", None)
    if not qt:
        return y, np.ones(64, np.float32)
    q = np.asarray(qt[0], dtype=np.float32)
    if _pil_qtable_order() == "zigzag":
        nat = np.empty(64, np.float32)
        nat[_ZIGZAG] = q
        q = nat
    return y, q


def degrade_jpeg(im: Image.Image, qualities: Sequence[int], gray: bool = True):
    """Same pixels as degrade(), plus the final JPEG's luma plane and quantization table."""
    im = im.convert("L") if gray else im.convert("RGB")
    for q in qualities:
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=int(q))
        buf.seek(0)
        im = Image.open(buf)
        im.load()
    y, qt = jpeg_luma_info(im)
    return im.convert("RGB"), y, qt


def random_qualities(rng: random.Random, minq: int, max_rounds: int = 3) -> List[int]:
    k = rng.randint(1, max_rounds)
    return [rng.randint(minq, 100) for _ in range(k)]


def heldout_split(n_total: int, val_count: int = 2500, seed: int = 42):
    """Identical to train.py: torch.randperm with seed 42; first val_count -> validation."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=g).tolist()
    return perm[val_count:], perm[:val_count]


def to_tensor(rgb: np.ndarray) -> torch.Tensor:
    x = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy()).float()


_LMDB_ENVS = {}


def _lmdb_env(path: str):
    """One read-only LMDB environment per (process, path): py-lmdb refuses to open the same path
    twice in one process (e.g. train + val datasets with workers=0). Keyed by pid, so a forked
    worker never reuses its parent's handle."""
    key = (os.getpid(), path)
    env = _LMDB_ENVS.get(key)
    if env is None:
        env = lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False, max_readers=128)
        _LMDB_ENVS[key] = env
    return env


class _LMDBBase(Dataset):
    def __init__(self, lmdb_path: str, indices: Sequence[int], resolution: int = 512, jpeg_info: bool = False):
        self.lmdb_path = lmdb_path
        # jpeg_info: also return the final JPEG's luma plane + quantization table (for the
        # quantized-DCT stream). Off by default, so existing runs get exactly the same samples.
        self.jpeg_info = jpeg_info
        # numpy array, not a Python list: DataLoader workers touching the refcounts of ~120k int
        # objects copy those pages out of the parent (copy-on-write), growing every worker's RSS
        # (pytorch/pytorch#13246). Values and order are unchanged.
        self.indices = np.asarray(indices, dtype=np.int64)
        self.resolution = resolution
        self.env = None
        self.tokenizer = OtsuEntityTokenizer()

    def __len__(self):
        return len(self.indices)

    def _read(self, idx: int):
        if self.env is None:
            self.env = _lmdb_env(self.lmdb_path)
        with self.env.begin(write=False) as txn:
            img_buf = txn.get(b"image-%09d" % idx)
            lbl_buf = txn.get(b"label-%09d" % idx)
        im = Image.open(io.BytesIO(img_buf))
        mask = (cv2.imdecode(np.frombuffer(lbl_buf, np.uint8), 0) != 0).astype(np.uint8)
        return im, mask

    def _finish(self, rgb: np.ndarray, mask: np.ndarray, y: Optional[np.ndarray] = None,
                qt: Optional[np.ndarray] = None) -> Dict[str, torch.Tensor]:
        r = self.resolution
        if y is not None and rgb.shape[:2] != (r, r):
            raise ValueError("jpeg_info needs native resolution: resizing destroys the 8x8 JPEG grid")
        if rgb.shape[:2] != (r, r):
            # DocTamper images are 512x512; only reached if a different resolution is requested.
            rgb = cv2.resize(rgb, (r, r), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (r, r), interpolation=cv2.INTER_NEAREST)
        prior = self.tokenizer.extract_entity_mask(rgb)
        return {
            "image": to_tensor(rgb),
            "mask": torch.from_numpy(mask.astype(np.float32))[None],
            "label": torch.tensor([float(mask.max() > 0)], dtype=torch.float32),
            "entity_prior": torch.from_numpy(prior)[None].float(),
            **({"jpeg_y": torch.from_numpy(np.array(y, dtype=np.uint8))[None],   # copy: PIL buffers are read-only
                "qtable": torch.from_numpy(qt)} if y is not None else {}),
        }


class DocTamperTrainDataset(_LMDBBase):
    """Training split with protocol-family degradations drawn from our own RNG."""

    def __init__(self, lmdb_path, indices, resolution=512, minq=95, gray_prob=1.0,
                 flip_prob=0.5, seed=0, jpeg_info=False):
        super().__init__(lmdb_path, indices, resolution, jpeg_info)
        # minq / epoch live in shared memory so persistent DataLoader workers see the trainer's
        # per-epoch curriculum update (a plain attribute would stay frozen inside each worker).
        self._minq = mp.RawValue("i", int(minq))
        self._epoch = mp.RawValue("i", 0)
        self.gray_prob = gray_prob
        self.flip_prob = flip_prob
        self.seed = seed

    @property
    def minq(self) -> int:
        return self._minq.value

    @minq.setter
    def minq(self, v):
        self._minq.value = int(v)

    @property
    def epoch(self) -> int:
        return self._epoch.value

    @epoch.setter
    def epoch(self, v):
        self._epoch.value = int(v)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        rng = random.Random((self.seed, self.epoch, idx).__hash__())
        im, mask = self._read(idx)
        qs, gray = random_qualities(rng, self.minq), rng.random() < self.gray_prob
        y = qt = None
        if self.jpeg_info:
            im, y, qt = degrade_jpeg(im, qs, gray=gray)
        else:
            im = degrade(im, qs, gray=gray)
        rgb = np.asarray(im)
        # Flips after compression keep the 8x8 JPEG grid aligned (512 is a multiple of 8); a flip
        # only changes the sign of odd-frequency DCT coefficients, so |coefficients| are unchanged.
        if rng.random() < self.flip_prob:
            rgb, mask = rgb[:, ::-1], mask[:, ::-1]
            y = y[:, ::-1] if y is not None else None
        if rng.random() < self.flip_prob:
            rgb, mask = rgb[::-1], mask[::-1]
            y = y[::-1] if y is not None else None
        return self._finish(np.ascontiguousarray(rgb), np.ascontiguousarray(mask), y, qt)


class DocTamperValDataset(_LMDBBase):
    """Held-out validation (from the TRAINING LMDB) with a fixed, self-generated quality table."""

    def __init__(self, lmdb_path, indices, resolution=512, minq=75, seed=2026, gray=True,
                 table: Optional[Dict[int, List[int]]] = None, jpeg_info=False):
        super().__init__(lmdb_path, indices, resolution, jpeg_info)
        rng = random.Random(seed)
        self.table = table or {int(i): random_qualities(rng, minq) for i in self.indices}
        self.gray = gray

    def __getitem__(self, i):
        idx = int(self.indices[i])
        im, mask = self._read(idx)
        if self.jpeg_info:
            im, y, qt = degrade_jpeg(im, self.table[idx], gray=self.gray)
            return self._finish(np.asarray(im), mask, y, qt)
        im = degrade(im, self.table[idx], gray=self.gray)
        return self._finish(np.asarray(im), mask)
