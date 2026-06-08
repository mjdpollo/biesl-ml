"""2D-CNN on Poincare images — leave-one-recording-out (LORO) on local data.

Test set is always a held-out recording (recording-grouped, honest). For each
fold one other recording is held out as the inner validation set for early
stopping, mirroring src/dl_train.py.

Run:  uv run python -m src.poincare_train
"""
from __future__ import annotations

import copy
import json
import os
import random
import time
from collections import Counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .pipeline import PHASE_CLASSES
from .poincare_cnn import PoincareCNN, count_parameters
from .poincare_images import LABEL_NAMES, build_and_cache, load_cache

BATCH_SIZE = 32
EPOCHS = 120
PATIENCE = 20
LR = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
CACHE = "outputs/poincare_dataset.npz"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ImgDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, *, augment: bool = False):
        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int):
        x = self.X[i]
        if self.augment:
            # mild additive Gaussian noise only — geometric flips would change
            # the physiological meaning of a Poincare plot, so they are avoided.
            x = x + (0.02 * np.random.randn(*x.shape)).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(x)), int(self.y[i])


def _class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.zeros(n_classes, dtype=np.float64)
    for c, n in Counter(y.tolist()).items():
        counts[c] = n
    counts = np.where(counts > 0, counts, 1.0)
    w = counts.sum() / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, average="macro",
                          labels=list(range(len(PHASE_CLASSES))), zero_division=0))


def train_one_fold(X_tr, y_tr, X_va, y_va, *, n_classes, epochs=EPOCHS, patience=PATIENCE):
    model = PoincareCNN(in_channels=1, n_classes=n_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=_class_weights(y_tr, n_classes))
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda") if DEVICE == "cuda" else None

    tr_loader = DataLoader(ImgDataset(X_tr, y_tr, augment=True),
                           batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    has_val = len(X_va) > 0
    va_loader = (DataLoader(ImgDataset(X_va, y_va, augment=False),
                            batch_size=BATCH_SIZE, shuffle=False) if has_val else None)

    history = {"train_loss": [], "val_macro_f1": []}
    best_f1, best_state, bad = -1.0, None, 0

    for _ in range(1, epochs + 1):
        model.train()
        tl, n = 0.0, 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    loss = criterion(model(xb), yb)
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss = criterion(model(xb), yb)
                loss.backward()
                optim.step()
            tl += float(loss.item()) * len(xb)
            n += len(xb)
        sched.step()
        history["train_loss"].append(tl / max(n, 1))

        if not has_val:
            best_state = copy.deepcopy(model.state_dict())
            continue

        preds, truths = _infer(model, va_loader)
        f1 = _macro_f1(np.array(truths), np.array(preds))
        history["val_macro_f1"].append(f1)
        if f1 > best_f1 + 1e-4:
            best_f1, best_state, bad = f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        if bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_val_macro_f1"] = float(best_f1)
    history["stopped_at_epoch"] = len(history["train_loss"])
    return model, history


@torch.no_grad()
def _infer(model, loader):
    model.eval()
    preds, truths = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        if DEVICE == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(xb)
        else:
            logits = model(xb)
        preds.extend(logits.argmax(dim=1).cpu().tolist())
        truths.extend(yb.tolist())
    return preds, truths


@torch.no_grad()
def predict(model, X):
    loader = DataLoader(ImgDataset(X, np.zeros(len(X), dtype=np.int64)),
                        batch_size=BATCH_SIZE, shuffle=False)
    preds, _ = _infer(model, loader)
    return np.array(preds, dtype=np.int64)


def _id_to_name(y):
    return np.array([LABEL_NAMES[i] for i in y])


def _metrics(y_true_lbl, y_pred_lbl):
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix, f1_score,
    )
    return dict(
        accuracy=float(accuracy_score(y_true_lbl, y_pred_lbl)),
        macro_f1=float(f1_score(y_true_lbl, y_pred_lbl, average="macro",
                                labels=PHASE_CLASSES, zero_division=0)),
        per_class_f1={c: float(f1_score(y_true_lbl, y_pred_lbl, labels=[c],
                                        average="macro", zero_division=0))
                      for c in PHASE_CLASSES},
        confusion=confusion_matrix(y_true_lbl, y_pred_lbl, labels=PHASE_CLASSES).tolist(),
        report=classification_report(y_true_lbl, y_pred_lbl, labels=PHASE_CLASSES,
                                     zero_division=0, output_dict=True),
    )


def run_loro(data: dict, *, epochs: int = EPOCHS) -> dict:
    _set_seed(SEED)
    X, y = data["X"], data["y"]
    rec_names = data["rec_names"]
    n_classes = len(PHASE_CLASSES)
    print(f"X={X.shape}  labels="
          f"{ {LABEL_NAMES[k]: v for k, v in sorted(Counter(y.tolist()).items())} }")

    recordings = sorted(set(rec_names.tolist()))
    folds = []
    for i, test_rec in enumerate(recordings, 1):
        te = rec_names == test_rec
        X_te, y_te = X[te], y[te]
        X_pool, y_pool, r_pool = X[~te], y[~te], rec_names[~te]

        other = sorted(set(r_pool.tolist()))
        val_rec = other[0] if other else None
        vm = (r_pool == val_rec) if val_rec is not None else np.zeros(len(X_pool), bool)
        X_va, y_va = X_pool[vm], y_pool[vm]
        X_tr, y_tr = X_pool[~vm], y_pool[~vm]

        t0 = time.time()
        model, hist = train_one_fold(X_tr, y_tr, X_va, y_va,
                                     n_classes=n_classes, epochs=epochs)
        y_pred = predict(model, X_te)
        m = _metrics(_id_to_name(y_te), _id_to_name(y_pred))
        print(f"  [fold {i:2d}/{len(recordings)} test={test_rec:<26s}] "
              f"macroF1={m['macro_f1']:.3f} acc={m['accuracy']:.3f}  "
              f"({time.time()-t0:.0f}s, stop@{hist['stopped_at_epoch']}, "
              f"test={dict(Counter(_id_to_name(y_te).tolist()))})")
        folds.append(dict(recording=test_rec, test_n=int(len(X_te)),
                          train_n=int(len(X_tr)), val_recording=val_rec,
                          val_n=int(len(X_va)), **m, history=hist))

    return dict(recordings=recordings, folds=folds, summary=_aggregate(folds))


def _aggregate(folds):
    acc = [f["accuracy"] for f in folds]
    f1m = [f["macro_f1"] for f in folds]
    per_cls = {c: [f["per_class_f1"][c] for f in folds] for c in PHASE_CLASSES}
    cm = np.sum([f["confusion"] for f in folds], axis=0).tolist()
    return dict(
        mean_accuracy=float(np.mean(acc)), std_accuracy=float(np.std(acc)),
        mean_macro_f1=float(np.mean(f1m)), std_macro_f1=float(np.std(f1m)),
        per_class_f1_mean={c: float(np.mean(per_cls[c])) for c in PHASE_CLASSES},
        confusion_total=cm,
    )


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)


def main(*, rebuild: bool = True, epochs: int = EPOCHS, out_dir: str = "outputs") -> dict:
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 78)
    print(" Poincare-image 2D-CNN  —  local LORO")
    print("=" * 78)
    print(f"device: {DEVICE}  model params: {count_parameters(PoincareCNN())}")

    if rebuild or not os.path.exists(CACHE):
        data = build_and_cache(cache_path=CACHE, norm="per_image")
    else:
        data = load_cache(CACHE)
        print(f"loaded cached dataset {data['X'].shape}")

    result = run_loro(data, epochs=epochs)
    path = os.path.join(out_dir, "poincare_loro.json")
    with open(path, "w") as fh:
        json.dump(result, fh, indent=2, default=_json_default)
    print(f"\n  -> wrote {path}")

    s = result["summary"]
    print("\n" + "=" * 78)
    print(" Poincare 2D-CNN LORO summary")
    print("=" * 78)
    header = "  ".join(f"F1[{c}]" for c in PHASE_CLASSES)
    per_cls = "  ".join(f"{s['per_class_f1_mean'][c]:.3f}" for c in PHASE_CLASSES)
    print(f"{'acc':>14s} {'macroF1':>14s}   {header}")
    print(f"{s['mean_accuracy']:.3f}+/-{s['std_accuracy']:.3f}   "
          f"{s['mean_macro_f1']:.3f}+/-{s['std_macro_f1']:.3f}   {per_cls}")
    return result


if __name__ == "__main__":
    main()
