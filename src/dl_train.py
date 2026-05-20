"""1D-CNN trained and evaluated on local data only (LORO).

Two configurations are reported side-by-side:
    (i)  PDF features only  — 3 channels: ECG, Resp, Mic Shannon envelope
    (ii) + temperature      — 4 channels: above + Temp (linear-interpolated)

Test set is always a held-out local recording (matches the classical LORO).
WESAD is intentionally excluded per user direction.
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

from .dl_models import BiosignalCNN1D, count_parameters
from .pipeline import PHASE_CLASSES
from .raw_windows import (
    ACTIVITY_TO_LABEL,
    LABEL_NAMES,
    RawWindow,
    WINDOW_N,
    local_raw_windows,
    stack_windows,
)


BATCH_SIZE = 32
EPOCHS = 80
PATIENCE = 12
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---- dataset --------------------------------------------------------------

class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, *, augment: bool = False):
        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int):
        x = self.X[i]
        if self.augment:
            # ±0.5 s time shift (≤125 samples at 250 Hz)
            shift = int(np.random.randint(-125, 126))
            x = np.roll(x, shift, axis=-1)
            # ±5% per-channel amplitude scale
            n_ch = x.shape[0]
            scale = (1.0 + 0.05 * (2 * np.random.rand(n_ch, 1) - 1)).astype(np.float32)
            x = x * scale
            # additive Gaussian noise σ=0.05 z-units
            x = x + (0.05 * np.random.randn(*x.shape)).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(x)), int(self.y[i])


# ---- training -------------------------------------------------------------

def _class_weights(y: np.ndarray, n_classes: int = 3) -> torch.Tensor:
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


def train_one_fold(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    *,
    in_channels: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = EPOCHS,
    patience: int = PATIENCE,
    batch_size: int = BATCH_SIZE,
    augment: bool = True,
) -> tuple[BiosignalCNN1D, dict]:
    model = BiosignalCNN1D(in_channels=in_channels, n_classes=len(PHASE_CLASSES)).to(DEVICE)
    weights = _class_weights(y_train, n_classes=len(PHASE_CLASSES))
    criterion = nn.CrossEntropyLoss(weight=weights)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda") if DEVICE == "cuda" else None

    train_loader = DataLoader(
        WindowDataset(X_train, y_train, augment=augment),
        batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0,
    )
    val_loader = DataLoader(
        WindowDataset(X_val, y_val, augment=False),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )

    history = {"train_loss": [], "val_loss": [], "val_macro_f1": []}
    best_f1 = -1.0
    best_state = None
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        tl = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optim.step()
            tl += float(loss.item()) * len(xb)
            n += len(xb)
        sched.step()
        history["train_loss"].append(tl / max(n, 1))

        model.eval()
        vl = 0.0
        n = 0
        preds: list[int] = []
        truths: list[int] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
                if scaler is not None:
                    with torch.amp.autocast("cuda"):
                        logits = model(xb)
                        loss = criterion(logits, yb)
                else:
                    logits = model(xb)
                    loss = criterion(logits, yb)
                vl += float(loss.item()) * len(xb)
                n += len(xb)
                preds.extend(logits.argmax(dim=1).cpu().tolist())
                truths.extend(yb.cpu().tolist())
        history["val_loss"].append(vl / max(n, 1))
        f1 = _macro_f1(np.array(truths), np.array(preds))
        history["val_macro_f1"].append(f1)

        if f1 > best_f1 + 1e-4:
            best_f1 = f1
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_val_macro_f1"] = float(best_f1)
    history["stopped_at_epoch"] = len(history["train_loss"])
    return model, history


@torch.no_grad()
def predict(model: BiosignalCNN1D, X: np.ndarray) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        WindowDataset(X, np.zeros(len(X), dtype=np.int64), augment=False),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    preds: list[int] = []
    for xb, _ in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        if DEVICE == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(xb)
        else:
            logits = model(xb)
        preds.extend(logits.argmax(dim=1).cpu().tolist())
    return np.array(preds, dtype=np.int64)


# ---- LORO over local data -------------------------------------------------

def _metrics(y_true_lbl: np.ndarray, y_pred_lbl: np.ndarray) -> dict:
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix, f1_score,
    )
    return dict(
        accuracy=float(accuracy_score(y_true_lbl, y_pred_lbl)),
        macro_f1=float(f1_score(y_true_lbl, y_pred_lbl,
                                average="macro", zero_division=0)),
        per_class_f1={
            c: float(f1_score(y_true_lbl, y_pred_lbl,
                              labels=[c], average="macro", zero_division=0))
            for c in PHASE_CLASSES
        },
        confusion=confusion_matrix(y_true_lbl, y_pred_lbl, labels=PHASE_CLASSES).tolist(),
        report=classification_report(
            y_true_lbl, y_pred_lbl, labels=PHASE_CLASSES, zero_division=0, output_dict=True,
        ),
    )


def _id_to_name(y: np.ndarray) -> np.ndarray:
    return np.array([LABEL_NAMES[i] for i in y])


def run_local_loro(*, include_temp: bool, epochs: int = EPOCHS) -> dict:
    """LORO over local recordings with the 1D-CNN. Returns per-fold + summary."""
    _set_seed(SEED)
    label = "PDF channels + temperature" if include_temp else "PDF channels"
    print(f"\n[{label}] building raw windows ...")
    ws = local_raw_windows(include_temp=include_temp)
    X, y = stack_windows(ws)
    rec_names = np.array([w.rec_name for w in ws])
    print(f"[{label}] X={X.shape} dtype={X.dtype}  labels={dict(Counter(y.tolist()))}")
    in_channels = X.shape[1]

    recordings = sorted(set(rec_names.tolist()))
    folds: list[dict] = []
    for fold_idx, test_rec in enumerate(recordings, 1):
        test_mask = (rec_names == test_rec)
        X_te, y_te = X[test_mask], y[test_mask]
        X_tr_pool, y_tr_pool, r_tr_pool = X[~test_mask], y[~test_mask], rec_names[~test_mask]

        # Inner val: hold out one other recording for early stopping
        other = sorted(set(r_tr_pool.tolist()))
        val_rec = other[0] if other else None
        val_mask = (r_tr_pool == val_rec) if val_rec is not None else np.zeros(len(X_tr_pool), bool)
        X_va, y_va = X_tr_pool[val_mask], y_tr_pool[val_mask]
        X_tr, y_tr = X_tr_pool[~val_mask], y_tr_pool[~val_mask]

        t0 = time.time()
        model, hist = train_one_fold(
            X_tr, y_tr, X_va, y_va,
            in_channels=in_channels, epochs=epochs,
        )
        y_pred = predict(model, X_te)
        m = _metrics(_id_to_name(y_te), _id_to_name(y_pred))
        dt = time.time() - t0
        print(f"  [fold {fold_idx}/{len(recordings)}  test={test_rec:<28s}] "
              f"macroF1={m['macro_f1']:.3f}  acc={m['accuracy']:.3f}  "
              f"({dt:.0f}s, stopped@{hist['stopped_at_epoch']})")
        folds.append(dict(
            recording=test_rec,
            test_n=int(len(X_te)),
            train_n=int(len(X_tr)),
            val_recording=val_rec,
            val_n=int(len(X_va)),
            test_activities={LABEL_NAMES[i]: int(c) for i, c in Counter(y_te.tolist()).items()},
            **m,
            history=hist,
        ))

    summary = _aggregate(folds)
    return dict(
        label=label,
        in_channels=in_channels,
        recordings=recordings,
        folds=folds,
        summary=summary,
    )


def _aggregate(folds: list[dict]) -> dict:
    acc = [f["accuracy"] for f in folds]
    f1m = [f["macro_f1"] for f in folds]
    per_cls = {c: [f["per_class_f1"][c] for f in folds] for c in PHASE_CLASSES}
    cm = np.sum([f["confusion"] for f in folds], axis=0).tolist()
    return dict(
        mean_accuracy=float(np.mean(acc)),
        std_accuracy=float(np.std(acc)),
        mean_macro_f1=float(np.mean(f1m)),
        std_macro_f1=float(np.std(f1m)),
        per_class_f1_mean={c: float(np.mean(per_cls[c])) for c in PHASE_CLASSES},
        confusion_total=cm,
    )


def run_local_temp_ablation(*, out_dir: str = "outputs", epochs: int = EPOCHS) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    print(f"device: {DEVICE}")
    print(f"model params: {count_parameters(BiosignalCNN1D(in_channels=3))}")

    pdf_only = run_local_loro(include_temp=False, epochs=epochs)
    with_temp = run_local_loro(include_temp=True, epochs=epochs)

    out = dict(pdf_only=pdf_only, with_temp=with_temp)
    path = os.path.join(out_dir, "dl_local_loro_temp_ablation.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=_json_default)
    print(f"\n  -> wrote {path}")

    _print_comparison(out)
    return out


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)


def _print_comparison(out: dict) -> None:
    print("\n" + "=" * 78)
    print(" 1D-CNN local-only LORO  —  with temperature vs without")
    print("=" * 78)
    header_cls = "  ".join(f"F1[{c}]" for c in PHASE_CLASSES)
    print(f"{'config':<30s} {'acc':>6s} {'macroF1':>9s}  {header_cls}")
    for key, lbl in (("pdf_only", "PDF channels (3 ch)"),
                     ("with_temp", "+temp channel (4 ch)")):
        s = out[key]["summary"]
        per_cls = "  ".join(f"{s['per_class_f1_mean'][c]:6.3f}" for c in PHASE_CLASSES)
        print(f"{lbl:<30s} {s['mean_accuracy']:>6.3f} "
              f"{s['mean_macro_f1']:>9.3f}  {per_cls}")

    print("\nDelta (with_temp − pdf_only):")
    a = out["pdf_only"]["summary"]
    b = out["with_temp"]["summary"]
    d_acc = b["mean_accuracy"] - a["mean_accuracy"]
    d_f1 = b["mean_macro_f1"] - a["mean_macro_f1"]
    d_pc = "  ".join(f"{b['per_class_f1_mean'][c] - a['per_class_f1_mean'][c]:+6.3f}"
                     for c in PHASE_CLASSES)
    print(f"{'1D-CNN':<14s} {d_acc:+8.3f} {d_f1:+11.3f}  {d_pc}")


def main() -> None:
    run_local_temp_ablation()


if __name__ == "__main__":
    main()
