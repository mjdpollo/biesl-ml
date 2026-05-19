"""Three-condition LORO comparison with the 1D CNN.

Conditions (each fold: test set is one held-out local recording):
    A        local-only             random init,         LR 1e-3
    B        WESAD-only -> local    random init,         LR 1e-3
    C-head   transfer, head-only    init from B's best,  LR 1e-3, conv frozen
    C-full   transfer, full FT      init from B's best,  LR 1e-4, all trainable

Matches the schema of `src/transfer.py` so the JSON output can be plotted
side-by-side with the classical XGBoost run.
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
    N_CHANNELS,
    RawWindow,
    WINDOW_N,
    local_raw_windows,
    stack_windows,
    wesad_raw_windows,
)
from .transfer import _aggregate, _metrics, _print_summary


# ---- config ----------------------------------------------------------------

BATCH_SIZE = 64
EPOCHS = 60
PATIENCE = 10
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
    """Wraps (X, y) arrays. Optional augmentation: time shift + amp scale + noise."""

    def __init__(self, X: np.ndarray, y: np.ndarray, *, augment: bool = False):
        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int):
        x = self.X[i]
        if self.augment:
            # time shift up to ±0.5 s (~125 samples at 250 Hz)
            shift = int(np.random.randint(-125, 126))
            x = np.roll(x, shift, axis=-1)
            # per-channel amp scale ±5%
            scale = (1.0 + 0.05 * (2 * np.random.rand(N_CHANNELS, 1) - 1)).astype(np.float32)
            x = x * scale
            # additive Gaussian noise σ=0.05 z-units
            x = x + (0.05 * np.random.randn(*x.shape)).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(x)), int(self.y[i])


# ---- training loop --------------------------------------------------------

def _class_weights(y: np.ndarray, n_classes: int = 3) -> torch.Tensor:
    """Inverse-frequency, normalized so mean = 1."""
    counts = np.zeros(n_classes, dtype=np.float64)
    for c, n in Counter(y.tolist()).items():
        counts[c] = n
    counts = np.where(counts > 0, counts, 1.0)
    w = counts.sum() / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, average="macro", labels=list(range(len(PHASE_CLASSES))),
                          zero_division=0))


def train_one_fold(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    *,
    n_classes: int = 3,
    init_state_dict: dict | None = None,
    freeze_features: bool = False,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = EPOCHS,
    patience: int = PATIENCE,
    batch_size: int = BATCH_SIZE,
    augment: bool = True,
    verbose: bool = False,
) -> tuple[BiosignalCNN1D, dict]:
    """Train + early-stop on val macro-F1. Returns (model, history dict)."""
    model = BiosignalCNN1D(in_channels=N_CHANNELS, n_classes=n_classes).to(DEVICE)
    if init_state_dict is not None:
        model.load_state_dict(init_state_dict)
    if freeze_features:
        model.freeze_features()

    weights = _class_weights(y_train, n_classes=n_classes)
    criterion = nn.CrossEntropyLoss(weight=weights)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
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

        # validation
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

        improved = f1 > best_f1 + 1e-4
        if improved:
            best_f1 = f1
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if verbose:
            print(f"    ep{epoch:3d}  tl={history['train_loss'][-1]:.3f} "
                  f"vl={history['val_loss'][-1]:.3f}  vf1={f1:.3f}"
                  f"{' *' if improved else ''}")

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
    use_amp = DEVICE == "cuda"
    for xb, _ in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(xb)
        else:
            logits = model(xb)
        preds.extend(logits.argmax(dim=1).cpu().tolist())
    return np.array(preds, dtype=np.int64)


# ---- three-condition LORO -------------------------------------------------

def _windows_to_arrays(windows: list[RawWindow]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack windows + return parallel rec_name array."""
    X, y = stack_windows(windows)
    rec_names = np.array([w.rec_name for w in windows])
    return X, y, rec_names


def _id_to_class(y: np.ndarray) -> np.ndarray:
    return np.array([PHASE_CLASSES[i] for i in y])


def _maybe_pick_val_recording(rec_names: np.ndarray, test_rec: str) -> str | None:
    """Deterministically pick one recording from the train pool for early stopping."""
    others = sorted(set(rec_names.tolist()) - {test_rec})
    return others[0] if others else None


def run_three_way_dl(
    *,
    out_dir: str = "outputs",
    ckpt_dir: str = "outputs/dl_checkpoints",
    save: bool = True,
    epochs: int = EPOCHS,
    seed: int = SEED,
) -> dict:
    _set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    print("[local] building raw windows ...")
    lw = local_raw_windows()
    print("[wesad] building raw windows ...")
    ww = wesad_raw_windows()

    Xl, yl, rl = _windows_to_arrays(lw)
    Xw, yw, rw = _windows_to_arrays(ww)
    if len(Xw) == 0:
        raise RuntimeError("No WESAD windows produced — populate WESAD/ first.")

    print(f"\nlocal: {Xl.shape}  labels={Counter(yl.tolist())}")
    print(f"wesad: {Xw.shape}  labels={Counter(yw.tolist())}")
    print(f"model params: {count_parameters(BiosignalCNN1D())}")
    print(f"device: {DEVICE}")

    recordings = sorted(set(rl.tolist()))
    print(f"local recordings ({len(recordings)}): {recordings}\n")

    # ---- Pretrain WESAD model ONCE (used by both B and C variants) --------
    # Inner val: one random WESAD subject's windows held out for early stopping.
    wesad_subs = sorted({rec.split('_lbl')[0] for rec in rw.tolist()})  # "WESAD_S7", ...
    val_sub = wesad_subs[0]
    val_mask_w = np.array([n.startswith(val_sub + "_") for n in rw])
    Xw_tr, yw_tr = Xw[~val_mask_w], yw[~val_mask_w]
    Xw_va, yw_va = Xw[val_mask_w], yw[val_mask_w]
    print(f"[B] pretrain WESAD: train={Counter(yw_tr.tolist())} (val held-out {val_sub})")
    t0 = time.time()
    wesad_model, wesad_hist = train_one_fold(
        Xw_tr, yw_tr, Xw_va, yw_va, epochs=epochs, verbose=False,
    )
    print(f"[B] WESAD pretrain done in {time.time()-t0:.1f}s, "
          f"best val macro-F1={wesad_hist['best_val_macro_f1']:.3f}, "
          f"stopped@{wesad_hist['stopped_at_epoch']}")
    torch.save(wesad_model.state_dict(), os.path.join(ckpt_dir, "wesad_pretrained.pt"))
    wesad_state = copy.deepcopy(wesad_model.state_dict())

    # ---- LORO folds over local data --------------------------------------
    folds: list[dict] = []
    for fold_idx, test_rec in enumerate(recordings, 1):
        print(f"\n[fold {fold_idx}/{len(recordings)}] test = {test_rec}")

        test_mask = (rl == test_rec)
        Xte, yte = Xl[test_mask], yl[test_mask]
        Xtr_pool, ytr_pool, rtr_pool = Xl[~test_mask], yl[~test_mask], rl[~test_mask]

        val_rec = _maybe_pick_val_recording(rtr_pool, test_rec)
        val_mask = (rtr_pool == val_rec) if val_rec is not None else np.zeros(len(Xtr_pool), bool)
        Xva, yva = Xtr_pool[val_mask], ytr_pool[val_mask]
        Xtr, ytr = Xtr_pool[~val_mask], ytr_pool[~val_mask]

        # B evaluates the pretrained WESAD model directly on this fold's test set
        yhat_B = predict(wesad_model, Xte)
        m_B = _metrics(_id_to_class(yte), _id_to_class(yhat_B))

        # A: local-only from scratch
        t0 = time.time()
        model_A, hist_A = train_one_fold(Xtr, ytr, Xva, yva, epochs=epochs)
        yhat_A = predict(model_A, Xte)
        m_A = _metrics(_id_to_class(yte), _id_to_class(yhat_A))
        dt_A = time.time() - t0

        # C-head: init from WESAD, freeze features, train head on local-train
        t0 = time.time()
        model_Ch, hist_Ch = train_one_fold(
            Xtr, ytr, Xva, yva,
            init_state_dict=wesad_state, freeze_features=True,
            lr=1e-3, epochs=epochs,
        )
        yhat_Ch = predict(model_Ch, Xte)
        m_Ch = _metrics(_id_to_class(yte), _id_to_class(yhat_Ch))
        dt_Ch = time.time() - t0

        # C-full: init from WESAD, fine-tune all params at lower LR
        t0 = time.time()
        model_Cf, hist_Cf = train_one_fold(
            Xtr, ytr, Xva, yva,
            init_state_dict=wesad_state, freeze_features=False,
            lr=1e-4, epochs=epochs,
        )
        yhat_Cf = predict(model_Cf, Xte)
        m_Cf = _metrics(_id_to_class(yte), _id_to_class(yhat_Cf))
        dt_Cf = time.time() - t0

        # save checkpoints
        torch.save(model_A.state_dict(), os.path.join(ckpt_dir, f"A_fold{fold_idx}_{test_rec}.pt"))
        torch.save(model_Ch.state_dict(), os.path.join(ckpt_dir, f"Chead_fold{fold_idx}_{test_rec}.pt"))
        torch.save(model_Cf.state_dict(), os.path.join(ckpt_dir, f"Cfull_fold{fold_idx}_{test_rec}.pt"))

        print(
            f"  A={m_A['macro_f1']:.3f}  B={m_B['macro_f1']:.3f}  "
            f"C-head={m_Ch['macro_f1']:.3f}  C-full={m_Cf['macro_f1']:.3f}  "
            f"(A {dt_A:.0f}s, Ch {dt_Ch:.0f}s, Cf {dt_Cf:.0f}s)"
        )

        folds.append(dict(
            recording=test_rec,
            test_n=int(len(Xte)),
            train_local_n=int(len(Xtr)),
            val_recording=val_rec,
            val_n=int(len(Xva)),
            train_wesad_n=int(len(Xw_tr)),
            test_activities={LABEL_NAMES[i]: int(c) for i, c in Counter(yte.tolist()).items()},
            A_local_shared=m_A,
            B_wesad=m_B,
            C_head=m_Ch,
            C_full=m_Cf,
            history={
                "A": hist_A,
                "C_head": hist_Ch,
                "C_full": hist_Cf,
            },
        ))

    # Aggregate. Reuse the helper schema from transfer.py for consistency.
    summary = _aggregate_dl(folds)
    out = dict(
        classes=PHASE_CLASSES,
        device=DEVICE,
        seed=seed,
        epochs=epochs,
        wesad_pretrain_history=wesad_hist,
        folds=folds,
        summary=summary,
    )
    if save:
        path = os.path.join(out_dir, "dl_transfer_results.json")
        with open(path, "w") as fh:
            json.dump(out, fh, indent=2, default=_json_default)
        print(f"\n  -> wrote {path}")

    _print_summary_dl(summary)
    return out


LABEL_NAMES = PHASE_CLASSES


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)


def _aggregate_dl(folds: list[dict]) -> dict:
    keys = ("A_local_shared", "B_wesad", "C_head", "C_full")
    summary: dict = {}
    for k in keys:
        acc = [f[k]["accuracy"] for f in folds]
        f1m = [f[k]["macro_f1"] for f in folds]
        per_class = {
            c: [f[k]["per_class_f1"][c] for f in folds] for c in PHASE_CLASSES
        }
        cm = np.sum([f[k]["confusion"] for f in folds], axis=0).tolist()
        summary[k] = dict(
            mean_accuracy=float(np.mean(acc)),
            std_accuracy=float(np.std(acc)),
            mean_macro_f1=float(np.mean(f1m)),
            std_macro_f1=float(np.std(f1m)),
            per_class_f1_mean={c: float(np.mean(per_class[c])) for c in PHASE_CLASSES},
            confusion_total=cm,
        )
    return summary


def _print_summary_dl(summary: dict) -> None:
    pretty = {
        "A_local_shared": "A  local-only (scratch)",
        "B_wesad":        "B  WESAD-only -> local",
        "C_head":         "C-head  pretrain + frozen conv",
        "C_full":         "C-full  pretrain + full FT",
    }
    print("\n========== 1D-CNN three-way comparison (LORO, test=local) ==========")
    header_classes = "  ".join(f"F1[{c}]" for c in PHASE_CLASSES)
    print(f"{'condition':<34s} {'acc':>6s} {'macroF1':>9s}  {header_classes}")
    for k, label in pretty.items():
        s = summary[k]
        per_cls = "  ".join(f"{s['per_class_f1_mean'][c]:6.3f}" for c in PHASE_CLASSES)
        print(
            f"{label:<34s} {s['mean_accuracy']:>6.3f} {s['mean_macro_f1']:>9.3f}  {per_cls}"
        )


def main() -> None:
    run_three_way_dl()


if __name__ == "__main__":
    main()
