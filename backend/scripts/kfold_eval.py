#!/usr/bin/env python3
"""K-fold cross-validation for an honest generalization estimate.

Single 41-sample held-out split is saturated at 100% and too small to
distinguish pipelines. This runs the exact train.py pipeline
(raw -> stratified split -> augment train only -> train stats ->
BiGRU + early stopping) K times with different held-out folds and
reports mean/worst-fold accuracy + margin.

Each fold trains a model from scratch; folds run sequentially.
Run from backend/:  venv/bin/python scripts/kfold_eval.py [--folds 5]
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold

import config as cfg
from vsl_translator.model import build_model
from vsl_translator.dataset import (
    load_raw_dataset, augment_train_data, standardize_features,
)
from vsl_translator.callbacks import (
    CosineAnnealingWarmup, TrainingMonitor, SmartEarlyStopping,
)


def run_fold(X_tr_raw, y_tr_raw, X_te_raw, y_te, fold_id):
    from keras.callbacks import TensorBoard
    n_classes = len(cfg.ACTIONS)
    batch_size = min(cfg.BATCH_SIZE, len(X_tr_raw))

    X_tr_aug, y_tr = augment_train_data(
        X_tr_raw, y_tr_raw, augment_factor=cfg.AUGMENT_FACTOR)
    X_tr, mean, std = standardize_features(X_tr_aug)
    X_te, _, _ = standardize_features(X_te_raw, mean=mean, std=std)

    # class weights on augmented train labels
    from sklearn.utils.class_weight import compute_class_weight
    yi = np.argmax(y_tr, axis=1)
    classes = np.unique(yi)
    cw = dict(zip(classes, compute_class_weight(
        'balanced', classes=classes, y=yi)))

    model = build_model(n_actions=n_classes, sequence_len=cfg.SEQUENCE_LEN)
    monitor = TrainingMonitor(overfit_gap_threshold=cfg.OVERFIT_GAP_THRESHOLD)
    stopper = SmartEarlyStopping(
        patience=cfg.ES_PATIENCE, min_delta=cfg.ES_MIN_DELTA,
        target_acc=cfg.TARGET_VAL_ACC, target_loss=cfg.TARGET_VAL_LOSS,
        overfit_gap_threshold=cfg.OVERFIT_GAP_THRESHOLD,
        overfit_patience=cfg.OVERFIT_PATIENCE,
        start_from_epoch=cfg.ES_START_EPOCH,
        loss_spike_patience=cfg.ES_LOSS_SPIKE_PATIENCE,
        plateau_patience=cfg.ES_PLATEAU_PATIENCE,
        plateau_min_delta=cfg.ES_PLATEAU_MIN_DELTA,
    )
    # val = 15% of the fold-train (stratified, untouched by augment)
    from sklearn.model_selection import train_test_split
    try:
        X_fit_raw, X_val_raw, y_fit_raw, y_val = train_test_split(
            X_tr_raw, y_tr_raw, test_size=cfg.VAL_SPLIT,
            random_state=cfg.RANDOM_SEED + fold_id,
            stratify=y_tr_raw.argmax(axis=1))
    except ValueError:
        X_fit_raw, X_val_raw, y_fit_raw, y_val = train_test_split(
            X_tr_raw, y_tr_raw, test_size=cfg.VAL_SPLIT,
            random_state=cfg.RANDOM_SEED + fold_id)
    X_fit_aug, y_fit = augment_train_data(
        X_fit_raw, y_fit_raw, augment_factor=cfg.AUGMENT_FACTOR)
    X_fit, fmean, fstd = standardize_features(X_fit_aug)
    X_val, _, _ = standardize_features(X_val_raw, mean=fmean, std=fstd)
    X_te2, _, _ = standardize_features(X_te_raw, mean=fmean, std=fstd)

    yi_fit = np.argmax(y_fit, axis=1)
    cl = np.unique(yi_fit)
    cw_fit = dict(zip(cl, compute_class_weight(
        'balanced', classes=cl, y=yi_fit)))

    model.fit(
        X_fit, y_fit, validation_data=(X_val, y_val),
        epochs=cfg.EPOCHS, batch_size=min(cfg.BATCH_SIZE, len(X_fit)),
        shuffle=True, class_weight=cw_fit,
        callbacks=[
            CosineAnnealingWarmup(
                lr_max=cfg.LR_INITIAL, lr_min=cfg.LR_MIN,
                warmup_epochs=cfg.LR_WARMUP_EPOCHS,
                total_epochs=cfg.EPOCHS),
            monitor, stopper,
        ],
        verbose=0,
    )
    loss, acc = model.evaluate(X_te2, y_te, verbose=0)
    probs = model.predict(X_te2, verbose=0)
    pred = probs.argmax(axis=1)
    true = y_te.argmax(axis=1)
    part = np.partition(probs, -2, axis=1)
    margin = float((part[:, -1] - part[:, -2]).mean())
    per_class = {}
    for c in range(n_classes):
        m = true == c
        per_class[str(cfg.ACTIONS[c])] = (
            float((pred[m] == c).mean()) if m.sum() else float("nan"))
    return {"fold": fold_id, "n_test": len(X_te),
            "loss": float(loss), "acc": float(acc),
            "margin": margin, "epochs": len(monitor.history["loss"]),
            "per_class_acc": per_class}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--preset", default=None, choices=["base", "strong"],
                    help="augmentation preset (default: cfg.AUGMENT_PRESET)")
    args = ap.parse_args()
    if args.preset is not None:
        cfg.AUGMENT_PRESET = args.preset
    print(f"Augment preset: {cfg.AUGMENT_PRESET}")

    X_raw, y_raw = load_raw_dataset()
    y_int = y_raw.argmax(axis=1)
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True,
                          random_state=cfg.RANDOM_SEED)
    results = []
    for i, (tr, te) in enumerate(skf.split(X_raw, y_int)):
        print(f"\n===== Fold {i + 1}/{args.folds} "
              f"(train {len(tr)}, test {len(te)}) =====")
        r = run_fold(X_raw[tr], y_raw[tr], X_raw[te], y_raw[te], i)
        print(f"  fold acc={r['acc']:.4f} margin={r['margin']:.4f} "
              f"epochs={r['epochs']}")
        results.append(r)

    accs = [r["acc"] for r in results]
    margins = [r["margin"] for r in results]
    print("\n===== K-fold summary =====")
    print(f"  folds: {args.folds}, mean acc={np.mean(accs):.4f} "
          f"std={np.std(accs):.4f} worst={np.min(accs):.4f}")
    print(f"  mean margin={np.mean(margins):.4f} worst={np.min(margins):.4f}")
    for r in results:
        bad = {k: v for k, v in r["per_class_acc"].items() if v < 1.0}
        print(f"  fold {r['fold']}: acc={r['acc']:.4f} "
              f"sub-100% classes: {bad or 'none'}")


if __name__ == "__main__":
    main()
