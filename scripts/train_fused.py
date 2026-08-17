#!/usr/bin/env python3
"""Train the fused metadata model (DINOv2-S frozen + TinyCNN + MetaMLP).

This is the dermatoscopic base model served by the web app
(model_10_6 / checkpoint_20260610_230527.pt). Hyperparameter defaults are the
exact configuration of that run; see docs/training/model_10_6.md for the
recorded results (test: Acc 0.876, TPR 0.912, FPR 0.136).

Expects the metadata CSV to have image_id, age, sex, anatom_site, target and a
"split" column (from scripts/make_split.py), and the image dir to contain
pre_<image_id>.jpg files (from scripts/preprocess_fused_dataset.py).

Usage:
    python scripts/train_fused.py \
        --metadata final_metadata.csv \
        --img-dir preprocessed448 \
        --output-dir runs/fused
"""

import argparse
import datetime
import time
import warnings

import pandas as pd
import torch
from torch.utils.data import DataLoader

from lumen.training.fused import (
    BCEJLoss,
    FusedImageDataset,
    META_DIM,
    SEX_CATEGORIES,
    SITE_CATEGORIES,
    build_fused_model,
    build_optimizer,
    format_metrics_block,
    restore_state,
    run_epoch,
    save_fused_checkpoint,
    snapshot_state,
)

warnings.filterwarnings("ignore", message="xFormers")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", required=True, help="CSV with metadata + split column")
    parser.add_argument("--img-dir", required=True, help="Directory with pre_<image_id>.jpg images")
    parser.add_argument("--output-dir", default=".", help="Where to write checkpoint/logs (default: cwd)")
    parser.add_argument("--resize", type=int, default=448)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--t-max", type=int, default=40)
    parser.add_argument("--lam", type=float, default=0.9, help="BCEJLoss lambda")
    parser.add_argument("--tpr-weight", type=float, default=2.5, help="BCEJLoss TPR weight")
    parser.add_argument("--num-workers", type=int, default=12)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"{args.output_dir}/training_log_{timestamp}.txt"
    results_path = f"{args.output_dir}/results_{timestamp}.txt"
    ckpt_path = f"{args.output_dir}/checkpoint_{timestamp}.pt"

    df = pd.read_csv(args.metadata)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    n_pos = int((train_df["target"] == 1).sum())
    n_neg = int((train_df["target"] == 0).sum())

    age_train = pd.to_numeric(train_df["age"], errors="coerce")
    age_mean = float(age_train.mean())
    age_std = float(age_train.std())

    print(f"Device   : {device}")
    print(f"Train    : {len(train_df)}  ({n_pos} mal / {n_neg} ben)")
    print(f"Val      : {len(val_df)}")
    print(f"Test     : {len(test_df)}")
    print(f"Age stats (train): mean={age_mean:.2f}  std={age_std:.2f}")
    print(f"Meta dim : {META_DIM}  (age_norm + age_missing + sex[{len(SEX_CATEGORIES)}] + site[{len(SITE_CATEGORIES)}])\n")

    model = build_fused_model(device)

    n_cnn = sum(p.numel() for p in model.cnn.parameters())
    n_vp = sum(p.numel() for p in model.vision_proj.parameters())
    n_meta = sum(p.numel() for p in model.meta_mlp.parameters())
    n_clf = sum(p.numel() for p in model.classifier.parameters())

    header_lines = [
        "=" * 70,
        "DINOv2-S (frozen) ++ TinyCNN(AvgPool+MaxPool, 192) -> VisionProj -> vision_emb(256)",
        f"  ++  MetaMLP(11->16)  ->  Classifier(272->256->128->1)  |  {args.resize}px",
        "=" * 70,
        f"Optimizer  : AdamW (lr={args.lr}, wd={args.weight_decay})",
        f"Schedule   : LinearWarmup({args.warmup_epochs}ep) + CosineAnnealingLR (T_max={args.t_max})",
        f"Early stop : patience={args.patience} on val loss  |  Max epochs: {args.max_epochs}",
        f"Batch size : {args.batch_size}",
        f"Loss       : BCEJLoss (λ={args.lam}, tpr_weight={args.tpr_weight})",
        f"TinyCNN    : {n_cnn:,} param  |  VisionProj: {n_vp:,}  |  MetaMLP: {n_meta:,}  |  Classifier: {n_clf:,}",
        f"Trainable total: {n_cnn+n_vp+n_meta+n_clf:,}  (DINOv2-S frozen ~21M)",
        f"Age stats  : mean={age_mean:.2f}  std={age_std:.2f}  (from train split)",
        "",
    ]
    print("\n".join(header_lines))
    with open(log_path, "w") as f:
        f.write("\n".join(header_lines) + "\n")

    def make_loader(ds, shuffle):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

    train_ds = FusedImageDataset(train_df, args.img_dir, age_mean, age_std, args.resize, augment=True)
    train_eval_ds = FusedImageDataset(train_df, args.img_dir, age_mean, age_std, args.resize)
    val_ds = FusedImageDataset(val_df, args.img_dir, age_mean, age_std, args.resize)
    test_ds = FusedImageDataset(test_df, args.img_dir, age_mean, age_std, args.resize)

    train_loader = make_loader(train_ds, True)
    train_eval_loader = make_loader(train_eval_ds, False)
    val_loader = make_loader(val_ds, False)
    test_loader = make_loader(test_ds, False)

    criterion = BCEJLoss(lam=args.lam, tpr_weight=args.tpr_weight)
    optimizer, scheduler = build_optimizer(model, args.lr, args.weight_decay, args.warmup_epochs, args.t_max)

    autocast = device.type == "cuda"
    best_val_loss = float("inf")
    best_state, best_epoch, patience_cnt = None, 0, 0
    t0 = time.time()

    for epoch in range(1, args.max_epochs + 1):
        tr_loss, _ = run_epoch(model, train_loader, criterion, device, optimizer, scheduler, dino_autocast=autocast)
        ev_loss, ev_m = run_epoch(model, val_loader, criterion, device, dino_autocast=autocast)

        line = (f"  ep{epoch:03d}  tr={tr_loss:.4f}  ev={ev_loss:.4f}"
                f"  F1={ev_m['F1']:.3f}  TPR={ev_m['TPR']:.3f}  FPR={ev_m['FPR']:.3f}"
                f"  ({int(time.time()-t0)}s)")
        print(line)
        with open(log_path, "a") as f:
            f.write(line + "\n")

        if ev_loss < best_val_loss:
            best_val_loss = ev_loss
            best_state = snapshot_state(model)
            best_epoch, patience_cnt = epoch, 0
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                stop_line = f"  >> Early stop (best epoch={best_epoch}, best_val_loss={best_val_loss:.4f})"
                print(stop_line)
                with open(log_path, "a") as f:
                    f.write(stop_line + "\n")
                break

    elapsed = int(time.time() - t0)
    restore_state(model, best_state)
    save_fused_checkpoint(ckpt_path, best_state, age_mean, age_std, best_epoch, args.resize)

    _, tr_m = run_epoch(model, train_eval_loader, criterion, device, dino_autocast=autocast)
    _, val_m = run_epoch(model, val_loader, criterion, device, dino_autocast=autocast)
    _, te_m = run_epoch(model, test_loader, criterion, device, dino_autocast=autocast)

    results_lines = header_lines + [
        "=" * 70,
        f"RUN: train  |  best_epoch={best_epoch}  |  best_val_loss={best_val_loss:.4f}  |  {elapsed}s",
        f"Checkpoint: {ckpt_path}",
        "=" * 70,
        format_metrics_block(tr_m, "TRAIN"), "",
        format_metrics_block(val_m, "VAL"), "",
        format_metrics_block(te_m, "TEST"),
    ]
    print("\n".join(results_lines[len(header_lines):]))
    with open(results_path, "w") as f:
        f.write("\n".join(results_lines) + "\n")


if __name__ == "__main__":
    main()
