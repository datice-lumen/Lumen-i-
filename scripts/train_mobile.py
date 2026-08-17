#!/usr/bin/env python3
"""Fine-tune the fused metadata model on MILK10k mobile close-ups.

Produces the mobile-domain model shipped as web_app/backend/checkpoint_mobile_best.pt.
Starts from a trained dermatoscopic checkpoint (scripts/train_fused.py) and
fine-tunes all trainable heads (TinyCNN, VisionProj, MetaMLP, Classifier) on
phone photos. Preprocessing is lumen.preprocessing.preprocess_mobile —
deliberately NO hair removal (see docs/training/mobile_findings.md).

The eval CSV comes from the mobile OOD evaluation (scripts/eval_mobile.py data
prep): one row per MILK10k mobile image with metadata + target taken from its
dermoscopic twin, and a twin_split column (the split its twin was in during
base training, so fine-tuning never trains on a twin of a base-model test image).

Usage:
    python scripts/train_mobile.py \
        --pretrained checkpoint_20260610_230527.pt \
        --eval-csv mobile_eval.csv \
        --images MILK10k/MILK10k_Training_Input \
        --output-dir runs/mobile
"""

import argparse
import time
import warnings

import pandas as pd
import torch
from torch.utils.data import DataLoader

from lumen.training.fused import (
    BCEJLoss,
    MobileImageDataset,
    build_fused_model,
    build_milk_path_index,
    build_optimizer,
    format_metrics_block,
    restore_state,
    run_epoch,
    save_fused_checkpoint,
    snapshot_state,
)

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pretrained", required=True, help="Base fused-model checkpoint (.pt)")
    parser.add_argument("--eval-csv", required=True, help="Mobile metadata CSV with twin_split column")
    parser.add_argument("--images", required=True, help="MILK10k_Training_Input directory")
    parser.add_argument("--output-dir", default=".", help="Where to write checkpoint_mobile_best.pt")
    parser.add_argument("--resize", type=int, default=448)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4, help="Lower than base training (fine-tune)")
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--t-max", type=int, default=40)
    parser.add_argument("--lam", type=float, default=0.9)
    parser.add_argument("--tpr-weight", type=float, default=2.5)
    parser.add_argument("--num-workers", type=int, default=12)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Device: {device}")

    df = pd.read_csv(args.eval_csv)
    path_index = build_milk_path_index(args.images)

    ck = torch.load(args.pretrained, map_location="cpu", weights_only=True)
    age_mean, age_std = ck["age_mean"], ck["age_std"]

    valid_mask = df["image_id"].isin(path_index)
    dropped = int((~valid_mask).sum())
    if dropped > 0:
        print(f"Warning: {dropped} images could not be located in directory. Filtering them out.")
        df = df[valid_mask].reset_index(drop=True)

    train_df = df[df["twin_split"] == "train"].reset_index(drop=True)
    val_df = df[df["twin_split"] == "val"].reset_index(drop=True)
    test_df = df[df["twin_split"] == "test"].reset_index(drop=True)
    print(f"Dataset split: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")

    def make_loader(frame, augment, shuffle):
        ds = MobileImageDataset(frame, age_mean, age_std, path_index, args.resize, augment=augment)
        return ds, DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                              num_workers=args.num_workers, pin_memory=True)

    _, train_loader = make_loader(train_df, True, True)
    val_ds, val_loader = make_loader(val_df, False, False)
    test_ds, test_loader = make_loader(test_df, False, False)

    model = build_fused_model(device, checkpoint=ck)
    optimizer, scheduler = build_optimizer(model, args.lr, args.weight_decay, args.warmup_epochs, args.t_max)
    criterion = BCEJLoss(lam=args.lam, tpr_weight=args.tpr_weight)

    best_val_loss = float("inf")
    best_state, best_epoch, patience_cnt = None, 0, 0

    print("\n" + "=" * 70)
    print("STARTING MOBILE FINE-TUNING")
    print("=" * 70)
    t0 = time.time()

    for epoch in range(1, args.max_epochs + 1):
        tr_loss, _ = run_epoch(model, train_loader, criterion, device, optimizer, scheduler)
        val_loss, val_m = run_epoch(model, val_loader, criterion, device)

        print(f"Epoch {epoch:02d}/{args.max_epochs} | Train Loss={tr_loss:.4f} Val Loss={val_loss:.4f} "
              f"Val F1={val_m['F1']:.3f} Val TPR={val_m['TPR']:.3f} Val FPR={val_m['FPR']:.3f} "
              f"Time={int(time.time() - t0)}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = snapshot_state(model)
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}! Best epoch: {best_epoch} (Val Loss: {best_val_loss:.4f})")
                break

    restore_state(model, best_state)

    print("\n" + "=" * 70)
    print("FINAL POST-TRAINING PERFORMANCE")
    print("=" * 70)
    _, final_val_m = run_epoch(model, val_loader, criterion, device)
    print(format_metrics_block(final_val_m, "Validation Split"))
    _, final_test_m = run_epoch(model, test_loader, criterion, device)
    print(format_metrics_block(final_test_m, "Test Split"))

    ckpt_save_path = f"{args.output_dir}/checkpoint_mobile_best.pt"
    print(f"\nSaving best checkpoint to {ckpt_save_path}...")
    save_fused_checkpoint(ckpt_save_path, best_state, age_mean, age_std, best_epoch, args.resize)


if __name__ == "__main__":
    main()
