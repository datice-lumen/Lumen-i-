"""Compare slavica's reference predictions.csv against the refactor's predictions_refactor.csv.

Reports row counts, binary class agreement, probability deltas, and lists images
whose binary prediction flipped between the two runs.

Usage:
    python compare_runs.py
"""

import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Diff slavica vs refactor predictions")
    parser.add_argument(
        "--old",
        default="/home/datice/data/baseline_eval/predictions.csv",
        help="Slavica's reference predictions CSV",
    )
    parser.add_argument(
        "--new",
        default="/home/datice/data/baseline_eval/predictions_refactor.csv",
        help="Refactor predictions CSV",
    )
    parser.add_argument(
        "--show-flips",
        type=int,
        default=10,
        help="Print this many flipped images (default: 10, 0 to skip)",
    )
    args = parser.parse_args()

    old = pd.read_csv(args.old)
    new = pd.read_csv(args.new)

    m = old.merge(new, on="image_name", suffixes=("_old", "_new"))

    print("=" * 60)
    print("COVERAGE")
    print("=" * 60)
    print(f"Old rows:        {len(old)}")
    print(f"New rows:        {len(new)}")
    print(f"Matched rows:    {len(m)}")
    only_old = set(old.image_name) - set(new.image_name)
    only_new = set(new.image_name) - set(old.image_name)
    print(f"Only in old:     {len(only_old)}")
    print(f"Only in new:     {len(only_new)}")

    print("\n" + "=" * 60)
    print("BINARY AGREEMENT")
    print("=" * 60)
    agree = (m.target_old == m.target_new).mean()
    flips = m[m.target_old != m.target_new]
    print(f"Agreement:       {agree:.4f} ({agree*100:.2f}%)")
    print(f"Flipped images:  {len(flips)}")
    if len(flips):
        flip_0_to_1 = ((flips.target_old == 0) & (flips.target_new == 1)).sum()
        flip_1_to_0 = ((flips.target_old == 1) & (flips.target_new == 0)).sum()
        print(f"  0 -> 1:        {flip_0_to_1}")
        print(f"  1 -> 0:        {flip_1_to_0}")

    print("\n" + "=" * 60)
    print("PROBABILITY DELTAS")
    print("=" * 60)
    diff = (m.conf_old - m.conf_new).abs()
    print(f"Mean |conf diff|: {diff.mean():.6f}")
    print(f"Max  |conf diff|: {diff.max():.6f}")
    print(f"P50:              {diff.median():.6f}")
    print(f"P95:              {diff.quantile(0.95):.6f}")
    print(f"P99:              {diff.quantile(0.99):.6f}")

    if args.show_flips and len(flips):
        print("\n" + "=" * 60)
        print(f"FIRST {min(args.show_flips, len(flips))} FLIPPED IMAGES")
        print("=" * 60)
        sample = flips.head(args.show_flips)[
            ["image_name", "target_old", "target_new", "conf_old", "conf_new"]
        ]
        print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
