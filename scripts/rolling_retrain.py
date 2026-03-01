#!/usr/bin/env python
"""Rolling retraining helper for the Time-Series-Library.

This library's `custom` dataset loader (`Dataset_Custom`) always splits the given
CSV into train/val/test by fixed ratios (70/10/20). To do a *rolling* retrain,
we create a sliced CSV containing only the most recent window and then call
`run.py` on that sliced file.

Expected CSV schema: a `date` column plus one or more feature columns, including
`--target`.

Example:
  /home/vandy/Work/Time-Series-Library/.venv/bin/python scripts/rolling_retrain.py \
    --input_csv data/usage_1.csv \
    --window_days 730 \
    --out_dir data/_rolling \
    -- \
    --task_name short_term_forecast --is_training 1 --model TimesNet --data custom \
    --root_path ./data/_rolling --data_path usage_1_rolling.csv \
    --features S --target Usage --freq d --seq_len 365 --label_len 30 --pred_len 1 \
    --enc_in 1 --dec_in 1 --c_out 1 --d_model 64 --n_heads 8 --e_layers 2 --d_layers 1 \
    --d_ff 128 --top_k 5 --num_kernels 6 --learning_rate 0.0005 --train_epochs 5 \
    --batch_size 32 --patience 2 --itr 1 --des rolling --loss MSE --embed timeF --num_workers 0

Notes:
- Everything after `--` is passed directly to `run.py`.
- Make sure `--root_path/--data_path` point at the generated rolling file.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _resolve_python_from_argv0() -> str:
    # Use the current interpreter by default (works well when invoked with the venv python).
    return sys.executable


def _slice_df(df: pd.DataFrame, *, window_rows: int | None, window_days: int | None) -> pd.DataFrame:
    if 'date' not in df.columns:
        raise ValueError("Input CSV must have a 'date' column.")

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date')

    if window_days is not None:
        end = df['date'].max()
        start = end - pd.Timedelta(days=window_days)
        df = df[df['date'] > start]

    if window_rows is not None:
        df = df.tail(window_rows)

    return df


def main() -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--input_csv', required=True, help='Path to full-history CSV (must contain date column).')
    parser.add_argument('--out_dir', default='data/_rolling', help='Directory to write the rolling CSV into.')
    parser.add_argument('--out_name', default=None, help='Output filename. Default: <stem>_rolling.csv')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--window_days', type=int, help='Keep only rows within the last N days (based on date column).')
    group.add_argument('--window_rows', type=int, help='Keep only the last N rows.')

    parser.add_argument('--python', default=_resolve_python_from_argv0(), help='Python executable to run run.py with.')

    # Everything after `--` is forwarded to run.py
    parser.add_argument('run_args', nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if not args.run_args or args.run_args[0] != '--':
        raise SystemExit("Missing '--' separator before run.py args. Example: scripts/rolling_retrain.py ... -- --task_name ...")

    run_args = args.run_args[1:]

    input_path = Path(args.input_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_name = args.out_name
    if out_name is None:
        out_name = f"{input_path.stem}_rolling.csv"

    out_path = out_dir / out_name

    df = pd.read_csv(input_path)
    sliced = _slice_df(df, window_rows=args.window_rows, window_days=args.window_days)
    if len(sliced) < 10:
        raise SystemExit(f"Rolling window is too small after slicing ({len(sliced)} rows).")

    sliced.to_csv(out_path, index=False)
    print(f"Wrote rolling CSV: {out_path} ({len(sliced)} rows)")

    cmd = [args.python, '-u', 'run.py', *run_args]
    print('Running:', ' '.join(cmd))

    env = os.environ.copy()
    return subprocess.call(cmd, env=env)


if __name__ == '__main__':
    raise SystemExit(main())
