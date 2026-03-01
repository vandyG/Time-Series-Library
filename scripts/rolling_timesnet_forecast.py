import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import TimesNet
from utils.timefeatures import time_features


@dataclass
class TrainConfig:
    task_name: str = "short_term_forecast"
    model: str = "TimesNet"
    data: str = "custom"
    features: str = "S"
    target: str = "Usage"
    freq: str = "d"
    seq_len: int = 365
    label_len: int = 30
    pred_len: int = 7
    seasonal_patterns: str = "Yearly"
    enc_in: int = 1
    dec_in: int = 1
    c_out: int = 1
    e_layers: int = 2
    d_layers: int = 1
    d_model: int = 64
    d_ff: int = 128
    top_k: int = 5
    batch_size: int = 32
    train_epochs: int = 5
    patience: int = 2
    learning_rate: float = 0.0005
    num_workers: int = 0
    itr: int = 1
    gpu: int = 0
    use_gpu: bool = True
    n_heads: int = 8
    expand: int = 2
    d_conv: int = 4
    factor: int = 1
    embed: str = "timeF"
    distil: bool = True
    dropout: float = 0.1
    num_kernels: int = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rolling TimesNet forecast with retraining")
    parser.add_argument("--root_path", type=str, default="./data")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["usage_1.csv", "usage_2.csv", "usage_2_1.csv"],
        help="Dataset filenames under root_path",
    )
    parser.add_argument("--output_dir", type=str, default="./rolling_forecast_outputs")
    parser.add_argument("--forecast_start", type=str, default="2024-01-01")
    parser.add_argument("--forecast_end", type=str, default="2024-12-31")
    parser.add_argument("--date_col", type=str, default="date")
    parser.add_argument("--target_col", type=str, default="Usage")
    parser.add_argument("--python_bin", type=str, default="python")
    parser.add_argument("--run_file", type=str, default="run.py")
    parser.add_argument("--checkpoint_root", type=str, default="./checkpoints")
    parser.add_argument("--pred_len", type=int, default=7)
    parser.add_argument(
        "--rolling_horizon_days",
        type=int,
        default=1,
        help="Forecast window size per iteration (default: 1 day)",
    )
    parser.add_argument(
        "--rolling_step_days",
        type=int,
        default=None,
        help="How many days to move the rolling window each iteration (defaults to rolling_horizon_days)",
    )
    parser.add_argument(
        "--allow_partial_window",
        action="store_true",
        help="Allow the final window to be shorter than rolling_horizon_days",
    )
    parser.add_argument("--seq_len", type=int, default=365)
    parser.add_argument("--label_len", type=int, default=30)
    parser.add_argument("--train_epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true", help="Force CPU training/inference")
    parser.add_argument(
        "--calibration_method",
        type=str,
        default="none",
        choices=["none", "bias", "scale", "affine"],
        help="Online calibration using past realized forecast errors",
    )
    parser.add_argument(
        "--calibration_lookback_points",
        type=int,
        default=28,
        help="How many past forecast points to use for online calibration",
    )
    parser.add_argument(
        "--calibration_min_points",
        type=int,
        default=7,
        help="Minimum realized points before applying calibration",
    )
    parser.add_argument(
        "--calibration_scale_min",
        type=float,
        default=0.5,
        help="Lower bound for multiplicative scale in calibration",
    )
    parser.add_argument(
        "--calibration_scale_max",
        type=float,
        default=1.5,
        help="Upper bound for multiplicative scale in calibration",
    )
    return parser.parse_args()


def make_train_config(args: argparse.Namespace) -> TrainConfig:
    cfg = TrainConfig()
    cfg.pred_len = args.pred_len
    cfg.seq_len = args.seq_len
    cfg.label_len = args.label_len
    cfg.train_epochs = args.train_epochs
    cfg.patience = args.patience
    cfg.batch_size = args.batch_size
    cfg.learning_rate = args.learning_rate
    cfg.gpu = args.gpu
    cfg.use_gpu = not args.cpu
    return cfg


def build_setting(model_id: str, des: str, cfg: TrainConfig) -> str:
    return (
        f"{cfg.task_name}_{model_id}_{cfg.model}_{cfg.data}"
        f"_ft{cfg.features}_sl{cfg.seq_len}_ll{cfg.label_len}_pl{cfg.pred_len}"
        f"_dm{cfg.d_model}_nh{cfg.n_heads}_el{cfg.e_layers}_dl{cfg.d_layers}"
        f"_df{cfg.d_ff}_expand{cfg.expand}_dc{cfg.d_conv}_fc{cfg.factor}"
        f"_eb{cfg.embed}_dt{cfg.distil}_{des}_0"
    )


def rolling_windows(
    start: pd.Timestamp,
    end: pd.Timestamp,
    horizon_days: int,
    step_days: int,
    allow_partial_window: bool,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    windows = []
    current = start
    while current <= end:
        window_end = current + pd.Timedelta(days=horizon_days - 1)
        if window_end > end:
            if not allow_partial_window:
                break
            window_end = end

        windows.append((current, window_end))
        current = current + pd.Timedelta(days=step_days)

    return windows


def run_training(
    cfg: TrainConfig,
    python_bin: str,
    run_file: str,
    root_path: str,
    data_path: str,
    model_id: str,
    des: str,
) -> None:
    cmd = [
        python_bin,
        "-u",
        run_file,
        "--task_name",
        cfg.task_name,
        "--is_training",
        "1",
        "--model_id",
        model_id,
        "--model",
        cfg.model,
        "--data",
        cfg.data,
        "--root_path",
        root_path,
        "--data_path",
        data_path,
        "--features",
        cfg.features,
        "--target",
        cfg.target,
        "--freq",
        cfg.freq,
        "--seq_len",
        str(cfg.seq_len),
        "--label_len",
        str(cfg.label_len),
        "--pred_len",
        str(cfg.pred_len),
        "--seasonal_patterns",
        cfg.seasonal_patterns,
        "--enc_in",
        str(cfg.enc_in),
        "--dec_in",
        str(cfg.dec_in),
        "--c_out",
        str(cfg.c_out),
        "--e_layers",
        str(cfg.e_layers),
        "--d_layers",
        str(cfg.d_layers),
        "--d_model",
        str(cfg.d_model),
        "--d_ff",
        str(cfg.d_ff),
        "--top_k",
        str(cfg.top_k),
        "--batch_size",
        str(cfg.batch_size),
        "--train_epochs",
        str(cfg.train_epochs),
        "--patience",
        str(cfg.patience),
        "--learning_rate",
        str(cfg.learning_rate),
        "--num_workers",
        str(cfg.num_workers),
        "--itr",
        str(cfg.itr),
        "--gpu",
        str(cfg.gpu),
        "--des",
        des,
    ]
    if cfg.use_gpu:
        cmd.append("--use_gpu")
    else:
        cmd.append("--no_use_gpu")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def make_model_args(cfg: TrainConfig, use_gpu: bool, gpu: int) -> SimpleNamespace:
    return SimpleNamespace(
        task_name=cfg.task_name,
        model=cfg.model,
        seq_len=cfg.seq_len,
        label_len=cfg.label_len,
        pred_len=cfg.pred_len,
        top_k=cfg.top_k,
        d_model=cfg.d_model,
        d_ff=cfg.d_ff,
        e_layers=cfg.e_layers,
        d_layers=cfg.d_layers,
        factor=cfg.factor,
        enc_in=cfg.enc_in,
        dec_in=cfg.dec_in,
        c_out=cfg.c_out,
        dropout=cfg.dropout,
        embed=cfg.embed,
        freq=cfg.freq,
        activation="gelu",
        num_kernels=cfg.num_kernels,
        use_gpu=use_gpu,
        use_multi_gpu=False,
        gpu=gpu,
    )


def fit_scaler_like_custom_split(series: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    num_train = int(len(series) * 0.7)
    num_train = max(num_train, 1)
    scaler.fit(series[:num_train].reshape(-1, 1))
    return scaler


def infer_next_week(
    train_df: pd.DataFrame,
    checkpoint_path: Path,
    cfg: TrainConfig,
    target_col: str,
    date_col: str,
) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    if len(train_df) < cfg.seq_len:
        raise ValueError(f"Need at least seq_len={cfg.seq_len} rows for inference, got {len(train_df)}")

    device = torch.device(
        f"cuda:{cfg.gpu}" if cfg.use_gpu and torch.cuda.is_available() else "cpu"
    )
    model_args = make_model_args(cfg, cfg.use_gpu, cfg.gpu)
    model = TimesNet.Model(model_args).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    values = train_df[target_col].to_numpy(dtype=np.float32)
    scaler = fit_scaler_like_custom_split(values)
    scaled = scaler.transform(values.reshape(-1, 1)).reshape(-1)

    x_values = scaled[-cfg.seq_len:]
    x_dates = pd.to_datetime(train_df[date_col].iloc[-cfg.seq_len:]).to_numpy()
    train_end = pd.Timestamp(train_df[date_col].iloc[-1])
    future_dates = pd.date_range(start=train_end + pd.Timedelta(days=1), periods=cfg.pred_len, freq="D")

    dec_known = x_values[-cfg.label_len:]
    dec_values = np.concatenate([dec_known, np.zeros(cfg.pred_len, dtype=np.float32)])

    dec_dates = np.concatenate([x_dates[-cfg.label_len:], future_dates.to_numpy()])
    x_mark = time_features(pd.to_datetime(x_dates), freq=cfg.freq).transpose(1, 0).astype(np.float32)
    dec_mark = time_features(pd.to_datetime(dec_dates), freq=cfg.freq).transpose(1, 0).astype(np.float32)

    x_tensor = torch.from_numpy(x_values.reshape(1, cfg.seq_len, 1)).float().to(device)
    x_mark_tensor = torch.from_numpy(x_mark.reshape(1, cfg.seq_len, -1)).float().to(device)
    dec_tensor = torch.from_numpy(dec_values.reshape(1, cfg.label_len + cfg.pred_len, 1)).float().to(device)
    dec_mark_tensor = torch.from_numpy(dec_mark.reshape(1, cfg.label_len + cfg.pred_len, -1)).float().to(device)

    with torch.no_grad():
        pred_scaled = model(x_tensor, x_mark_tensor, dec_tensor, dec_mark_tensor)
    pred_scaled = pred_scaled.detach().cpu().numpy().reshape(-1, 1)
    pred = scaler.inverse_transform(pred_scaled).reshape(-1)
    return future_dates, pred


def compute_metrics(actual: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    err = pred - actual
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    denom = np.where(np.abs(actual) < 1e-8, 1e-8, np.abs(actual))
    mape = float(np.mean(np.abs(err) / denom) * 100)
    return {"mae": mae, "rmse": rmse, "mape": mape}


def calibrate_predictions(
    raw_pred: np.ndarray,
    history_raw_pred: np.ndarray,
    history_actual: np.ndarray,
    method: str,
    scale_min: float,
    scale_max: float,
) -> np.ndarray:
    if method == "none" or len(history_raw_pred) == 0:
        return raw_pred

    if method == "bias":
        bias = float(np.mean(history_actual - history_raw_pred))
        return raw_pred + bias

    if method == "scale":
        denom = np.where(np.abs(history_raw_pred) < 1e-8, 1e-8, history_raw_pred)
        scale = float(np.mean(history_actual / denom))
        scale = float(np.clip(scale, scale_min, scale_max))
        return raw_pred * scale

    if method == "affine":
        x = history_raw_pred.astype(float)
        y = history_actual.astype(float)
        x_mean = float(np.mean(x))
        y_mean = float(np.mean(y))
        x_var = float(np.var(x))
        if x_var < 1e-12:
            slope = 1.0
        else:
            cov = float(np.mean((x - x_mean) * (y - y_mean)))
            slope = cov / x_var
        slope = float(np.clip(slope, scale_min, scale_max))
        intercept = y_mean - slope * x_mean
        return raw_pred * slope + intercept

    return raw_pred


def make_plots(df: pd.DataFrame, output_dir: Path, dataset_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 5))
    plt.plot(df["date"], df["actual"], label="Actual", linewidth=1.5)
    plt.plot(df["date"], df["prediction"], label="Forecast", linewidth=1.5)
    plt.title(f"Rolling Weekly Forecast vs Actual - {dataset_name}")
    plt.xlabel("Date")
    plt.ylabel("Usage")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "actual_vs_forecast.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(df["actual"], df["prediction"], alpha=0.6)
    min_v = min(df["actual"].min(), df["prediction"].min())
    max_v = max(df["actual"].max(), df["prediction"].max())
    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--")
    plt.title(f"Prediction Scatter - {dataset_name}")
    plt.xlabel("Actual")
    plt.ylabel("Forecast")
    plt.tight_layout()
    plt.savefig(output_dir / "scatter_actual_vs_forecast.png", dpi=150)
    plt.close()

    weekly_mae = (
        df.assign(abs_err=(df["prediction"] - df["actual"]).abs())
        .groupby("week_start", as_index=False)["abs_err"]
        .mean()
    )
    plt.figure(figsize=(14, 4))
    plt.plot(weekly_mae["week_start"], weekly_mae["abs_err"], marker="o")
    plt.title(f"Weekly MAE - {dataset_name}")
    plt.xlabel("Week start")
    plt.ylabel("MAE")
    plt.tight_layout()
    plt.savefig(output_dir / "weekly_mae.png", dpi=150)
    plt.close()


def run_for_dataset(
    dataset_path: Path,
    args: argparse.Namespace,
    cfg: TrainConfig,
    output_root: Path,
    temp_root: Path,
) -> pd.DataFrame:
    dataset_name = dataset_path.stem
    df = pd.read_csv(dataset_path)
    if args.date_col not in df.columns or args.target_col not in df.columns:
        raise ValueError(
            f"Dataset {dataset_path.name} must contain columns '{args.date_col}' and '{args.target_col}'"
        )

    df = df[[args.date_col, args.target_col]].copy()
    df[args.date_col] = pd.to_datetime(df[args.date_col])
    df = df.sort_values(args.date_col).drop_duplicates(subset=[args.date_col]).reset_index(drop=True)

    forecast_start = pd.Timestamp(args.forecast_start)
    forecast_end = min(pd.Timestamp(args.forecast_end), pd.Timestamp(df[args.date_col].max()))

    rolling_horizon_days = args.rolling_horizon_days if args.rolling_horizon_days is not None else cfg.pred_len
    rolling_step_days = args.rolling_step_days if args.rolling_step_days is not None else rolling_horizon_days

    windows = rolling_windows(
        forecast_start,
        forecast_end,
        horizon_days=rolling_horizon_days,
        step_days=rolling_step_days,
        allow_partial_window=args.allow_partial_window,
    )

    if not windows:
        print(f"No forecast windows for {dataset_name} in the selected period.")
        return pd.DataFrame()

    temp_root.mkdir(parents=True, exist_ok=True)
    (output_root / dataset_name).mkdir(parents=True, exist_ok=True)

    records: List[Dict] = []
    for window_start, window_end in windows:
        train_end = window_start - pd.Timedelta(days=1)
        train_df = df[df[args.date_col] <= train_end].copy()
        actual_window = df[(df[args.date_col] >= window_start) & (df[args.date_col] <= window_end)].copy()

        if len(actual_window) == 0:
            print(
                f"Skipping window {window_start.date()} to {window_end.date()} for {dataset_name}: "
                "no actual points available in this range"
            )
            continue

        if len(train_df) < (cfg.seq_len + cfg.pred_len):
            print(
                f"Skipping window {window_start.date()} for {dataset_name}: "
                f"insufficient history ({len(train_df)} rows)."
            )
            continue

        temp_name = f"{dataset_name}_train_until_{train_end.strftime('%Y%m%d')}.csv"
        temp_rel = Path("tmp") / temp_name
        temp_abs = temp_root / temp_name
        train_df.to_csv(temp_abs, index=False)

        model_id = f"{dataset_name}_rolling_{window_start.strftime('%Y%m%d')}"
        des = f"win{window_start.strftime('%Y%m%d')}"

        run_training(
            cfg=cfg,
            python_bin=args.python_bin,
            run_file=args.run_file,
            root_path=str(output_root),
            data_path=str(temp_rel),
            model_id=model_id,
            des=des,
        )

        setting = build_setting(model_id=model_id, des=des, cfg=cfg)
        ckpt_path = Path(args.checkpoint_root) / setting / "checkpoint.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        pred_dates, pred_values = infer_next_week(
            train_df=train_df,
            checkpoint_path=ckpt_path,
            cfg=cfg,
            target_col=args.target_col,
            date_col=args.date_col,
        )

        actual_window = actual_window.sort_values(args.date_col)
        actual_values = actual_window[args.target_col].to_numpy(dtype=float)

        effective_len = min(len(actual_values), cfg.pred_len)
        pred_values = pred_values[:effective_len]
        pred_dates = pred_dates[:effective_len]
        actual_values = actual_values[:effective_len]

        raw_pred_values = pred_values.copy()
        if args.calibration_method != "none":
            history_df = pd.DataFrame(records)
            if not history_df.empty:
                lookback = max(int(args.calibration_lookback_points), 1)
                min_points = max(int(args.calibration_min_points), 1)
                history_df = history_df.tail(lookback)
                history_raw_pred = history_df["prediction_raw"].to_numpy(dtype=float)
                history_actual = history_df["actual"].to_numpy(dtype=float)
                if len(history_df) >= min_points:
                    pred_values = calibrate_predictions(
                        raw_pred=raw_pred_values,
                        history_raw_pred=history_raw_pred,
                        history_actual=history_actual,
                        method=args.calibration_method,
                        scale_min=min(args.calibration_scale_min, args.calibration_scale_max),
                        scale_max=max(args.calibration_scale_min, args.calibration_scale_max),
                    )

        metric_values = compute_metrics(actual=actual_values, pred=pred_values)

        for i in range(effective_len):
            records.append(
                {
                    "dataset": dataset_name,
                    "window_start": window_start,
                    "window_end": window_end,
                    "week_start": window_start,
                    "week_end": window_end,
                    "date": pred_dates[i],
                    "prediction_raw": float(raw_pred_values[i]),
                    "prediction": float(pred_values[i]),
                    "actual": float(actual_values[i]),
                    "error": float(pred_values[i] - actual_values[i]),
                    "mae_week": metric_values["mae"],
                    "rmse_week": metric_values["rmse"],
                    "mape_week": metric_values["mape"],
                }
            )

    result_df = pd.DataFrame(records)
    if result_df.empty:
        return result_df

    dataset_out = output_root / dataset_name
    result_df.to_csv(dataset_out / "rolling_forecast.csv", index=False)
    summary = {
        "dataset": dataset_name,
        "calibration_method": args.calibration_method,
        **compute_metrics(result_df["actual"].to_numpy(), result_df["prediction"].to_numpy()),
        "n_points": int(len(result_df)),
    }
    pd.DataFrame([summary]).to_csv(dataset_out / "metrics_summary.csv", index=False)
    make_plots(result_df, dataset_out, dataset_name)
    return result_df


def main() -> None:
    args = parse_args()
    cfg = make_train_config(args)

    rolling_horizon_days = args.rolling_horizon_days if args.rolling_horizon_days is not None else cfg.pred_len
    rolling_step_days = args.rolling_step_days if args.rolling_step_days is not None else rolling_horizon_days
    if rolling_horizon_days <= 0:
        raise ValueError("rolling_horizon_days must be > 0")
    if rolling_step_days <= 0:
        raise ValueError("rolling_step_days must be > 0")

    output_root = Path(args.output_dir).resolve()
    tmp_root = output_root / "tmp"
    output_root.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    all_results = []
    for name in args.datasets:
        dataset_path = Path(args.root_path) / name
        if not dataset_path.exists():
            print(f"Skipping missing dataset: {dataset_path}")
            continue
        print(f"\n=== Processing {dataset_path.name} ===")
        dataset_result = run_for_dataset(
            dataset_path=dataset_path,
            args=args,
            cfg=cfg,
            output_root=output_root,
            temp_root=tmp_root,
        )
        if not dataset_result.empty:
            all_results.append(dataset_result)

    if not all_results:
        print("No results were generated.")
        return

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(output_root / "all_datasets_rolling_forecast.csv", index=False)

    dataset_metrics = (
        combined.groupby("dataset", as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "mae": float(np.mean(np.abs(g["prediction"].to_numpy() - g["actual"].to_numpy()))),
                    "rmse": float(
                        np.sqrt(np.mean((g["prediction"].to_numpy() - g["actual"].to_numpy()) ** 2))
                    ),
                    "mape": float(
                        np.mean(
                            np.abs(g["prediction"].to_numpy() - g["actual"].to_numpy())
                            / np.where(np.abs(g["actual"].to_numpy()) < 1e-8, 1e-8, np.abs(g["actual"].to_numpy()))
                        )
                        * 100
                    ),
                    "n_points": int(len(g)),
                }
            )
        )
        .reset_index(drop=True)
    )
    dataset_metrics.to_csv(output_root / "all_datasets_metrics_summary.csv", index=False)

    print("\nCompleted rolling forecasts.")
    print(f"Outputs saved to: {output_root}")


if __name__ == "__main__":
    main()