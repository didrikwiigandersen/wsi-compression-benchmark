import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from wsi_compression.utils.classes.Result import Result

# ---------- basic ----------
def results_to_df(results: List[Result]) -> pd.DataFrame:
    return pd.DataFrame([{
        "tile_id": r.tile_data.id,
        "codec":   r.codec,
        "raw_bytes": r.raw_bytes,
        "cr": r.cr,
        "enc_ms": r.enc_ms,
        "dec_ms": r.dec_ms,
        "ssim":   r.ssim,
    } for r in results])

# ---------- merge helpers ----------
def _prep(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    keep = ["tile_id", "cr", "raw_bytes", "enc_ms", "dec_ms", "ssim"]
    out = df[keep].copy()
    out.columns = [f"{c}_{tag}" if c != "tile_id" else "tile_id" for c in out.columns]
    return out

def merge_on_tile3(df_jpg: pd.DataFrame, df_jxl: pd.DataFrame, df_j2k: pd.DataFrame) -> pd.DataFrame:
    a = _prep(df_jpg, "jpg")
    b = _prep(df_jxl, "jxl")
    c = _prep(df_j2k, "j2k")
    df = a.merge(b, on="tile_id", how="inner").merge(c, on="tile_id", how="inner")

    # convenience: derived bytes & gains
    df["bytes_jpg"] = df["raw_bytes_jpg"] / df["cr_jpg"]
    df["bytes_jxl"] = df["raw_bytes_jxl"] / df["cr_jxl"]
    df["bytes_j2k"] = df["raw_bytes_j2k"] / df["cr_j2k"]

    # % savings relative to JPEG (positive = better than JPEG)
    df["gain_jxl_vs_jpg"] = 1.0 - (df["cr_jpg"] / df["cr_jxl"])
    df["gain_j2k_vs_jpg"] = 1.0 - (df["cr_jpg"] / df["cr_j2k"])

    return df

# ---------- plots ----------
def plot_scatter_pair(df: pd.DataFrame, xcol: str, ycol: str,
                      xlabel: str, ylabel: str, title: str):
    x = df[xcol].to_numpy(); y = df[ycol].to_numpy()
    lo = min(x.min(), y.min()); hi = max(x.max(), y.max())
    plt.figure()
    plt.scatter(x, y, s=12, alpha=0.5)
    plt.plot([lo, hi], [lo, hi], "--", linewidth=1)
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout(); plt.show()

def plot_ecdf_overlay3(df3: pd.DataFrame):
    def ecdf(v):
        x = np.sort(v); y = np.arange(1, len(x)+1) / len(x); return x, y
    x1,y1 = ecdf(df3["cr_jpg"])
    x2,y2 = ecdf(df3["cr_jxl"])
    x3,y3 = ecdf(df3["cr_j2k"])
    plt.figure()
    plt.plot(x1, y1, label="JPEG")
    plt.plot(x2, y2, label="JXL")
    plt.plot(x3, y3, label="J2K")
    plt.xscale("log"); plt.xlabel("Compression Ratio (raw/bytes)"); plt.ylabel("ECDF")
    plt.title("ECDF of Compression Ratios (JPEG, JXL, J2K)")
    plt.legend(); plt.tight_layout(); plt.show()

def plot_gain_hist_pair(df3: pd.DataFrame, num_bins=40, baseline="jpg", other="jxl", title=None):
    # gain = 100 * (1 - CR_baseline / CR_other)
    gain_col = f"gain_{other}_vs_{baseline}"
    if gain_col not in df3.columns:
        raise KeyError(f"{gain_col} not found. Did you call merge_on_tile3?")
    plt.figure()
    plt.hist(df3[gain_col] * 100.0, bins=num_bins)
    plt.xlabel(f"% bytes saved by {other.upper()} vs {baseline.upper()}")
    plt.ylabel("Tile count")
    plt.title(title or f"% bytes saved by {other.upper()} vs {baseline.upper()}")
    plt.tight_layout(); plt.show()

def plot_box3(df3: pd.DataFrame):
    plt.figure()
    plt.boxplot([df3["cr_jpg"], df3["cr_jxl"], df3["cr_j2k"]],
                labels=["JPEG", "JXL", "J2K"], showfliers=False)
    plt.ylabel("Compression Ratio (raw/bytes)")
    plt.title("Compression Ratio distribution by codec")
    plt.tight_layout(); plt.show()

# ---------- summaries ----------
def print_summary_pairs(df3: pd.DataFrame):
    def summarize(baseline, other):
        bytes_base = df3[f"bytes_{baseline}"].sum()
        bytes_other = df3[f"bytes_{other}"].sum()
        win_pct = (df3[f"cr_{other}"] > df3[f"cr_{baseline}"]).mean() * 100.0
        med_gain = np.median(df3[f"gain_{other}_vs_{baseline}"]) * 100.0
        p90_gain = np.percentile(df3[f"gain_{other}_vs_{baseline}"] * 100.0, 90)
        total_save = 100.0 * (1.0 - bytes_other / bytes_base)
        print(f"{other.upper()} vs {baseline.upper()}:")
        print(f"  • Tiles where {other.upper()} > {baseline.upper()} (CR): {win_pct:.1f}%")
        print(f"  • Median per-tile savings: {med_gain:.2f}%   (P90: {p90_gain:.2f}%)")
        print(f"  • Total bytes saved on sample: {total_save:.2f}%")

    summarize("jpg", "jxl")
    summarize("jpg", "j2k")
