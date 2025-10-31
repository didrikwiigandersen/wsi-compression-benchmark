"""
Helpers to run analytics on the results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import List
from wsi_compression.utils.classes.Result import Result

def results_to_df(results: List[Result]) -> pd.DataFrame:
    rows = []
    for r in results:
        t = r.tile_data
        rows.append({
            "tile_id": t.id,            # use tile's id directly
            "codec": r.codec,
            "raw_bytes": r.raw_bytes,
            "cr": r.cr,
            "enc_ms": r.enc_ms,
            "dec_ms": r.dec_ms,
            "ssim": r.ssim,
        })
    return pd.DataFrame(rows)

def merge_on_tile(df_jpg: pd.DataFrame, df_jxl: pd.DataFrame) -> pd.DataFrame:
    # Keep only columns we need; suffix to avoid collisions
    keep = ["tile_id", "cr", "raw_bytes", "enc_ms", "dec_ms", "ssim"]
    df = df_jpg[keep].merge(
        df_jxl[keep],
        on="tile_id",
        suffixes=("_jpg", "_jxl"),
        how="inner"
    )
    # convenience columns
    df["bytes_jpg"] = df["raw_bytes_jpg"] / df["cr_jpg"]
    df["bytes_jxl"] = df["raw_bytes_jxl"] / df["cr_jxl"]
    df["gain_pct"]  = 1.0 - (df["cr_jpg"] / df["cr_jxl"])   # >0 means JXL smaller bytes
    df["ratio_jxl_over_jpg"] = df["cr_jxl"] / df["cr_jpg"]  # >1 means JXL better CR
    return df

def plot_scatter_cr(df: pd.DataFrame):
    x = df["cr_jpg"].to_numpy()
    y = df["cr_jxl"].to_numpy()
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    plt.figure()
    plt.scatter(x, y, s=12, alpha=0.5)
    plt.plot([lo, hi], [lo, hi], "--", linewidth=1)  # y = x reference
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("CR (JPEG)  raw/bytes"); plt.ylabel("CR (JXL)  raw/bytes")
    plt.title("Per-tile Compression Ratio: JPEG vs JXL (higher is better)")
    plt.tight_layout(); plt.show()

def plot_ecdf_overlay(df_jpg: pd.DataFrame, df_jxl: pd.DataFrame):
    def ecdf(a):
        xs = np.sort(a)
        ys = np.arange(1, len(xs)+1) / len(xs)
        return xs, ys
    x1,y1 = ecdf(df_jpg["cr"])
    x2,y2 = ecdf(df_jxl["cr"])
    plt.figure()
    plt.plot(x1, y1, label="JPEG")
    plt.plot(x2, y2, label="JXL")
    plt.xscale("log")
    plt.xlabel("Compression Ratio (raw/bytes)"); plt.ylabel("ECDF")
    plt.title("ECDF of Compression Ratios")
    plt.legend(); plt.tight_layout(); plt.show()

def plot_gain_hist(df: pd.DataFrame, bins=40):
    plt.figure()
    plt.hist(df["gain_pct"] * 100.0, bins=bins)
    plt.xlabel("Size saving vs JPEG (%)  = 100 * (1 - CR_jpg/CR_jxl)")
    plt.ylabel("Tile count")
    plt.title("Per-tile % bytes saved by JXL vs JPEG (right = better)")
    plt.tight_layout(); plt.show()

def plot_box(df_jpg: pd.DataFrame, df_jxl: pd.DataFrame):
    plt.figure()
    plt.boxplot([df_jpg["cr"], df_jxl["cr"]], labels=["JPEG", "JXL"], showfliers=False)
    plt.ylabel("Compression Ratio (raw/bytes)")
    plt.title("CR distribution by codec")
    plt.tight_layout(); plt.show()

def print_summary(df_pairs: pd.DataFrame):
    wins = (df_pairs["cr_jxl"] > df_pairs["cr_jpg"]).mean()
    med_gain = np.median(df_pairs["gain_pct"]) * 100.0
    p90_gain = np.percentile(df_pairs["gain_pct"]*100.0, 90)
    # Aggregate storage effect on this sample set
    total_jpg = df_pairs["bytes_jpg"].sum()
    total_jxl = df_pairs["bytes_jxl"].sum()
    total_save_pct = 100.0 * (1.0 - total_jxl / total_jpg)
    print(f"% tiles where JXL > JPEG (CR): {wins*100:.1f}%")
    print(f"Median per-tile savings vs JPEG: {med_gain:.2f}%")
    print(f"90th pct per-tile savings: {p90_gain:.2f}%")
    print(f"Total bytes saved on sample: {total_save_pct:.2f}%")
