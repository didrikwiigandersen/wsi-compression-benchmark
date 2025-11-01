"""
Helper functions for analysis.

Made with GPT, not for production.
"""

# ----------------------- Packages ----------------------- #
import matplotlib.pyplot as plt
from typing import List, Optional, Iterable, Union
from wsi_compression.utils.classes.Result import Result

# ----------------------- Functions ----------------------- #

import numpy as np, pandas as pd
from scipy import stats

def _prep_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    # expects columns: tile_id, codec, cr (codec in {'jpeg','jxl','j2k'})
    df = df_long.copy()
    df["codec"] = df["codec"].str.lower().replace({"jpg": "jpeg"})
    wide = df.pivot_table(index="tile_id", columns="codec", values="cr", aggfunc="first")
    need = [c for c in ["jxl","jpeg","j2k"] if c in wide.columns]
    wide = wide.dropna(subset=need)
    return wide[need]

def _bootstrap_ci(x: np.ndarray, stat_fn, n_boot=10_000, seed=42, alpha=0.05):
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = stat_fn(x[idx])
    lo = np.quantile(boots, alpha/2)
    hi = np.quantile(boots, 1 - alpha/2)
    return float(lo), float(hi)

def jxl_superiority_tests(df_long: pd.DataFrame, alpha=0.05, n_boot=10_000):
    W = _prep_wide(df_long)  # columns present: jxl, jpeg, j2k
    out = {}

    def _one(comp: str):
        x = W["jxl"].to_numpy()
        y = W[comp].to_numpy()
        # Differences & ratios
        diff = x - y
        log_ratio = np.log(x / y)

        # Paired Wilcoxon (robust) — H1: JXL > comparator
        w_stat, p_wilc = stats.wilcoxon(x, y, alternative="greater", zero_method="pratt")

        # Paired t-test on log-ratio (means multiplicative) — H1: log(x/y) > 0
        t_stat, p_t = stats.ttest_rel(log_ratio, np.zeros_like(log_ratio), alternative="greater")

        # Effect sizes
        med_diff = float(np.median(diff))
        med_gain = float(np.median(x / y))           # median multiplicative gain
        med_diff_ci = _bootstrap_ci(diff, np.median, n_boot=n_boot, alpha=alpha)
        med_gain_ci = _bootstrap_ci(x / y, np.median, n_boot=n_boot, alpha=alpha)

        return {
            "n_tiles": int(len(diff)),
            "wilcoxon_stat": float(w_stat), "p_wilcoxon": float(p_wilc),
            "ttest_logratio_t": float(t_stat), "p_ttest_logratio": float(p_t),
            "median_diff_cr": med_diff, "median_diff_cr_ci": med_diff_ci,
            "median_gain_ratio": med_gain, "median_gain_ratio_ci": med_gain_ci,
            "pct_tiles_jxl_higher": float((diff > 0).mean()*100.0),
        }

    # Two paired comparisons
    res_jpeg = _one("jpeg") if "jpeg" in W.columns else None
    res_j2k  = _one("j2k")  if "j2k"  in W.columns else None

    # Holm correction for two tests
    pvals = [r["p_wilcoxon"] for r in [res_jpeg, res_j2k] if r is not None]
    order = np.argsort(pvals)
    adj = [None]*len(pvals)
    for rank, idx in enumerate(order, start=1):
        adj[idx] = min((len(pvals) - rank + 1)*pvals[idx], 1.0)

    k = 0
    if res_jpeg is not None:
        res_jpeg["p_wilcoxon_holm"] = adj[k]; k+=1
    if res_j2k is not None:
        res_j2k["p_wilcoxon_holm"] = adj[k] if len(adj)>k else None

    return {"JXL_vs_JPEG": res_jpeg, "JXL_vs_J2K": res_j2k}

def _results_to_df_any(x: Union[List[Result], pd.DataFrame]) -> pd.DataFrame:
    """

    :param x:
    :return:
    """
    if isinstance(x, pd.DataFrame):
        return x.copy()
    return pd.DataFrame([r.as_dict() for r in x])

def plot_cr_scatter_by_codec(
    jpeg_results: Union[List[Result], pd.DataFrame],
    jxl_results: Optional[Union[List[Result], pd.DataFrame]] = None,
    j2k_results: Optional[Union[List[Result], pd.DataFrame]] = None,
    *,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (10, 5),
    palette: Optional[dict] = None,
    title: str = "Per-tile Compression Ratio by Codec",
    savepath: Optional[str] = None,
    show: bool = True,
) -> pd.DataFrame:
    """
    Scatter: x = tile_id, y = compression ratio (cr), colored by codec (JPEG/JXL/J2K).
    Returns the tidy DataFrame used for plotting.
    """
    parts: List[pd.DataFrame] = []
    for chunk in (jpeg_results, jxl_results, j2k_results):
        if chunk is not None:
            parts.append(_results_to_df_any(chunk))
    if not parts:
        raise ValueError("No results provided to plot.")

    df = pd.concat(parts, ignore_index=True)
    # Keep only what we need
    df = df[["tile_id", "codec", "cr"]].copy()
    # Normalize codec labels
    df["codec"] = df["codec"].str.lower().replace({"jpg": "jpeg"})
    df = df.sort_values("tile_id")

    if palette is None:
        palette = {"jpeg": "#1f77b4", "jxl": "#ff7f0e", "j2k": "#2ca02c"}

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    for codec in ("jpeg", "jxl", "j2k"):
        sub = df[df["codec"] == codec]
        if not sub.empty:
            ax.plot(
                sub["tile_id"],
                sub["cr"],
                marker="o",
                markersize=3.5,
                linewidth=1.0,
                alpha=0.8,
                label=codec.upper(),
                color=palette.get(codec),
            )

    ax.set_xlabel("Tile ID")
    ax.set_ylabel("Compression Ratio (raw_bytes / encoded_bytes)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=150)
    if show:
        plt.show()

    return df
