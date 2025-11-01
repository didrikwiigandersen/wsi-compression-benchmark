"""
Helper functions for analysis.
"""

# ----------------------- Packages ----------------------- #
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional, Iterable, Union
from wsi_compression.utils.classes.Result import Result

# ----------------------- Functions ----------------------- #

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
