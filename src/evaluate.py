"""
Evaluation and Visualization Module for Sentiment Analysis Models.

Calculates standard classification performance metrics, outputs terminal benchmark
reports, and generates visual artifacts (WordClouds, Confusion Matrices, and Distributions).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
# Use non-interactive backend for headless and production environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from wordcloud import WordCloud

logger = logging.getLogger("evaluate")


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    model_name: str = "Model",
) -> dict[str, Any]:
    """
    Compute accuracy, precision, recall, and F1-score across macro and weighted averages.

    Args:
        y_true: Ground truth target labels.
        y_pred: Predicted target labels.
        model_name: Identifier for the model.

    Returns:
        Dictionary containing calculated metrics and the confusion matrix.
    """
    acc = accuracy_score(y_true, y_pred)
    prec_w = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_w = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    prec_m = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_m = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_m = f1_score(y_true, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred)

    return {
        "Model": model_name,
        "Accuracy": acc,
        "Precision (Weighted)": prec_w,
        "Recall (Weighted)": rec_w,
        "F1-Score (Weighted)": f1_w,
        "Precision (Macro)": prec_m,
        "Recall (Macro)": rec_m,
        "F1-Score (Macro)": f1_m,
        "confusion_matrix": cm,
    }


def display_benchmark_table(
    metrics_list: list[dict[str, Any]],
    output_dir: Optional[str | Path] = None,
    filename: str = "benchmark_summary.csv",
) -> pd.DataFrame:
    """
    Format and log an ASCII table of benchmark results to console and export to CSV.

    Args:
        metrics_list: List of metric dictionaries computed for candidate models.
        output_dir: Optional directory to persist CSV summary.
        filename: CSV filename.

    Returns:
        Structured pandas DataFrame summary.
    """
    display_cols = [
        "Model",
        "Accuracy",
        "Precision (Weighted)",
        "Recall (Weighted)",
        "F1-Score (Weighted)",
        "F1-Score (Macro)",
    ]

    clean_records = [
        {k: v for k, v in m.items() if k != "confusion_matrix"} for m in metrics_list
    ]
    summary_df = pd.DataFrame(clean_records)
    summary_df = summary_df.sort_values(by="F1-Score (Weighted)", ascending=False)

    # Format numbers for clean presentation
    formatted_df = summary_df[display_cols].copy()
    for col in display_cols[1:]:
        formatted_df[col] = formatted_df[col].map(lambda x: f"{x:.4f}")

    # Build ASCII presentation border
    col_widths = {col: max(len(col), max(len(str(val)) for val in formatted_df[col])) + 2 for col in display_cols}
    separator = "+" + "+".join("-" * col_widths[col] for col in display_cols) + "+"
    header = "|" + "|".join(f" {col.center(col_widths[col] - 2)} " for col in display_cols) + "|"

    rows = []
    for _, row in formatted_df.iterrows():
        r = "|" + "|".join(f" {str(row[col]).center(col_widths[col] - 2)} " for col in display_cols) + "|"
        rows.append(r)

    ascii_table = "\n" + "\n".join([separator, header, separator] + rows + [separator])
    
    print("\n" + "=" * 80)
    print(" MODEL PERFORMANCE BENCHMARK COMPARISON ".center(80, "="))
    print("=" * 80)
    print(ascii_table)
    print("=" * 80 + "\n")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / filename
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Saved benchmark summary table to '{summary_path}'.")

    return summary_df


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    model_name: str,
    output_path: str | Path,
) -> None:
    """
    Render and save a Confusion Matrix heatmap for the top-performing model.

    Args:
        cm: Confusion matrix numpy array.
        class_names: Labels for axes.
        model_name: Name of evaluated model.
        output_path: Destination image path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        linewidths=0.5,
    )
    plt.title(f"Confusion Matrix: {model_name}", fontsize=14, pad=15)
    plt.xlabel("Predicted Sentiment Label", fontsize=12)
    plt.ylabel("Actual True Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Confusion matrix plot saved to '{output_path}'.")


def plot_distributions(
    df: pd.DataFrame,
    rating_col: str,
    sentiment_col: str,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """
    Render bar charts illustrating rating score distributions and target class balances.

    Args:
        df: Processed DataFrame containing ratings and sentiment classes.
        rating_col: Column with star ratings (1 to 5).
        sentiment_col: Column with mapped sentiment string classes.
        output_dir: Target directory.

    Returns:
        Tuple of (rating_dist_path, class_balance_path).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Rating Distribution Chart
    rating_path = output_dir / "rating_distribution.png"
    plt.figure(figsize=(8, 5))
    rating_counts = df[rating_col].value_counts().sort_index()
    palette = sns.color_palette("viridis", len(rating_counts))
    ax = sns.barplot(
        x=[str(x) for x in rating_counts.index],
        y=rating_counts.values,
        hue=[str(x) for x in rating_counts.index],
        palette=palette,
        legend=False,
    )
    plt.title("Amazon Review Star Rating Distribution", fontsize=14, pad=15)
    plt.xlabel("Review Rating (Stars)", fontsize=12)
    plt.ylabel("Sample Count", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    # Annotate bar heights
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f"{int(height):,}",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=10,
            xytext=(0, 3),
            textcoords="offset points",
        )

    plt.tight_layout()
    plt.savefig(rating_path, dpi=300)
    plt.close()
    logger.info(f"Rating distribution plot saved to '{rating_path}'.")

    # 2. Class Balance Chart
    balance_path = output_dir / "class_balance.png"
    plt.figure(figsize=(7, 5))
    class_counts = df[sentiment_col].value_counts()
    colors = {"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral": "#f39c12"}
    bar_colors = [colors.get(c, "#3498db") for c in class_counts.index]
    
    ax = sns.barplot(
        x=class_counts.index.tolist(),
        y=class_counts.values,
        hue=class_counts.index.tolist(),
        palette=bar_colors,
        legend=False,
    )
    plt.title("Target Sentiment Class Balance", fontsize=14, pad=15)
    plt.xlabel("Sentiment Class", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    total_samples = len(df)
    for p in ax.patches:
        height = p.get_height()
        pct = (height / total_samples) * 100
        ax.annotate(
            f"{int(height):,} ({pct:.1f}%)",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=10,
            xytext=(0, 3),
            textcoords="offset points",
        )

    plt.tight_layout()
    plt.savefig(balance_path, dpi=300)
    plt.close()
    logger.info(f"Class balance plot saved to '{balance_path}'.")

    return rating_path, balance_path


def generate_wordclouds(
    df: pd.DataFrame,
    text_col: str,
    sentiment_col: str,
    output_dir: str | Path,
) -> Path:
    """
    Generate and save high-resolution WordClouds for Positive and Negative reviews.

    Args:
        df: Ingested and processed DataFrame.
        text_col: Column containing review texts.
        sentiment_col: Column indicating sentiment class.
        output_dir: Target destination directory.

    Returns:
        Path to the saved comparative WordCloud visual.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "wordcloud_sentiment.png"

    pos_text = " ".join(df[df[sentiment_col] == "Positive"][text_col].dropna().astype(str).tolist()[:5000])
    neg_text = " ".join(df[df[sentiment_col] == "Negative"][text_col].dropna().astype(str).tolist()[:5000])

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Positive WordCloud
    if pos_text.strip():
        wc_pos = WordCloud(
            width=800,
            height=600,
            background_color="white",
            colormap="Greens",
            max_words=150,
            collocations=False,
        ).generate(pos_text)
        axes[0].imshow(wc_pos, interpolation="bilinear")
    axes[0].set_title("Positive Reviews - Prominent Word Cloud", fontsize=16, pad=10)
    axes[0].axis("off")

    # Negative WordCloud
    if neg_text.strip():
        wc_neg = WordCloud(
            width=800,
            height=600,
            background_color="white",
            colormap="Reds",
            max_words=150,
            collocations=False,
        ).generate(neg_text)
        axes[1].imshow(wc_neg, interpolation="bilinear")
    axes[1].set_title("Negative Reviews - Prominent Word Cloud", fontsize=16, pad=10)
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    logger.info(f"Sentiment WordCloud saved to '{out_file}'.")
    return out_file
