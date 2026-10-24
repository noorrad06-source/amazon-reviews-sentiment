"""
End-to-End Pipeline Orchestrator for Amazon Reviews Sentiment Analysis.

Coordinates data loading, sentiment labeling, stratified splitting,
multi-model pipeline training, benchmark evaluation, visualization export,
and winning model persistence.
"""

from __future__ import annotations

import argparse
import datetime
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List

import yaml

from src.data_loader import DataLoader
from src.evaluate import (
    compute_metrics,
    display_benchmark_table,
    generate_wordclouds,
    plot_confusion_matrix,
    plot_distributions,
)
from src.models import (
    ModelPersistence,
    build_candidate_pipelines,
    perform_train_test_split,
)
from src.preprocessing import SentimentLabeler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("main")


def load_config(config_path: str | Path = "config/config.yaml") -> dict[str, Any]:
    """Load and validate YAML configuration settings."""
    config_path = Path(config_path)
    if not config_path.exists():
        logger.error(f"Configuration file not found at '{config_path}'.")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded configuration from '{config_path}'.")
    return config


def run_pipeline(
    config_path: str | Path = "config/config.yaml",
    mode_override: str | None = None,
    sample_size_override: int | None = None,
) -> None:
    """
    Execute complete sentiment analysis lifecycle.

    Args:
        config_path: Path to YAML configuration.
        mode_override: Optional CLI override for classification mode.
        sample_size_override: Optional CLI override for ingested sample size.
    """
    start_time = datetime.datetime.now()
    logger.info("=" * 80)
    logger.info("STARTING AMAZON REVIEWS SENTIMENT ANALYSIS PIPELINE")
    logger.info("=" * 80)

    # 1. Load Configurations
    cfg = load_config(config_path)
    data_cfg = cfg.get("data", {})
    split_cfg = cfg.get("split", {})
    prep_cfg = cfg.get("preprocessing", {})
    tfidf_cfg = cfg.get("tfidf", {})
    models_cfg = cfg.get("models", {})
    output_cfg = cfg.get("output", {})

    classification_mode = mode_override or data_cfg.get("classification_mode", "binary")
    sample_size = sample_size_override or data_cfg.get("sample_size", 30000)
    output_dir = Path(output_cfg.get("output_dir", "outputs"))
    model_dir = Path(output_cfg.get("model_dir", "outputs/models"))
    best_model_name = output_cfg.get("best_model_filename", "best_sentiment_model.joblib")
    metrics_filename = output_cfg.get("metrics_filename", "benchmark_summary.csv")

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # 2. Data Loading & Extraction
    loader = DataLoader(
        zip_path=data_cfg.get("zip_path"),
        extracted_dir=data_cfg.get("extracted_dir", "data/raw"),
        fallback_csv_path=data_cfg.get("fallback_csv_path", "data/sample_reviews.csv"),
        sample_size=sample_size,
    )
    raw_df = loader.load_data()

    # 3. Sentiment Labeling & Mapping
    processed_df = SentimentLabeler.apply_labeling(
        raw_df,
        rating_column="overall",
        target_column="label",
        mode=classification_mode,
    )

    class_names_map = (
        SentimentLabeler.LABEL_NAMES_BINARY
        if classification_mode == "binary"
        else SentimentLabeler.LABEL_NAMES_3CLASS
    )
    ordered_class_names = [class_names_map[i] for i in sorted(class_names_map.keys())]

    # 4. Generate Exploratory Visualizations
    logger.info("Generating exploratory distributions and WordCloud visual outputs...")
    plot_distributions(
        processed_df,
        rating_col="overall",
        sentiment_col="sentiment",
        output_dir=output_dir,
    )
    generate_wordclouds(
        processed_df,
        text_col="reviewText",
        sentiment_col="sentiment",
        output_dir=output_dir,
    )

    # 5. Stratified Train / Test Partitioning
    X_train, X_test, y_train, y_test = perform_train_test_split(
        processed_df,
        text_col="reviewText",
        target_col="label",
        test_size=split_cfg.get("test_size", 0.20),
        random_state=split_cfg.get("random_state", 42),
        stratify=split_cfg.get("stratify", True),
    )

    # 6. Candidate Pipelines Construction
    candidate_pipelines = build_candidate_pipelines(
        tfidf_config=tfidf_cfg,
        preprocessing_config=prep_cfg,
        model_hyperparams=models_cfg,
    )

    # 7. Model Training & Evaluation Benchmark
    logger.info("Initiating model training and comparative benchmarking...")
    benchmark_metrics: list[dict[str, Any]] = []
    fitted_pipelines: dict[str, Any] = {}

    for name, pipeline in candidate_pipelines.items():
        logger.info(f"Training candidate model: '{name}'...")
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        logger.info(f"Evaluating candidate model: '{name}' on held-out test set...")
        y_pred = pipeline.predict(X_test)
        metrics = compute_metrics(y_test, y_pred, model_name=name)
        benchmark_metrics.append(metrics)

    # 8. Display & Save Benchmark Results
    summary_df = display_benchmark_table(
        benchmark_metrics,
        output_dir=output_dir,
        filename=metrics_filename,
    )

    # 9. Winning Model Selection & Persistence
    winning_model_name = summary_df.iloc[0]["Model"]
    winning_f1 = summary_df.iloc[0]["F1-Score (Weighted)"]
    winning_acc = summary_df.iloc[0]["Accuracy"]
    best_pipeline = fitted_pipelines[winning_model_name]

    logger.info(
        f"Winning Model Identified: '{winning_model_name}' "
        f"with F1-Score={winning_f1:.4f} and Accuracy={winning_acc:.4f}."
    )

    # Plot Confusion Matrix for Top Performer
    winning_cm = next(
        m["confusion_matrix"]
        for m in benchmark_metrics
        if m["Model"] == winning_model_name
    )
    plot_confusion_matrix(
        winning_cm,
        class_names=ordered_class_names,
        model_name=winning_model_name,
        output_path=output_dir / "confusion_matrix_best_model.png",
    )

    # Persist Best Model Bundle
    save_path = model_dir / best_model_name
    metadata = {
        "classification_mode": classification_mode,
        "class_names": class_names_map,
        "f1_weighted": float(winning_f1),
        "accuracy": float(winning_acc),
        "trained_timestamp": datetime.datetime.now().isoformat(),
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "tfidf_params": tfidf_cfg,
    }

    ModelPersistence.save_model(
        pipeline=best_pipeline,
        model_name=winning_model_name,
        target_path=save_path,
        metadata=metadata,
    )

    duration = datetime.datetime.now() - start_time
    logger.info("=" * 80)
    logger.info(f"PIPELINE COMPLETED SUCCESSFULLY IN {duration.total_seconds():.2f}s")
    logger.info(f"Serialized Production Artifact: '{save_path.resolve()}'")
    logger.info(f"Visual Outputs Directory:      '{output_dir.resolve()}'")
    logger.info("=" * 80)


def parse_args() -> argparse.Namespace:
    """Parse CLI options for pipeline orchestrator."""
    parser = argparse.ArgumentParser(
        description="Run end-to-end Amazon Reviews Sentiment Analysis pipeline."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config/config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=["binary", "3-class"],
        default=None,
        help="Override classification mode ('binary' or '3-class').",
    )
    parser.add_argument(
        "--sample-size",
        "-s",
        type=int,
        default=None,
        help="Override maximum sample size to load from source dataset.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    run_pipeline(
        config_path=cli_args.config,
        mode_override=cli_args.mode,
        sample_size_override=cli_args.sample_size,
    )
