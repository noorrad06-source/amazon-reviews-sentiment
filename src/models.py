"""
Model Definition, Pipeline Architecture, and Training Engine.

Constructs scikit-learn Pipelines encapsulating TextCleaner, TfidfVectorizer,
and candidate classifiers (Logistic Regression, Naive Bayes, LinearSVC) to eliminate
data leakage and guarantee reproducible serialization.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.preprocessing import TextCleaner

logger = logging.getLogger("models")


def build_candidate_pipelines(
    tfidf_config: Optional[dict[str, Any]] = None,
    preprocessing_config: Optional[dict[str, Any]] = None,
    model_hyperparams: Optional[dict[str, Any]] = None,
) -> dict[str, Pipeline]:
    """
    Construct self-contained scikit-learn pipelines for all candidate classifiers.

    Args:
        tfidf_config: Hyperparameters for TfidfVectorizer.
        preprocessing_config: Settings for TextCleaner.
        model_hyperparams: Dictionary specifying individual classifier params.

    Returns:
        Dictionary mapping model names to their instantiated un-fitted Pipelines.
    """
    tfidf_config = tfidf_config or {}
    preprocessing_config = preprocessing_config or {}
    model_hyperparams = model_hyperparams or {}

    cleaner = TextCleaner(
        lowercase=preprocessing_config.get("lowercase", True),
        remove_punctuation=preprocessing_config.get("remove_punctuation", True),
        remove_numbers=preprocessing_config.get("remove_numbers", True),
        remove_stopwords=preprocessing_config.get("remove_stopwords", True),
        language=preprocessing_config.get("language", "english"),
    )

    tfidf_params = {
        "max_features": tfidf_config.get("max_features", 10000),
        "ngram_range": tuple(tfidf_config.get("ngram_range", [1, 2])),
        "min_df": tfidf_config.get("min_df", 2),
        "sublinear_tf": tfidf_config.get("sublinear_tf", True),
        "norm": tfidf_config.get("norm", "l2"),
    }

    # Model 1: Logistic Regression
    lr_params = model_hyperparams.get("logistic_regression", {})
    lr_clf = LogisticRegression(
        C=lr_params.get("C", 1.0),
        max_iter=lr_params.get("max_iter", 1000),
        solver=lr_params.get("solver", "lbfgs"),
        random_state=lr_params.get("random_state", 42),
    )

    # Model 2: Multinomial Naive Bayes
    nb_params = model_hyperparams.get("multinomial_nb", {})
    nb_clf = MultinomialNB(
        alpha=nb_params.get("alpha", 1.0),
        fit_prior=nb_params.get("fit_prior", True),
    )

    # Model 3: Support Vector Machine (LinearSVC)
    svc_params = model_hyperparams.get("linear_svc", {})
    svc_clf = LinearSVC(
        C=svc_params.get("C", 1.0),
        max_iter=svc_params.get("max_iter", 2000),
        random_state=svc_params.get("random_state", 42),
    )

    pipelines = {
        "Logistic Regression": Pipeline(
            [
                ("cleaner", cleaner),
                ("tfidf", TfidfVectorizer(**tfidf_params)),
                ("classifier", lr_clf),
            ]
        ),
        "Multinomial Naive Bayes": Pipeline(
            [
                ("cleaner", cleaner),
                ("tfidf", TfidfVectorizer(**tfidf_params)),
                ("classifier", nb_clf),
            ]
        ),
        "Support Vector Machine (LinearSVC)": Pipeline(
            [
                ("cleaner", cleaner),
                ("tfidf", TfidfVectorizer(**tfidf_params)),
                ("classifier", svc_clf),
            ]
        ),
    }

    return pipelines


def perform_train_test_split(
    df: pd.DataFrame,
    text_col: str = "reviewText",
    target_col: str = "label",
    test_size: float = 0.20,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Split the dataset into training and testing partitions using stratified sampling.

    Args:
        df: Input DataFrame containing features and targets.
        text_col: Name of column containing review texts.
        target_col: Name of column containing encoded sentiment classes.
        test_size: Fraction of samples to allocate to test partition.
        random_state: Reproducibility seed.
        stratify: Whether to preserve class proportions across splits.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X = df[text_col]
    y = df[target_col]

    stratify_target = y if stratify else None

    logger.info(
        f"Splitting dataset ({len(df)} rows) into train ({1 - test_size:.0%}) "
        f"and test ({test_size:.0%}) partitions (seed={random_state}, stratify={stratify})..."
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    logger.info(
        f"Split complete. Train set size: {len(X_train)} samples, "
        f"Test set size: {len(X_test)} samples."
    )
    return X_train, X_test, y_train, y_test


class ModelPersistence:
    """Handles serialization and deserialization of production model artifacts."""

    @staticmethod
    def save_model(
        pipeline: Pipeline,
        model_name: str,
        target_path: str | Path,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Path:
        """
        Serialize trained pipeline and metadata bundle via joblib.

        Args:
            pipeline: Trained scikit-learn Pipeline instance.
            model_name: Human-readable model identifier.
            target_path: File system destination.
            metadata: Associated runtime details, hyperparameters, and class labels.

        Returns:
            Resolved Path of the saved artifact.
        """
        target_path = Path(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        bundle = {
            "model_name": model_name,
            "pipeline": pipeline,
            "metadata": metadata or {},
        }

        joblib.dump(bundle, target_path)
        logger.info(f"Model artifact successfully saved to '{target_path}'.")
        return target_path

    @staticmethod
    def load_model(model_path: str | Path) -> dict[str, Any]:
        """
        Load serialized model bundle from disk.

        Args:
            model_path: Path to serialized artifact.

        Returns:
            Dictionary containing 'pipeline', 'model_name', and 'metadata'.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at '{model_path}'.")

        logger.info(f"Loading model artifact from '{model_path}'...")
        bundle: dict[str, Any] = joblib.load(model_path)
        return bundle
