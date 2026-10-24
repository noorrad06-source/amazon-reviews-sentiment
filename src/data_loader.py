"""
Data Ingestion and Archive Extraction Module.

Handles automated zip archive extraction, schema inspection, fastText/CSV parsing,
and fallback generation for Amazon Reviews Sentiment Analysis.
"""

from __future__ import annotations

import bz2
import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys
import zipfile

# Ensure project root is available in module search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.make_sample_data import generate_sample_data

logger = logging.getLogger("data_loader")

# Candidate column names across various Amazon review public datasets
REVIEW_TEXT_CANDIDATES = [
    "reviewtext", "review_text", "review", "text", "review_body",
    "body", "comment", "content", "summary_and_text"
]
RATING_CANDIDATES = [
    "overall", "rating", "ratings", "score", "stars", "star_rating", "label"
]


class DataLoader:
    """
    Manages dataset acquisition, archive extraction, column inspection,
    and normalized schema loading.
    """

    def __init__(
        self,
        zip_path: Optional[str | Path] = None,
        extracted_dir: str | Path = "data/raw",
        fallback_csv_path: str | Path = "data/sample_reviews.csv",
        sample_size: Optional[int] = 30000,
    ) -> None:
        """
        Initialize the DataLoader.

        Args:
            zip_path: Path to the local archive file (.zip).
            extracted_dir: Destination folder to unpack archive entries.
            fallback_csv_path: Path to synthetic/fallback CSV when zip is absent.
            sample_size: Maximum rows to ingest (None to ingest all rows).
        """
        self.zip_path = Path(zip_path) if zip_path else None
        self.extracted_dir = Path(extracted_dir)
        self.fallback_csv_path = Path(fallback_csv_path)
        self.sample_size = sample_size
        self.extracted_dir.mkdir(parents=True, exist_ok=True)

    def extract_archive(self) -> list[Path]:
        """
        Detect and extract the configured ZIP file into the destination directory.

        Returns:
            List of Path objects for all extracted files.
        """
        if not self.zip_path or not self.zip_path.exists():
            logger.warning(
                f"Configured zip path '{self.zip_path}' not found on local filesystem."
            )
            return []

        logger.info(f"Extracting zip archive '{self.zip_path}' to '{self.extracted_dir}'...")
        extracted_files: list[Path] = []
        with zipfile.ZipFile(self.zip_path, "r") as zf:
            for member in zf.infolist():
                # Avoid path traversal security risks
                target_path = self.extracted_dir / member.filename
                if not target_path.exists() or target_path.stat().st_size != member.file_size:
                    logger.info(f"Unpacking entry: {member.filename} ({member.file_size / (1024*1024):.2f} MB)")
                    zf.extract(member, self.extracted_dir)
                else:
                    logger.info(f"Existing file up-to-date: {member.filename}")
                extracted_files.append(target_path)

        logger.info(f"Archive successfully extracted ({len(extracted_files)} entries).")
        return extracted_files

    def inspect_columns(self, df: pd.DataFrame) -> tuple[str, str]:
        """
        Inspect dataset columns and identify review text and rating columns.

        Args:
            df: Ingested pandas DataFrame.

        Returns:
            Tuple of (text_column_name, rating_column_name).

        Raises:
            ValueError: If neither a text nor rating column could be identified.
        """
        lower_cols = {col.lower().strip(): col for col in df.columns}
        
        text_col: str | None = None
        rating_col: str | None = None

        for candidate in REVIEW_TEXT_CANDIDATES:
            if candidate in lower_cols:
                text_col = lower_cols[candidate]
                break

        for candidate in RATING_CANDIDATES:
            if candidate in lower_cols:
                rating_col = lower_cols[candidate]
                break

        if text_col is None or rating_col is None:
            available = list(df.columns)
            raise ValueError(
                f"Failed to identify required columns in dataset. "
                f"Available columns: {available}. "
                f"Identified text={text_col}, rating={rating_col}."
            )

        logger.info(
            f"Dataset column inspection successful: "
            f"Review Text Column='{text_col}', Rating Column='{rating_col}'"
        )
        return text_col, rating_col

    def _load_fasttext_stream(self, file_path: Path) -> pd.DataFrame:
        """
        Parse fastText-formatted Amazon reviews (__label__1 / __label__2).
        Supports raw text and .bz2 compression.
        """
        logger.info(f"Parsing FastText Amazon review file: '{file_path}' (sample_size={self.sample_size})...")
        rows: list[dict[str, Any]] = []

        # Open stream depending on file compression
        if file_path.suffix == ".bz2":
            stream = bz2.open(file_path, "rt", encoding="utf-8", errors="ignore")
        else:
            stream = open(file_path, "rt", encoding="utf-8", errors="ignore")

        try:
            for idx, line in enumerate(stream):
                if self.sample_size is not None and idx >= self.sample_size:
                    break
                line = line.strip()
                if not line:
                    continue

                # FastText format: __label__1 <review text> or __label__2 <review text>
                if line.startswith("__label__1"):
                    # __label__1 corresponds to 1 or 2 stars (Negative)
                    review = line[len("__label__1"):].strip()
                    rows.append({"reviewText": review, "overall": 1.0})
                elif line.startswith("__label__2"):
                    # __label__2 corresponds to 4 or 5 stars (Positive)
                    review = line[len("__label__2"):].strip()
                    rows.append({"reviewText": review, "overall": 5.0})
                else:
                    parts = line.split(" ", 1)
                    if len(parts) == 2 and parts[0].startswith("__label__"):
                        label_num = 1.0 if "1" in parts[0] else 5.0
                        rows.append({"reviewText": parts[1].strip(), "overall": label_num})
        finally:
            stream.close()

        df = pd.DataFrame(rows)
        logger.info(f"Parsed {len(df)} records from FastText source '{file_path.name}'.")
        return df

    def _load_tabular_file(self, file_path: Path) -> pd.DataFrame:
        """Load CSV, TSV, or JSON records into a DataFrame."""
        logger.info(f"Loading tabular dataset: '{file_path}'")
        suffix = file_path.suffix.lower()
        if suffix in [".csv", ".txt"]:
            df = pd.read_csv(file_path, nrows=self.sample_size)
        elif suffix in [".tsv"]:
            df = pd.read_csv(file_path, sep="\t", nrows=self.sample_size)
        elif suffix in [".json", ".jsonl"]:
            df = pd.read_json(file_path, lines=True, nrows=self.sample_size)
        elif suffix == ".parquet":
            df = pd.read_parquet(file_path)
            if self.sample_size is not None:
                df = df.head(self.sample_size)
        else:
            raise ValueError(f"Unsupported tabular file format: {suffix}")

        text_col, rating_col = self.inspect_columns(df)
        df = df[[text_col, rating_col]].rename(
            columns={text_col: "reviewText", rating_col: "overall"}
        )
        return df

    def load_data(self) -> pd.DataFrame:
        """
        Load dataset from the extracted ZIP contents, directly from zip,
        or fall back to synthetic data generation.

        Returns:
            Normalized pandas DataFrame with ['reviewText', 'overall'] columns.
        """
        # Step 1: Attempt extraction if zip path exists
        extracted_files = []
        if self.zip_path and self.zip_path.exists():
            extracted_files = self.extract_archive()
        else:
            logger.warning(
                f"Zip file not found at '{self.zip_path}'. Checking extracted dir '{self.extracted_dir}'..."
            )

        # Step 2: Search for data files in extracted_dir
        candidates: list[Path] = []
        if self.extracted_dir.exists():
            candidates = (
                list(self.extracted_dir.glob("*.bz2"))
                + list(self.extracted_dir.glob("*.csv"))
                + list(self.extracted_dir.glob("*.txt"))
                + list(self.extracted_dir.glob("*.json"))
                + list(self.extracted_dir.glob("*.jsonl"))
            )

        # Prioritize training data if train/test split exists in raw data
        train_candidates = [p for p in candidates if "train" in p.name.lower()]
        target_file = train_candidates[0] if train_candidates else (candidates[0] if candidates else None)

        if target_file and target_file.exists():
            logger.info(f"Selected dataset file: '{target_file}'")
            if ".ft.txt" in target_file.name or target_file.suffix == ".bz2":
                df = self._load_fasttext_stream(target_file)
            else:
                df = self._load_tabular_file(target_file)
        else:
            # Fallback path: generate synthetic dataset
            logger.warning(
                f"No suitable review dataset found in extracted directory. "
                f"Generating fallback synthetic data at '{self.fallback_csv_path}'..."
            )
            df = generate_sample_data(
                output_path=self.fallback_csv_path,
                n_samples=self.sample_size if self.sample_size and self.sample_size <= 5000 else 2500,
            )

        # Data cleanliness checks
        initial_len = len(df)
        df = df.dropna(subset=["reviewText", "overall"]).copy()
        df["reviewText"] = df["reviewText"].astype(str)
        df["overall"] = pd.to_numeric(df["overall"], errors="coerce")
        df = df.dropna(subset=["overall"]).copy()

        logger.info(
            f"Loaded dataset: {len(df)} records "
            f"(dropped {initial_len - len(df)} rows with missing values)."
        )
        return df
