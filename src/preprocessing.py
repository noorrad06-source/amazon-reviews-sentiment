"""
Text Cleaning, Normalization, and Sentiment Labeling Module.

Provides regular expression text purification, stop-word elimination,
and configurable rating-to-sentiment category mapping (binary vs. 3-class).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger("preprocessing")

# Fallback English stop words if NLTK corpus cannot be fetched
DEFAULT_STOPWORDS: set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}


def load_nltk_stopwords(language: str = "english") -> set[str]:
    """Retrieve NLTK stopwords with an automatic graceful fallback."""
    try:
        import nltk
        try:
            from nltk.corpus import stopwords
            words = set(stopwords.words(language))
            logger.debug(f"Loaded {len(words)} {language} stopwords from NLTK.")
            return words
        except LookupError:
            nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords
            words = set(stopwords.words(language))
            return words
    except Exception as exc:
        logger.warning(
            f"Could not initialize NLTK stopwords ({exc}). Using built-in fallback word list."
        )
        return DEFAULT_STOPWORDS


class SentimentLabeler:
    """
    Transforms continuous or discrete ratings (1-5) into sentiment classes.
    Supports 'binary' (0: Negative, 1: Positive) and '3-class' (0: Negative, 1: Neutral, 2: Positive).
    """

    LABEL_NAMES_3CLASS = {0: "Negative", 1: "Neutral", 2: "Positive"}
    LABEL_NAMES_BINARY = {0: "Negative", 1: "Positive"}

    @staticmethod
    def map_rating(rating: float | int, mode: str = "binary") -> int | None:
        """
        Map a single star rating to an integer sentiment class.

        Args:
            rating: Star rating value (typically 1.0 to 5.0).
            mode: Classification paradigm ('binary' or '3-class').

        Returns:
            Integer class label or None if rating is omitted in binary mode (e.g. 3 stars).
        """
        if pd.isna(rating):
            return None

        rating_val = float(rating)

        if mode == "binary":
            if rating_val <= 2.5:
                return 0  # Negative
            elif rating_val >= 3.5:
                return 1  # Positive
            else:
                return None  # Neutral rating excluded in binary polarity benchmark

        elif mode in ["3-class", "ternary"]:
            if rating_val <= 2.5:
                return 0  # Negative
            elif 2.5 < rating_val < 3.5:
                return 1  # Neutral
            else:
                return 2  # Positive

        else:
            raise ValueError(f"Unsupported classification mode: '{mode}'. Choose 'binary' or '3-class'.")

    @classmethod
    def apply_labeling(
        cls,
        df: pd.DataFrame,
        rating_column: str = "overall",
        target_column: str = "label",
        mode: str = "binary",
    ) -> pd.DataFrame:
        """
        Apply sentiment mapping across a DataFrame, filtering unmapped rows.

        Args:
            df: Input DataFrame.
            rating_column: Name of numerical rating column.
            target_column: Output column for encoded sentiment class.
            mode: 'binary' or '3-class'.

        Returns:
            New DataFrame with mapped target_column and filtered rows.
        """
        logger.info(f"Applying sentiment labeling using '{mode}' mode on column '{rating_column}'...")
        processed_df = df.copy()
        
        # Apply mapping
        processed_df[target_column] = processed_df[rating_column].apply(
            lambda r: cls.map_rating(r, mode=mode)
        )
        
        # Drop excluded neutral ratings in binary mode or invalid values
        valid_df = processed_df.dropna(subset=[target_column]).copy()
        valid_df[target_column] = valid_df[target_column].astype(int)

        class_names = cls.LABEL_NAMES_BINARY if mode == "binary" else cls.LABEL_NAMES_3CLASS
        valid_df["sentiment"] = valid_df[target_column].map(class_names)

        logger.info(
            f"Sentiment labeling complete. Retained {len(valid_df)} / {len(df)} records. "
            f"Class balance: {valid_df['sentiment'].value_counts().to_dict()}"
        )
        return valid_df


class TextCleaner(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer for text cleaning, regex normalization,
    and stop-word removal.
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_numbers: bool = True,
        remove_stopwords: bool = True,
        language: str = "english",
    ) -> None:
        """
        Initialize the text cleaner.

        Args:
            lowercase: Convert all tokens to lower case.
            remove_punctuation: Strip symbols and non-alphanumeric punctuation.
            remove_numbers: Strip digits.
            remove_stopwords: Remove standard English stop words.
            language: Language corpus for stopwords.
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.language = language

        # Compiled regex patterns for high performance
        self._html_regex = re.compile(r"<[^>]+>")
        self._url_regex = re.compile(r"https?://\S+|www\.\S+")
        self._char_filter = re.compile(r"[^a-zA-Z\s]") if remove_numbers else re.compile(r"[^a-zA-Z0-9\s]")
        self._whitespace_regex = re.compile(r"\s+")

        self.stopwords = load_nltk_stopwords(language) if remove_stopwords else set()

    def clean_text(self, text: Any) -> str:
        """
        Execute regex purification pipeline on a single text string.

        Args:
            text: Raw input string.

        Returns:
            Sanitized, lowercased, and filtered text.
        """
        if text is None or pd.isna(text):
            return ""

        s = str(text)

        # 1. Strip URLs and HTML tags
        s = self._url_regex.sub(" ", s)
        s = self._html_regex.sub(" ", s)

        # 2. Case normalization
        if self.lowercase:
            s = s.lower()

        # 3. Strip special characters, symbols, and optional numbers
        if self.remove_punctuation:
            s = self._char_filter.sub(" ", s)

        # 4. Tokenization, stop-words removal, and whitespace normalization
        tokens = s.split()
        if self.remove_stopwords and self.stopwords:
            tokens = [t for t in tokens if t not in self.stopwords and len(t) > 1]

        return " ".join(tokens)

    def fit(self, X: Any, y: Any = None) -> "TextCleaner":
        """Fit transformer (stateless)."""
        return self

    def transform(self, X: Union[pd.Series, List[str], np.ndarray]) -> List[str]:
        """
        Transform an array or Series of text documents.

        Args:
            X: Iterable of raw strings.

        Returns:
            List of cleaned strings.
        """
        if isinstance(X, pd.Series):
            return [self.clean_text(val) for val in X]
        return [self.clean_text(val) for val in X]
