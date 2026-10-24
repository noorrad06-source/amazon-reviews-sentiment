"""
Command-Line Interface (CLI) Inference Engine for Amazon Reviews Sentiment Analysis.

Loads the persisted production model pipeline, executes end-to-end inference
on custom input text, and provides an interactive prediction REPL.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Ensure project root is available in module search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import ModelPersistence
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("predict")


class SentimentPredictor:
    """Production inference wrapper around serialized sentiment analysis pipeline."""

    def __init__(self, model_path: str | Path = "outputs/models/best_sentiment_model.joblib") -> None:
        """
        Load serialized bundle from disk.

        Args:
            model_path: File system path to the joblib model artifact.
        """
        self.model_path = Path(model_path)
        bundle = ModelPersistence.load_model(self.model_path)
        self.pipeline = bundle["pipeline"]
        self.model_name = bundle.get("model_name", "Best Model")
        self.metadata = bundle.get("metadata", {})
        self.class_names = self.metadata.get(
            "class_names", {0: "Negative", 1: "Positive", 2: "Neutral"}
        )

        logger.info(
            f"Successfully initialized SentimentPredictor with '{self.model_name}' "
            f"(target classes: {self.class_names})."
        )

    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        """
        Execute sentiment classification on a collection of raw review strings.

        Args:
            texts: List of review text inputs.

        Returns:
            List of dictionaries containing predicted label, class name, and confidence scores.
        """
        preds = self.pipeline.predict(texts)
        results = []

        # Check if classifier exposes predict_proba or decision_function
        classifier = self.pipeline.named_steps.get("classifier")
        has_proba = hasattr(classifier, "predict_proba")
        has_decision = hasattr(classifier, "decision_function")

        probas = None
        decisions = None
        if has_proba:
            try:
                probas = self.pipeline.predict_proba(texts)
            except Exception:
                pass
        elif has_decision:
            try:
                decisions = self.pipeline.decision_function(texts)
            except Exception:
                pass

        for i, text in enumerate(texts):
            pred_id = int(preds[i])
            sentiment_name = self.class_names.get(pred_id, str(pred_id))
            
            res = {
                "text": text,
                "prediction": pred_id,
                "sentiment": sentiment_name,
            }

            if probas is not None:
                res["probabilities"] = {
                    self.class_names.get(cls_idx, str(cls_idx)): float(probas[i][cls_idx])
                    for cls_idx in range(probas.shape[1])
                }
                res["confidence"] = float(np.max(probas[i]))
            elif decisions is not None:
                score = float(decisions[i]) if decisions.ndim == 1 else [float(x) for x in decisions[i]]
                res["decision_score"] = score

            results.append(res)

        return results

    def predict_single(self, text: str) -> dict[str, Any]:
        """Convenience method for single string inference."""
        return self.predict([text])[0]


def run_interactive_session(predictor: SentimentPredictor) -> None:
    """Launch an interactive terminal REPL for real-time sentiment evaluation."""
    print("\n" + "=" * 70)
    print(" AMAZON REVIEWS SENTIMENT INFERENCE SHELL ".center(70, "="))
    print(f" Loaded Model: {predictor.model_name}".center(70))
    print(" Type your review below and press Enter. Enter 'exit' or 'quit' to stop.")
    print("=" * 70 + "\n")

    while True:
        try:
            user_input = input("Review > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting sentiment analysis session. Goodbye!")
                break

            result = predictor.predict_single(user_input)
            sentiment = result["sentiment"]
            label_color = "[+]" if sentiment == "Positive" else ("[-]" if sentiment == "Negative" else "[~]")
            
            print(f"\n{label_color} Predicted Sentiment: {sentiment.upper()} (Class {result['prediction']})")
            if "confidence" in result:
                print(f"    Confidence:          {result['confidence']:.2%}")
                print(f"    Class Probabilities: {result['probabilities']}")
            elif "decision_score" in result:
                print(f"    Decision Score:      {result['decision_score']}")
            print("-" * 70)
        except (KeyboardInterrupt, EOFError):
            print("\nSession interrupted. Exiting.")
            break


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for prediction tool."""
    parser = argparse.ArgumentParser(
        description="Predict sentiment of Amazon reviews using trained production pipeline."
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="outputs/models/best_sentiment_model.joblib",
        help="Path to serialized joblib model artifact.",
    )
    parser.add_argument(
        "--text",
        "-t",
        type=str,
        default=None,
        help="Raw review text string to classify.",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default=None,
        help="Path to text file containing one review per line.",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Launch interactive terminal REPL session.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for CLI prediction."""
    args = parse_args()
    model_path = Path(args.model)

    if not model_path.exists():
        logger.error(
            f"Model file '{model_path}' does not exist. "
            f"Please run 'python main.py' first to train and serialize the model."
        )
        sys.exit(1)

    predictor = SentimentPredictor(model_path=model_path)

    if args.interactive or (args.text is None and args.file is None):
        run_interactive_session(predictor)
        return

    if args.text:
        res = predictor.predict_single(args.text)
        print(f"\nReview:              \"{res['text']}\"")
        print(f"Predicted Sentiment: {res['sentiment']} (Class {res['prediction']})")
        if "confidence" in res:
            print(f"Confidence:          {res['confidence']:.2%}")
        elif "decision_score" in res:
            print(f"Decision Score:      {res['decision_score']:.4f}")
        return

    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            logger.error(f"Input file '{file_path}' not found.")
            sys.exit(1)

        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        logger.info(f"Loaded {len(lines)} reviews from '{file_path}'. Processing predictions...")
        results = predictor.predict(lines)
        for r in results:
            print(f"[{r['sentiment'].upper()}] (Score: {r.get('confidence', r.get('decision_score'))}) | {r['text'][:90]}...")


if __name__ == "__main__":
    main()
