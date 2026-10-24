"""
Synthetic Sample Dataset Generator for Amazon Reviews Sentiment Analysis.

Generates representative synthetic review records containing 'reviewText'
and 'overall' (1 to 5 numerical ratings) for offline development, pipeline testing,
and continuous integration.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import random
from typing import List, Tuple

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("make_sample_data")

# Templates for synthetic review generation across 1-5 star ratings
TEMPLATES: dict[int, list[str]] = {
    1: [
        "Terrible product! It broke after just two days of light use.",
        "Completely useless. Do not waste your money on this garbage.",
        "Worst purchase I have ever made on Amazon. Absolutely defective and cheap.",
        "Horrible customer service and the item arrived shattered. Total scam.",
        "Extremely disappointed. Failed right out of the box. Zero stars if possible.",
        "Awful quality. The plastic feels brittle and it stopped functioning immediately.",
        "Misleading product description. What arrived looks nothing like the picture.",
        "Defective unit. Smelled like burning electronics. Returning immediately.",
    ],
    2: [
        "Pretty poor quality overall. It barely functions as advertised.",
        "Disappointed with the durability. The buttons stick and battery life is pathetic.",
        "Not worth the price tag. Expected much better craftsmanship.",
        "Subpar performance. Worked okay for a week, then started glitching continuously.",
        "Cheap materials and frustrating setup. Would not recommend buying this.",
        "Mediocre at best, but leaning towards bad. Missing key accessories.",
        "Underwhelming experience. Stopped charging reliably after a few cycles.",
    ],
    3: [
        "It's okay. Nothing extraordinary, but it gets the basic job done.",
        "Decent for the price, but build quality is somewhat mediocre.",
        "Average product. Has some good features but several noticeable flaws.",
        "Acceptable performance. Neither impressed nor entirely disappointed.",
        "Works fine for occasional tasks, but wouldn't rely on it for heavy daily use.",
        "Fair value. Met minimum expectations, but documentation is quite confusing.",
        "Middle of the road. Works as expected, but feels a bit lightweight.",
    ],
    4: [
        "Good product overall. Meets my daily needs and performs reliably.",
        "Very satisfied with this purchase. Minor gripes about packaging, but item is solid.",
        "Great quality for the price. Fast shipping and smooth operation.",
        "Solid 4-star item. Well built, nice ergonomics, and easy to configure.",
        "Works nicely. Good value and respectable performance so far.",
        "Very pleased. Minor design flaw with the power cord, but otherwise great.",
        "Dependable and efficient. I would buy it again for a family member.",
    ],
    5: [
        "Absolutely fantastic! Exceeded all my expectations in every way.",
        "Outstanding quality and top-notch performance. Highly recommended!",
        "Best purchase of the year! Truly premium finish and works flawlessly.",
        "Incredible value. Easy to use, extremely durable, and super fast delivery.",
        "Five stars without hesitation! Excellent customer care and brilliant design.",
        "Perfect in every detail! I love this gadget, can't recommend it enough.",
        "Superb craftsmanship. Sturdy, beautiful, and performs like a charm.",
        "Phenomenal product! Exactly what I was searching for. 10/10!",
    ],
}

PRODUCT_TERMS: list[str] = [
    "wireless headphones", "coffee grinder", "bluetooth speaker",
    "mechanical keyboard", "running shoes", "smartphone case",
    "usb-c docking station", "laptop stand", "fitness tracker",
    "led desk lamp", "gaming mouse", "chef knife"
]


def generate_synthetic_record(rng: random.Random) -> tuple[str, int]:
    """
    Generate a single synthetic review text and overall star rating.

    Args:
        rng: Initialized Random instance for deterministic sampling.

    Returns:
        Tuple of (reviewText, overall_rating).
    """
    rating = rng.choices([1, 2, 3, 4, 5], weights=[0.20, 0.15, 0.15, 0.25, 0.25])[0]
    template = rng.choice(TEMPLATES[rating])
    product = rng.choice(PRODUCT_TERMS)
    
    # Introduce contextual variation
    suffixes = [
        f" Tested thoroughly with my {product}.",
        f" Definitely relevant for anyone shopping for a {product}.",
        f" Arrived in good packaging for a {product}.",
        "",
    ]
    full_text = f"{template}{rng.choice(suffixes)}"
    return full_text, rating


def generate_sample_data(
    output_path: str | Path = "data/sample_reviews.csv",
    n_samples: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create and persist a synthetic dataset matching Amazon review schema.

    Args:
        output_path: Target CSV destination.
        n_samples: Total number of synthetic samples to synthesize.
        seed: Random seed for reproducibility.

    Returns:
        Generated pandas DataFrame containing 'reviewText' and 'overall' columns.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    logger.info(f"Generating {n_samples} synthetic review records (seed={seed})...")

    records = [generate_synthetic_record(rng) for _ in range(n_samples)]
    df = pd.DataFrame(records, columns=["reviewText", "overall"])

    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Synthetic dataset saved successfully to '{output_path}' ({len(df)} rows).")
    return df


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for synthetic data generation."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic Amazon reviews dataset for testing."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/sample_reviews.csv",
        help="Destination path for the generated CSV.",
    )
    parser.add_argument(
        "--n_samples",
        "-n",
        type=int,
        default=2500,
        help="Number of synthetic review rows to generate.",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_sample_data(
        output_path=args.output,
        n_samples=args.n_samples,
        seed=args.seed,
    )
