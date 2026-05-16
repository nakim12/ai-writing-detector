"""Loading the Human-vs-AI essays dataset and producing a stratified split.

The dataset is expected to live at ``data/raw/AI_Human.csv`` (downloaded
manually from Kaggle). The loader normalizes column names so the rest of the
codebase can rely on ``text`` and ``generated`` being present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "AI_Human.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_essays(path: Path | str = DEFAULT_RAW_PATH) -> pd.DataFrame:
    """Load the raw essays CSV and return a tidy DataFrame.

    Returned columns:
        text (str): the essay text.
        generated (int): 0 = human, 1 = AI.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find dataset at {path}. Download it from Kaggle and "
            "place it at data/raw/AI_Human.csv."
        )

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "generated" not in df.columns:
        # Some Kaggle versions ship the label as a float or under a different name.
        for alt in ("label", "is_ai", "ai_generated"):
            if alt in df.columns:
                df = df.rename(columns={alt: "generated"})
                break

    if "text" not in df.columns or "generated" not in df.columns:
        raise ValueError(
            f"Expected columns 'text' and 'generated' in {path}, got {list(df.columns)}"
        )

    df = df[["text", "generated"]].dropna()
    df["generated"] = df["generated"].astype(int)
    df["text"] = df["text"].astype(str)
    return df.reset_index(drop=True)


def stratified_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Stratified train/test split that preserves the 50/50 class balance."""
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["generated"],
        test_size=test_size,
        random_state=random_state,
        stratify=df["generated"],
    )
    return X_train, X_test, y_train, y_test
