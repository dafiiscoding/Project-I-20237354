"""
data_loader.py — Tải và kiểm tra dữ liệu thô từ Kaggle Give Me Some Credit

Mục đích: Cung cấp giao diện nhất quán để đọc dữ liệu ở mọi bước pipeline,
đảm bảo tái lập kết quả (reproducibility) và kiểm định cơ bản.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ─── Hằng số ─────────────────────────────────────────────────────────────────

RAW_DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "cs-training.csv"
PROCESSED_PATH = Path(__file__).parent.parent / "data" / "processed"
SPLITS_PATH = Path(__file__).parent.parent / "data" / "splits"

TARGET_COL = "SeriousDlqin2yrs"

FEATURE_COLS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

EXPECTED_SHAPE = (150000, 11)


# ─── Hàm tải dữ liệu ─────────────────────────────────────────────────────────

def load_raw(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Tải file cs-training.csv gốc từ Kaggle.

    Parameters
    ----------
    path : Path
        Đường dẫn đến file CSV. Mặc định: data/raw/cs-training.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame với index gốc của Kaggle (bắt đầu từ 1), 11 cột.

    Notes
    -----
    index_col=0 để bỏ cột index không tên của Kaggle.
    Không thực hiện bất kỳ bước tiền xử lý nào ở bước này.
    """
    df = pd.read_csv(path, index_col=0)
    _validate_raw(df)
    return df


def load_processed(filename: str) -> pd.DataFrame:
    """
    Tải dữ liệu đã qua tiền xử lý từ data/processed/.

    Parameters
    ----------
    filename : str
        Tên file (ví dụ: 'features_engineered.csv').

    Returns
    -------
    pd.DataFrame
    """
    path = PROCESSED_PATH / filename
    if not path.exists():
        raise FileNotFoundError(f"Processed file không tồn tại: {path}")
    return pd.read_csv(path, index_col=0)


def load_split(split: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Tải tập train/val/test.

    Parameters
    ----------
    split : str
        Một trong: 'train', 'val', 'test'.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        (X, y) — đặc trưng và nhãn mục tiêu riêng biệt.
    """
    valid_splits = {"train", "val", "test"}
    if split not in valid_splits:
        raise ValueError(f"split phải là một trong {valid_splits}, nhận được: '{split}'")

    path = SPLITS_PATH / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Split file không tồn tại: {path}. Chạy notebook 02_Preprocessing.ipynb trước."
        )

    df = pd.read_csv(path, index_col=0)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return X, y


# ─── Hàm nội bộ ──────────────────────────────────────────────────────────────

def _validate_raw(df: pd.DataFrame) -> None:
    """Kiểm tra shape và columns của dữ liệu thô."""
    if df.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"Raw data có shape {df.shape}, expected {EXPECTED_SHAPE}. "
            "Kiểm tra lại file cs-training.csv."
        )
    expected_cols = set([TARGET_COL] + FEATURE_COLS)
    actual_cols = set(df.columns)
    if expected_cols != actual_cols:
        raise ValueError(f"Columns không khớp. Missing: {expected_cols - actual_cols}")


def get_data_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo bảng thống kê tóm tắt đầy đủ cho EDA.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame thô hoặc đã qua tiền xử lý.

    Returns
    -------
    pd.DataFrame
        Bảng tóm tắt gồm: count, missing%, mean, median, std, min, max,
        skewness (độ lệch), kurtosis (độ nhọn).
    """
    summary = pd.DataFrame({
        "count": df.count(),
        "missing_pct": df.isnull().mean() * 100,
        "mean": df.mean(numeric_only=True),
        "median": df.median(numeric_only=True),
        "std": df.std(numeric_only=True),
        "min": df.min(numeric_only=True),
        "max": df.max(numeric_only=True),
        "skewness": df.skew(numeric_only=True),
        "kurtosis": df.kurt(numeric_only=True),
    })
    return summary.round(4)
