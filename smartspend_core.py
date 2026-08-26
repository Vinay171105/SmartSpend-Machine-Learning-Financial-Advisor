"""Reusable data, modelling, and recommendation helpers for SmartSpend.

The bundled sample contains only twelve monthly observations.  The model and
its evaluation metrics are educational demonstrations, not financial advice.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS = (
    "income",
    "food",
    "transportation",
    "entertainment",
    "utilities",
    "shopping",
)
REQUIRED_COLUMNS = ("month", *FEATURE_COLUMNS, "savings")
EXPENSE_COLUMNS = FEATURE_COLUMNS[1:]
DEFAULT_DATASET_PATH = Path(__file__).resolve().parent / "expenses.csv"


class DataValidationError(ValueError):
    """Raised when the expenses dataset or a user input is invalid."""


@dataclass(frozen=True)
class TrainingResult:
    """A fitted savings model and holdout-set evaluation metrics."""

    model: LinearRegression
    mae: float
    r2: float
    train_rows: int
    test_rows: int


def calculate_disposable_savings(
    income: float,
    food: float,
    transportation: float,
    entertainment: float,
    utilities: float,
    shopping: float,
) -> float:
    """Return income left after the five tracked monthly expense categories."""
    values = {
        "income": income,
        "food": food,
        "transportation": transportation,
        "entertainment": entertainment,
        "utilities": utilities,
        "shopping": shopping,
    }
    validated = _validate_financial_inputs(values)
    return validated["income"] - sum(validated[column] for column in EXPENSE_COLUMNS)


def load_expenses(path: str | Path | None = None) -> pd.DataFrame:
    """Load and validate the SmartSpend CSV using a path relative to this file."""
    dataset_path = Path(path) if path is not None else DEFAULT_DATASET_PATH
    try:
        data = pd.read_csv(dataset_path)
    except FileNotFoundError as error:
        raise DataValidationError(f"Dataset not found: {dataset_path}") from error
    except pd.errors.ParserError as error:
        raise DataValidationError(f"Dataset could not be parsed: {dataset_path}") from error
    return validate_expenses(data)


def validate_expenses(data: pd.DataFrame) -> pd.DataFrame:
    """Validate schema, numeric values, and the savings accounting identity.

    Savings must equal income minus all tracked expenses.  This deliberate
    check prevents a stale or manually edited savings column from being used
    for model training without an explicit correction.
    """
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise DataValidationError(f"Dataset is missing required columns: {', '.join(missing_columns)}")
    if data.empty:
        raise DataValidationError("Dataset must contain at least one row.")

    validated = data.loc[:, list(REQUIRED_COLUMNS)].copy()
    if validated["month"].isna().any() or (validated["month"].astype(str).str.strip() == "").any():
        raise DataValidationError("Column 'month' cannot contain missing or blank values.")

    for column in (*FEATURE_COLUMNS, "savings"):
        validated[column] = pd.to_numeric(validated[column], errors="coerce")
    if validated[list((*FEATURE_COLUMNS, "savings"))].isna().any().any():
        raise DataValidationError("Financial columns must contain numeric, non-missing values.")
    if (validated[list(FEATURE_COLUMNS)] < 0).any().any():
        raise DataValidationError("Income and expense values cannot be negative.")

    expected_savings = validated["income"] - validated[list(EXPENSE_COLUMNS)].sum(axis=1)
    inconsistent = ~np.isclose(validated["savings"], expected_savings, rtol=0, atol=1e-9)
    if inconsistent.any():
        affected_months = ", ".join(validated.loc[inconsistent, "month"].astype(str).tolist())
        raise DataValidationError(
            "Savings must equal income minus food, transportation, entertainment, utilities, "
            f"and shopping. Inconsistent rows: {affected_months}."
        )
    return validated


def train_savings_model(
    data: pd.DataFrame | None = None,
    *,
    test_size: float = 0.25,
    random_state: int = 42,
) -> TrainingResult:
    """Fit Linear Regression on a training split and evaluate on a holdout split."""
    validated = validate_expenses(data) if data is not None else load_expenses()
    if len(validated) < 4:
        raise DataValidationError("At least four rows are required for a train/test split.")

    features = validated.loc[:, list(FEATURE_COLUMNS)]
    target = validated["savings"]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )
    model = LinearRegression()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return TrainingResult(
        model=model,
        mae=float(mean_absolute_error(y_test, predictions)),
        r2=float(r2_score(y_test, predictions)),
        train_rows=len(x_train),
        test_rows=len(x_test),
    )


def predict_savings(model: LinearRegression, **financial_inputs: float) -> float:
    """Predict monthly savings for one validated set of inputs."""
    validated = _validate_financial_inputs(financial_inputs)
    feature_frame = pd.DataFrame([[validated[column] for column in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    return float(model.predict(feature_frame)[0])


def calculate_savings_goal(
    income: float, disposable_savings: float, target_rate: float = 0.20
) -> float:
    """Return an achievable monthly goal up to the requested income percentage."""
    if not np.isfinite(income) or income < 0:
        raise DataValidationError("Income must be a non-negative finite number.")
    if not np.isfinite(disposable_savings):
        raise DataValidationError("Disposable savings must be a finite number.")
    if not 0 <= target_rate <= 1:
        raise DataValidationError("Target rate must be between 0 and 1.")
    return float(min(income * target_rate, max(disposable_savings, 0)))


def savings_recommendation(income: float, disposable_savings: float) -> str:
    """Provide a practical, non-advisory recommendation from current inputs."""
    goal = calculate_savings_goal(income, disposable_savings)
    target = income * 0.20
    if disposable_savings <= 0:
        reduction_needed = target - disposable_savings
        return (
            "Your tracked expenses meet or exceed your income. Focus on reducing expenses by "
            f"₹{reduction_needed:,.0f} per month before setting a savings goal."
        )
    if goal < target:
        return (
            f"You can currently set aside about ₹{goal:,.0f} per month. Reducing expenses by "
            f"₹{target - goal:,.0f} would reach a 20% savings target."
        )
    return f"A monthly savings goal of ₹{goal:,.0f} (20% of income) is currently achievable."


def predict_next_month_expenses(data: pd.DataFrame | None = None) -> float:
    """Fit a simple time-index trend and estimate next month's total expenses."""
    validated = validate_expenses(data) if data is not None else load_expenses()
    if len(validated) < 2:
        raise DataValidationError("At least two rows are required for an expense trend prediction.")
    positions = np.arange(len(validated)).reshape(-1, 1)
    totals = validated.loc[:, list(EXPENSE_COLUMNS)].sum(axis=1)
    model = LinearRegression().fit(positions, totals)
    return float(model.predict([[len(validated)]])[0])


def _validate_financial_inputs(inputs: dict[str, Any]) -> dict[str, float]:
    missing = [column for column in FEATURE_COLUMNS if column not in inputs]
    if missing:
        raise DataValidationError(f"Missing financial inputs: {', '.join(missing)}")
    validated: dict[str, float] = {}
    for column in FEATURE_COLUMNS:
        try:
            value = float(inputs[column])
        except (TypeError, ValueError) as error:
            raise DataValidationError(f"{column.title()} must be numeric.") from error
        if not np.isfinite(value) or value < 0:
            raise DataValidationError(f"{column.title()} must be a non-negative finite number.")
        validated[column] = value
    return validated
