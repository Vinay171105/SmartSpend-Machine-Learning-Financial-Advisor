"""Behavioural tests for SmartSpend's data and modelling helpers."""

import math

import pytest

from smartspend_core import (
    EXPENSE_COLUMNS,
    FEATURE_COLUMNS,
    REQUIRED_COLUMNS,
    DataValidationError,
    build_budget_summary,
    build_expense_breakdown,
    calculate_disposable_savings,
    calculate_savings_goal,
    compare_to_sample,
    estimate_months_to_goal,
    load_expenses,
    predict_next_month_expenses,
    predict_savings,
    train_savings_model,
    validate_expenses,
)


@pytest.fixture(scope="module")
def expenses():
    return load_expenses()


@pytest.fixture(scope="module")
def training(expenses):
    return train_savings_model(expenses)


def test_dataset_loads_and_has_required_columns(expenses):
    assert set(REQUIRED_COLUMNS).issubset(expenses.columns)
    assert len(expenses) == 12


def test_dataset_has_no_missing_values(expenses):
    assert not expenses.isna().any().any()


def test_savings_matches_income_minus_expenses(expenses):
    expected = expenses["income"] - expenses.loc[:, list(EXPENSE_COLUMNS)].sum(axis=1)
    assert expenses["savings"].equals(expected)


def test_invalid_savings_column_is_rejected(expenses):
    invalid = expenses.copy()
    invalid.loc[0, "savings"] += 1
    with pytest.raises(DataValidationError, match="Savings must equal"):
        validate_expenses(invalid)


def test_missing_required_column_is_rejected(expenses):
    with pytest.raises(DataValidationError, match="missing required columns"):
        validate_expenses(expenses.drop(columns="shopping"))


def test_missing_financial_value_is_rejected(expenses):
    invalid = expenses.copy()
    invalid.loc[0, "food"] = None
    with pytest.raises(DataValidationError, match="non-missing"):
        validate_expenses(invalid)


def test_model_trains_and_reports_holdout_metrics(training):
    assert training.train_rows + training.test_rows == 12
    assert training.test_rows >= 2
    assert math.isfinite(training.mae)
    assert math.isfinite(training.r2)


def test_prediction_is_numeric_and_matches_known_accounting(training):
    inputs = dict(
        income=45_000,
        food=7_000,
        transportation=2_500,
        entertainment=2_000,
        utilities=3_500,
        shopping=4_000,
    )
    prediction = predict_savings(training.model, **inputs)
    assert isinstance(prediction, float)
    assert prediction == pytest.approx(calculate_disposable_savings(**inputs), abs=1e-6)


def test_zero_and_negative_disposable_savings():
    assert calculate_disposable_savings(10_000, 2_000, 2_000, 2_000, 2_000, 2_000) == 0
    assert calculate_disposable_savings(10_000, 3_000, 2_000, 2_000, 2_000, 2_000) == -1_000


def test_high_income_and_high_expenses_are_handled():
    assert calculate_disposable_savings(1_000_000, 10_000, 10_000, 10_000, 10_000, 10_000) == 950_000
    assert calculate_disposable_savings(50_000, 20_000, 10_000, 10_000, 10_000, 10_000) == -10_000


def test_budget_summary_and_expense_breakdown():
    inputs = dict(
        income=50_000,
        food=7_000,
        transportation=2_500,
        entertainment=2_000,
        utilities=3_500,
        shopping=4_000,
    )
    summary = build_budget_summary(**inputs)
    breakdown = build_expense_breakdown(**inputs)
    assert summary.total_expenses == 19_000
    assert summary.disposable_savings == 31_000
    assert summary.largest_expense_category == "food"
    assert breakdown["amount"].sum() == 19_000
    assert breakdown["share_of_expenses"].sum() == pytest.approx(100)


def test_sample_comparison_uses_dataset_averages(expenses):
    comparison = compare_to_sample(
        expenses,
        income=50_000,
        food=7_000,
        transportation=2_500,
        entertainment=2_000,
        utilities=3_500,
        shopping=4_000,
    )
    food = comparison.loc[comparison["category"] == "Food"].iloc[0]
    assert food["sample_average"] == pytest.approx(expenses["food"].mean())
    assert food["difference"] == pytest.approx(7_000 - expenses["food"].mean())


def test_savings_goal_calculation_and_goal_projection():
    assert calculate_savings_goal(50_000, 30_000) == 10_000
    assert calculate_savings_goal(50_000, 5_000) == 5_000
    assert calculate_savings_goal(50_000, -1_000) == 0
    assert estimate_months_to_goal(50_000, 10_000) == 5
    assert estimate_months_to_goal(50_000, 0) is None


def test_expense_trend_prediction_is_finite_and_positive(expenses):
    prediction = predict_next_month_expenses(expenses)
    assert isinstance(prediction, float)
    assert math.isfinite(prediction)
    assert prediction > 0
