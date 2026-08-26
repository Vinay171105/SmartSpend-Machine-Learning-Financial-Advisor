"""Create achievable 20%-of-income savings-goal recommendations."""

from pathlib import Path

import pandas as pd

from smartspend_core import EXPENSE_COLUMNS, calculate_savings_goal, load_expenses, validate_expenses


def build_recommendations(data: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return each month with total expenses, disposable savings, and a feasible goal."""
    expenses = load_expenses() if data is None else validate_expenses(data)
    expenses["total_expenses"] = expenses.loc[:, list(EXPENSE_COLUMNS)].sum(axis=1)
    expenses["actual_savings"] = expenses["income"] - expenses["total_expenses"]
    expenses["recommended_savings_goal"] = expenses.apply(
        lambda row: calculate_savings_goal(row["income"], row["actual_savings"]), axis=1
    )
    return expenses


def main() -> None:
    recommendations = build_recommendations()
    output_path = Path(__file__).resolve().parent / "savings_goal_recommendations.csv"
    recommendations.to_csv(output_path, index=False)
    print(recommendations[["month", "income", "actual_savings", "recommended_savings_goal"]].to_string(index=False))
    print(f"Recommendations saved to {output_path.name}")


if __name__ == "__main__":
    main()
