"""Estimate months to a savings goal using the validated SmartSpend dataset."""

from smartspend_core import estimate_months_to_goal, predict_savings, train_savings_model


def predict_months_to_goal(
    savings_goal: float,
    income: float,
    food: float,
    transportation: float,
    entertainment: float,
    utilities: float,
    shopping: float,
) -> float | None:
    """Predict monthly savings, then estimate the number of months to a goal."""
    result = train_savings_model()
    monthly_savings = predict_savings(
        result.model,
        income=income,
        food=food,
        transportation=transportation,
        entertainment=entertainment,
        utilities=utilities,
        shopping=shopping,
    )
    return estimate_months_to_goal(savings_goal, monthly_savings)


def main() -> None:
    months = predict_months_to_goal(50_000, 45_000, 7_000, 2_500, 2_000, 3_500, 4_000)
    if months is None:
        print("Your expenses exceed your income. Adjust spending before setting this goal.")
    else:
        print(f"A ₹50,000 goal would take approximately {months:.1f} months.")


if __name__ == "__main__":
    main()
