"""Estimate next month's total expenses from the checked-in dataset."""

from smartspend_core import predict_next_month_expenses


def main() -> None:
    predicted_expense = predict_next_month_expenses()
    print(f"Predicted total expense for next month: ₹{predicted_expense:.2f}")


if __name__ == "__main__":
    main()
