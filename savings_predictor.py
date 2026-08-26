"""Train and evaluate the SmartSpend savings model from the command line."""

from smartspend_core import train_savings_model


def main() -> None:
    result = train_savings_model()
    print(f"Training rows: {result.train_rows}; holdout rows: {result.test_rows}")
    print(f"Holdout MAE: ₹{result.mae:.2f}")
    print(f"Holdout R²: {result.r2:.3f}")
    print("Note: the 12-row sample makes these demonstration-only metrics.")


if __name__ == "__main__":
    main()
