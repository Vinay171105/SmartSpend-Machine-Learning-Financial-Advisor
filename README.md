# SmartSpend — Machine Learning Financial Advisor

SmartSpend is a Streamlit application that estimates monthly savings from income and five spending categories: food, transportation, entertainment, utilities, and shopping. It is an educational project, not financial advice.

## Overview

Enter a monthly income and expenses to compare a Linear Regression estimate with the disposable savings calculated directly from those inputs. The app also gives a practical, achievable savings-goal suggestion.

## Features

- Streamlit dashboard with validated income and expense inputs
- Disposable-savings calculation: `income - tracked expenses`
- Linear Regression savings estimate
- Holdout MAE and R² reporting
- Savings-goal and next-month expense-trend utilities
- Dataset schema, missing-value, numeric-value, and accounting validation
- Clear educational-estimate disclaimer

## Architecture

```text
expenses.csv
    ↓
smartspend_core.py (load, validate, train, evaluate, predict)
    ├── app.py (Streamlit entry point)
    ├── savings_predictor.py
    ├── savings_goal_predictor.py
    ├── savings_goal_recommender.py
    └── expense_trend_prediction.py
```

`smartspend_core.py` is the single source of truth for dataset paths, validation, calculations, model training, evaluation, and predictions. Each command-line script is import-safe and runs only when called directly.

## Tech stack

- Python 3.10+
- Streamlit
- pandas
- scikit-learn
- matplotlib (for the exploratory notebook)
- pytest (development/testing)

## Dataset

[`expenses.csv`](expenses.csv) contains 12 sample monthly records with these columns:

`month`, `income`, `food`, `transportation`, `entertainment`, `utilities`, `shopping`, and `savings`.

The project validates every load to ensure that `savings` equals:

```text
income - food - transportation - entertainment - utilities - shopping
```

Rows with missing/non-numeric financial values, negative incomes or expenses, missing required columns, or inconsistent savings are rejected with a clear error. The checked-in savings column has been corrected to follow this formula.

## ML approach and evaluation

The model uses `sklearn.linear_model.LinearRegression` with six inputs: income plus the five expense categories. It uses a deterministic 75/25 train/test split (`random_state=42`), fits only on the training partition, and calculates MAE and R² on the held-out rows.

With the bundled corrected dataset, the current holdout result is MAE **₹0.00** and R² **1.000** (9 training rows, 3 test rows). This is expected because the target is exactly defined by the input accounting formula. The dataset is very small, so the metrics are demonstration-only and should not be treated as evidence of real-world predictive performance.

## Testing

The pytest suite covers dataset loading, required columns, missing values, accounting validation, model training, numeric prediction output, zero/negative disposable savings, high incomes and expenses, savings goals, and expense-trend prediction.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## Installation

```bash
git clone <your-repository-url>
cd SmartSpend-Machine-Learning-Financial-Advisor
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Running locally

Start the dashboard:

```bash
streamlit run app.py
```

Optional command-line utilities:

```bash
python savings_predictor.py
python savings_goal_predictor.py
python expense_trend_prediction.py
python savings_goal_recommender.py
```

The recommender writes `savings_goal_recommendations.csv`, which is intentionally ignored by Git.

## Deployment to Streamlit Community Cloud

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create an app from that repository.
3. Select `app.py` as the main file.
4. Deploy. Community Cloud installs the three runtime dependencies in `requirements.txt`.

The app uses paths resolved relative to its source file, has no absolute local paths, and does not require secrets or configuration files.

## Limitations

- The bundled dataset has only 12 synthetic-style monthly observations.
- Savings is an accounting identity in this dataset, so the model primarily demonstrates the pipeline rather than discovering a real behavioural relationship.
- Inputs cover only five expense categories; taxes, debt, investments, emergencies, and regional costs are excluded.
- The trend estimate uses a simple linear time index and is not a robust forecast.

## Future improvements

- Collect a larger, representative dataset over multiple users and time periods.
- Add optional categories, recurring bills, and user-uploaded data.
- Compare models with cross-validation and a meaningful independent target.
- Add charts, budget categories, and goal-progress tracking.
- Add privacy controls and user authentication before handling personal financial data.
