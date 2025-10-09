# Savings Goal Predictor
# This script predicts how many months are needed to reach a savings goal based on average monthly expenses
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# ---- Step 1: Load dataset ----
data = pd.read_csv('monthly_expenses.csv') 
data['Savings'] = data['Income'] - (
    data['Food'] + data['Transportation'] + data['Entertainment'] +
    data['Utilities'] + data['Shopping']
)

# ---- Step 2: Train regression model to predict savings per month ----
X = data[['Income', 'Food', 'Transportation', 'Entertainment', 'Utilities', 'Shopping']]
y = data['Savings']

model = LinearRegression()
model.fit(X, y)

# ---- Step 3: Predict how many months are needed to reach a savings goal ----
def predict_months_to_goal(savings_goal, avg_income, avg_food, avg_transport, avg_ent, avg_util, avg_shop):
    """Predict how many months are needed to reach the given savings goal."""
    monthly_savings = model.predict([[avg_income, avg_food, avg_transport, avg_ent, avg_util, avg_shop]])[0]
    if monthly_savings <= 0:
        return "Your expenses exceed your income. Adjust spending to save."
    months_needed = savings_goal / monthly_savings
    return f"To reach ₹{savings_goal}, it will take approximately {months_needed:.1f} months."

# ---- Step 4: Example use ----
if __name__ == "__main__":
    result = predict_months_to_goal(
        savings_goal=50000,
        avg_income=45000,
        avg_food=7000,
        avg_transport=2500,
        avg_ent=2000,
        avg_util=3500,
        avg_shop=4000
    )
    print(result)
