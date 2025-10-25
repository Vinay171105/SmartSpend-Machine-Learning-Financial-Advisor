import pandas as pd
import numpy as np

# Step 1: Load dataset
data = pd.read_csv("expenses.csv")  # Use your existing SmartSpend dataset
print("Data loaded successfully!\n")
print(data.head())

# Step 2: Calculate monthly total expenses
data["Total_Expenses"] = data[["food", "transportation", "entertainment", "utilities", "shopping"]].sum(axis=1)

# Step 3: Calculate actual savings
data["Actual_Savings"] = data["income"] - data["Total_Expenses"]

# Step 4: Recommend savings goal based on spending habits
avg_saving_ratio = data["Actual_Savings"].mean() / data["income"].mean()
recommended_goal = avg_saving_ratio * 1.1  # Slightly higher goal (10% improvement)

data["Recommended_Savings_Goal"] = data["income"] * recommended_goal

# Step 5: Display recommendations
print("\nRecommended Monthly Savings Goals:")
print(data[["month", "income", "Actual_Savings", "Recommended_Savings_Goal"]])

# Step 6: Save updated data
data.to_csv("savings_goal_recommendations.csv", index=False)
print("\nRecommendations saved to 'savings_goal_recommendations.csv'")
