import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Step 1: Load your dataset
data = pd.read_csv("smartspend_data.csv")  # replace with your actual dataset name
print("✅ Data loaded successfully!\n")
print(data.head())

# Step 2: Create a new column for Total Expenses
data["Total_Expenses"] = data[["food", "transportation", "entertainment", "utilities", "shopping"]].sum(axis=1)

# Step 3: Visualize monthly trends
plt.figure(figsize=(10, 6))
plt.plot(data["month"], data["Total_Expenses"], marker="o", color="teal")
plt.title("📈 Monthly Total Expense Trend")
plt.xlabel("Month")
plt.ylabel("Total Expenses (INR)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# Step 4: Predict next month's spending
months = np.arange(len(data)).reshape(-1, 1)
total = data["Total_Expenses"].values

model = LinearRegression()
model.fit(months, total)
next_month = np.array([[len(data)]])
predicted_expense = model.predict(next_month)[0]

print(f"💰 Predicted total expense for next month: ₹{predicted_expense:.2f}")

# Step 5: Visualize prediction
plt.figure(figsize=(8, 5))
plt.scatter(months, total, color="blue", label="Actual")
plt.plot(months, model.predict(months), color="green", label="Trend Line")
plt.scatter(next_month, predicted_expense, color="red", label="Predicted")
plt.title("Expense Prediction")
plt.xlabel("Month")
plt.ylabel("Total Expenses")
plt.legend()
plt.tight_layout()
plt.show()