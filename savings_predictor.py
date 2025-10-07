# savings_predictor.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv('expenses.csv')

# Define features (X) and target (y)
X = data[['income', 'food', 'transportation', 'entertainment', 'utilities', 'shopping']]
y = data['savings']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error: ₹{mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Visualize actual vs predicted
plt.figure(figsize=(7,5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Savings')
plt.ylabel('Predicted Savings')
plt.title('Actual vs Predicted Monthly Savings')
plt.show()
