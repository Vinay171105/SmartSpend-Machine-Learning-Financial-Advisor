import streamlit as st
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

st.title("💰 SmartSpend - ML Financial Advisor")

data = pd.read_csv('expenses.csv')

st.subheader("📊 Data Preview")
st.write(data.head())

income = st.number_input("Monthly Income:", 10000, 200000, 50000)
food = st.number_input("Food Expenses:", 0, 20000, 7000)
transport = st.number_input("Transport Expenses:", 0, 10000, 2500)
entertainment = st.number_input("Entertainment:", 0, 10000, 2000)
utilities = st.number_input("Utilities:", 0, 10000, 3500)
shopping = st.number_input("Shopping:", 0, 15000, 4000)

if st.button("Predict Savings"):
    X = data[['income', 'food', 'transportation', 'entertainment', 'utilities', 'shopping']]
    y = data['savings']
    model = LinearRegression().fit(X, y)
    pred = model.predict([[income, food, transport, entertainment, utilities, shopping]])
    st.success(f"💵 Estimated Savings: ₹{pred[0]:.2f}")
