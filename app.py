"""Streamlit entry point for SmartSpend."""

import streamlit as st

from smartspend_core import (
    DataValidationError,
    calculate_disposable_savings,
    load_expenses,
    predict_savings,
    savings_recommendation,
    train_savings_model,
)


@st.cache_data
def get_expenses():
    """Read the checked-in dataset once per Streamlit session cache."""
    return load_expenses()


@st.cache_resource
def get_training_result():
    """Train once per Streamlit resource cache rather than on every click."""
    return train_savings_model(get_expenses())


def main() -> None:
    st.set_page_config(page_title="SmartSpend", page_icon="💰")
    st.title("💰 SmartSpend")
    st.caption("An educational monthly savings estimator — not financial advice.")

    try:
        data = get_expenses()
        training = get_training_result()
    except (DataValidationError, OSError) as error:
        st.error(f"SmartSpend could not load its dataset: {error}")
        st.stop()

    st.subheader("Monthly spending inputs")
    income = st.number_input("Monthly income (₹)", min_value=0.0, value=50_000.0, step=500.0)
    left, right = st.columns(2)
    with left:
        food = st.number_input("Food (₹)", min_value=0.0, value=7_000.0, step=250.0)
        transportation = st.number_input("Transportation (₹)", min_value=0.0, value=2_500.0, step=250.0)
        entertainment = st.number_input("Entertainment (₹)", min_value=0.0, value=2_000.0, step=250.0)
    with right:
        utilities = st.number_input("Utilities (₹)", min_value=0.0, value=3_500.0, step=250.0)
        shopping = st.number_input("Shopping (₹)", min_value=0.0, value=4_000.0, step=250.0)

    if st.button("Estimate savings", type="primary"):
        inputs = {
            "income": income,
            "food": food,
            "transportation": transportation,
            "entertainment": entertainment,
            "utilities": utilities,
            "shopping": shopping,
        }
        try:
            predicted = predict_savings(training.model, **inputs)
            disposable = calculate_disposable_savings(**inputs)
        except DataValidationError as error:
            st.error(f"Please correct the inputs: {error}")
        else:
            prediction_column, disposable_column = st.columns(2)
            prediction_column.metric("Model-estimated savings", f"₹{predicted:,.0f}")
            disposable_column.metric("Disposable savings from your inputs", f"₹{disposable:,.0f}")
            if disposable <= 0:
                st.warning("Your tracked expenses are equal to or higher than your income.")
            else:
                st.success("Your tracked expenses leave room for savings.")
            st.info(savings_recommendation(income, disposable))

    with st.expander("Dataset and model details"):
        st.write(f"Holdout MAE: ₹{training.mae:,.2f} · Holdout R²: {training.r2:.3f}")
        st.warning(
            "This dataset has only 12 monthly records. These metrics and predictions are "
            "demonstration-only and should not be used as financial advice."
        )
        st.dataframe(data, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
