"""Streamlit entry point for the SmartSpend financial-learning dashboard."""

from __future__ import annotations

import streamlit as st

from smartspend_core import (
    DataValidationError,
    build_budget_summary,
    build_expense_breakdown,
    calculate_savings_goal,
    compare_to_sample,
    estimate_months_to_goal,
    load_expenses,
    predict_next_month_expenses,
    predict_savings,
    savings_recommendation,
    train_savings_model,
)


DEFAULT_INPUTS = {
    "income": 50_000.0,
    "food": 7_000.0,
    "transportation": 2_500.0,
    "entertainment": 2_000.0,
    "utilities": 3_500.0,
    "shopping": 4_000.0,
}


@st.cache_data
def get_expenses():
    """Read the checked-in dataset once per Streamlit session cache."""
    return load_expenses()


@st.cache_resource
def get_training_result():
    """Train once per Streamlit resource cache rather than on every interaction."""
    return train_savings_model(get_expenses())


def format_currency(value: float) -> str:
    """Format an INR amount consistently across the dashboard."""
    return f"₹{value:,.0f}"


def initialise_inputs() -> None:
    """Populate widget state with the sample plan once per browser session."""
    for key, value in DEFAULT_INPUTS.items():
        st.session_state.setdefault(f"input_{key}", value)


def render_sidebar() -> dict[str, float]:
    """Render persistent budget inputs and return their current values."""
    initialise_inputs()
    with st.sidebar:
        st.markdown("## Your monthly plan")
        st.caption("Adjust a category to update every dashboard view.")
        if st.button("Reset to sample plan", width="stretch"):
            for key, value in DEFAULT_INPUTS.items():
                st.session_state[f"input_{key}"] = value
            st.rerun()

        st.markdown("#### Income")
        income = st.number_input(
            "Monthly income (₹)", min_value=0.0, step=500.0, key="input_income"
        )
        st.markdown("#### Expenses")
        food = st.number_input("Food (₹)", min_value=0.0, step=250.0, key="input_food")
        transportation = st.number_input(
            "Transportation (₹)", min_value=0.0, step=250.0, key="input_transportation"
        )
        entertainment = st.number_input(
            "Entertainment (₹)", min_value=0.0, step=250.0, key="input_entertainment"
        )
        utilities = st.number_input("Utilities (₹)", min_value=0.0, step=250.0, key="input_utilities")
        shopping = st.number_input("Shopping (₹)", min_value=0.0, step=250.0, key="input_shopping")
        st.divider()
        st.caption("All figures are monthly and in Indian rupees (₹).")
    return {
        "income": income,
        "food": food,
        "transportation": transportation,
        "entertainment": entertainment,
        "utilities": utilities,
        "shopping": shopping,
    }


def inject_styles() -> None:
    """Apply a small visual system without adding a frontend dependency."""
    st.markdown(
        """
        <style>
            [data-testid="stSidebar"] { background: #0b1f33; border-right: 1px solid #24425e; }
            [data-testid="stSidebar"] * { color: #e7f0fa; }
            [data-testid="stSidebar"] input {
                color: #f8fbff !important; background: #152e48 !important; border-color: #3b5f82 !important;
            }
            [data-testid="stSidebar"] button {
                color: #e7f0fa !important; background: #1c5870 !important; border: 1px solid #3cd2c8 !important;
            }
            [data-testid="stSidebar"] button:hover { background: #26738a !important; }
            .block-container { max-width: 1250px; padding-top: 2.2rem; padding-bottom: 3rem; }
            .hero { background: linear-gradient(120deg, #102a43, #147d92); border: 1px solid #2d6d7d; border-radius: 18px;
                    color: #ffffff; padding: 2rem 2.1rem; margin-bottom: 1.3rem; }
            .hero h1 { color: #ffffff; margin: 0 0 .35rem; font-size: 2.25rem; }
            .hero p { color: #d9edf2; margin: 0; font-size: 1.05rem; }
            [data-testid="stMetric"] { background: #12253a; border: 1px solid #2d4965;
                                       border-radius: 14px; padding: .85rem; }
            [data-testid="stMetricLabel"] { color: #b7c9dc !important; }
            [data-testid="stMetricValue"] { color: #f8fbff !important; }
            [data-testid="stTabs"] button { color: #b7c9dc !important; font-weight: 600; }
            [data-testid="stTabs"] button[aria-selected="true"] { color: #3cd2c8 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard(inputs: dict[str, float], summary, breakdown, predicted_savings: float) -> None:
    """Render the high-level financial health dashboard."""
    st.subheader("This month at a glance")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Income", format_currency(inputs["income"]))
    metric_columns[1].metric("Tracked expenses", format_currency(summary.total_expenses))
    metric_columns[2].metric(
        "Disposable savings",
        format_currency(summary.disposable_savings),
        delta=f"{summary.savings_rate:.1f}% of income",
    )
    metric_columns[3].metric(
        "Model estimate",
        format_currency(predicted_savings),
        delta=f"{predicted_savings - summary.disposable_savings:+,.0f} vs. calculated",
    )

    target_rate = 20.0
    progress = min(max(summary.savings_rate / target_rate, 0.0), 1.0)
    st.progress(progress, text=f"Savings-rate progress toward a {target_rate:.0f}% monthly target")

    insight_column, chart_column = st.columns((0.85, 1.15), gap="large")
    with insight_column:
        st.markdown("#### Budget health")
        if summary.disposable_savings <= 0:
            st.error("Your tracked spending is higher than or equal to your income.")
        elif summary.savings_rate < target_rate:
            st.warning("You have a positive balance, but are below a 20% savings target.")
        else:
            st.success("Your plan supports a 20% or higher savings rate.")
        st.info(savings_recommendation(inputs["income"], summary.disposable_savings))
        st.markdown(
            f"**Largest category:** {summary.largest_expense_category.title()} "
            f"at {format_currency(summary.largest_expense_amount)}."
        )
    with chart_column:
        st.markdown("#### Expense breakdown")
        st.bar_chart(breakdown.set_index("category")["amount"], color="#147d92")

    with st.expander("Category breakdown", expanded=False):
        displayed = breakdown.copy()
        displayed["amount"] = displayed["amount"].map(format_currency)
        displayed["share_of_expenses"] = displayed["share_of_expenses"].map(lambda value: f"{value:.1f}%")
        st.dataframe(displayed, width="stretch", hide_index=True)


def render_insights(comparison, summary) -> None:
    """Render benchmark comparisons and focused spending observations."""
    st.subheader("Spending insights")
    st.caption("Comparison is against the 12-row sample dataset, not a population benchmark.")
    st.bar_chart(
        comparison.set_index("category")[["your_spending", "sample_average"]],
        color=["#147d92", "#9fb3c8"],
    )

    above_average = comparison.loc[comparison["difference"] > 0].sort_values("difference", ascending=False)
    left, right = st.columns(2)
    with left:
        st.markdown("#### Where you are spending more")
        if above_average.empty:
            st.success("Every tracked category is at or below the sample average.")
        else:
            for row in above_average.itertuples(index=False):
                st.write(f"• **{row.category}:** {format_currency(row.difference)} above the sample average")
    with right:
        st.markdown("#### One useful focus")
        largest = comparison.loc[comparison["category"].str.lower() == summary.largest_expense_category].iloc[0]
        if largest.difference > 0:
            st.write(
                f"{largest.category} is your largest category and is {format_currency(largest.difference)} "
                "above the sample average. Review this category first."
            )
        else:
            st.write(
                f"{largest.category} is your largest category, but it is not above the sample average. "
                "Look across smaller categories for easy reductions."
            )

    displayed = comparison.copy()
    for column in ("your_spending", "sample_average", "difference"):
        displayed[column] = displayed[column].map(format_currency)
    st.dataframe(displayed, width="stretch", hide_index=True)


def render_goal_planner(summary, inputs: dict[str, float]) -> None:
    """Render an interactive target and what-if savings planner."""
    st.subheader("Savings goal planner")
    planner_left, planner_right = st.columns(2, gap="large")
    with planner_left:
        target_amount = st.number_input(
            "Savings goal (₹)", min_value=1_000.0, value=100_000.0, step=5_000.0
        )
        target_rate = st.slider("Desired monthly savings rate", min_value=5, max_value=50, value=20, step=5)
        feasible_monthly_goal = calculate_savings_goal(
            inputs["income"], summary.disposable_savings, target_rate / 100
        )
        months = estimate_months_to_goal(target_amount, summary.disposable_savings)
        st.metric("Achievable monthly set-aside", format_currency(feasible_monthly_goal))
        if months is None:
            st.warning("A timeline is unavailable until your disposable savings are positive.")
        else:
            st.metric("Time to goal at current plan", f"{months:.1f} months")

    with planner_right:
        reduction = st.slider("Try a total-expense reduction", min_value=0, max_value=50, value=10, step=5)
        reduced_expenses = summary.total_expenses * (1 - reduction / 100)
        scenario_savings = inputs["income"] - reduced_expenses
        scenario_months = estimate_months_to_goal(target_amount, scenario_savings)
        st.metric("Savings in this scenario", format_currency(scenario_savings))
        st.metric(
            "Time to goal after reduction",
            f"{scenario_months:.1f} months" if scenario_months is not None else "Not currently fundable",
        )
        improvement = scenario_savings - summary.disposable_savings
        st.caption(f"A {reduction}% reduction frees up an estimated {format_currency(improvement)} each month.")

    needed_for_rate = max(inputs["income"] * target_rate / 100 - summary.disposable_savings, 0)
    if needed_for_rate:
        st.info(
            f"To reach your selected {target_rate}% monthly savings rate, reduce spending by "
            f"about {format_currency(needed_for_rate)} per month."
        )
    else:
        st.success(f"Your current plan already supports a {target_rate}% monthly savings rate.")


def render_data_and_model(data, training, next_month_expenses: float) -> None:
    """Render transparent historical data and demonstration-model context."""
    st.subheader("Sample data and model")
    historical = data.copy()
    historical["total_expenses"] = historical[["food", "transportation", "entertainment", "utilities", "shopping"]].sum(axis=1)
    st.line_chart(historical.set_index("month")[["total_expenses", "savings"]], color=["#147d92", "#e29b45"])

    first, second, third = st.columns(3)
    first.metric("Sample records", f"{len(data)} months")
    second.metric("Holdout MAE", format_currency(training.mae))
    third.metric("Next-month expense trend", format_currency(next_month_expenses))
    st.caption(f"Holdout R²: {training.r2:.3f} · Training rows: {training.train_rows} · Test rows: {training.test_rows}")
    st.warning(
        "Educational estimate only — not financial advice. The dataset has only 12 records and "
        "savings is calculated directly from the tracked input columns, so model metrics are illustrative."
    )
    st.dataframe(data, width="stretch", hide_index=True)


def main() -> None:
    st.set_page_config(page_title="SmartSpend | Financial Dashboard", page_icon="💰", layout="wide")
    inject_styles()
    inputs = render_sidebar()

    try:
        data = get_expenses()
        training = get_training_result()
        summary = build_budget_summary(**inputs)
        breakdown = build_expense_breakdown(**inputs)
        comparison = compare_to_sample(data, **inputs)
        predicted_savings = predict_savings(training.model, **inputs)
        next_month_expenses = predict_next_month_expenses(data)
    except (DataValidationError, OSError) as error:
        st.error(f"SmartSpend could not calculate your plan: {error}")
        st.stop()

    st.markdown(
        """
        <section class="hero">
            <h1>SmartSpend</h1>
            <p>Turn monthly spending into clear, practical learning insights.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Educational estimate · No account connection · No personal data is stored")

    dashboard_tab, insights_tab, planner_tab, data_tab = st.tabs(
        ["Dashboard", "Spending insights", "Goal planner", "Data & model"]
    )
    with dashboard_tab:
        render_dashboard(inputs, summary, breakdown, predicted_savings)
    with insights_tab:
        render_insights(comparison, summary)
    with planner_tab:
        render_goal_planner(summary, inputs)
    with data_tab:
        render_data_and_model(data, training, next_month_expenses)


if __name__ == "__main__":
    main()
