import streamlit as st
import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Helper function to format numbers with thousands separators
def format_number(num):
    return f"{num:,.0f}"

# Helper function to parse input with thousand separators
def parse_number(num_string):
    return int(num_string.replace(',', ''))

def calculate_expected_return(investments):
    total_amount = sum(inv["amount"] for inv in investments)
    weighted_return = sum(inv["amount"] * inv["return"] for inv in investments) / total_amount
    return weighted_return

def get_expense_multiplier(year, expense_multipliers):
    cumulative_years = 0
    for multiplier, duration in expense_multipliers:
        if duration is None or year < cumulative_years + duration:
            return multiplier
        cumulative_years += duration
    return expense_multipliers[-1][0]

def simulate_portfolio(initial_portfolio, initial_expenses, inflation_rate, tax_rate, income1, income2, expense_multipliers, investments, years):
    portfolio = initial_portfolio
    expenses = initial_expenses
    portfolio_values = []
    annual_returns = []
    after_tax_returns = []

    for year in range(years):
        portfolio_values.append(portfolio)

        # Calculate returns for each investment
        total_return = 0
        for inv in investments:
            amount = inv["amount"] if year == 0 else portfolio * (inv["amount"] / initial_portfolio)
            annual_return = np.random.normal(inv["return"], inv["volatility"])
            total_return += amount * annual_return

        # Calculate annual return percentage
        annual_return_pct = total_return / portfolio
        annual_returns.append(annual_return_pct)

        # Calculate after-tax return
        tax_amount = total_return * tax_rate
        after_tax_return = total_return - tax_amount
        after_tax_return_pct = after_tax_return / portfolio
        after_tax_returns.append(after_tax_return_pct)

        # Update portfolio value
        portfolio += after_tax_return
        
        # Apply expense multiplier
        expense_multiplier = get_expense_multiplier(year, expense_multipliers)
        portfolio -= expenses * expense_multiplier

        # Add income sources
        if year < income1["duration"]:
            portfolio += income1["amount"] * (1 + inflation_rate)**year
        if year < income2["duration"]:
            portfolio += income2["amount"] * (1 + inflation_rate)**year

        # Increase expenses for next year due to inflation
        expenses *= (1 + inflation_rate)

    portfolio_values.append(portfolio)  # Add final portfolio value

    return portfolio_values, annual_returns, after_tax_returns

def run_monte_carlo(num_simulations, years, **kwargs):
    all_portfolio_values = []
    all_annual_returns = []
    all_after_tax_returns = []
    for _ in range(num_simulations):
        portfolio_values, annual_returns, after_tax_returns = simulate_portfolio(**kwargs, years=years)
        all_portfolio_values.append(portfolio_values)
        all_annual_returns.append(annual_returns)
        all_after_tax_returns.append(after_tax_returns)
    return np.array(all_portfolio_values), np.array(all_annual_returns), np.array(all_after_tax_returns)

def main():
    st.title('Portfolio Monte Carlo Simulation')

    # Move simulation parameters to the top
    st.header('Simulation Parameters')
    num_simulations = st.number_input('Number of Simulations', 
                                      min_value=100, max_value=10000, value=1000, step=100,
                                      help="The number of times to run the simulation. More simulations increase accuracy but take longer.")
    years = st.number_input('Simulation Years', 
                            min_value=10, max_value=100, value=50, step=1,
                            help="The number of years to project into the future.")

    # Run simulation button at the top
    run_simulation = st.button('Run Simulation')

    # Sidebar for other inputs
    st.sidebar.header('Input Parameters')
    initial_portfolio = st.sidebar.text_input('Initial Portfolio Value (€)', 
                                              value='5,000,000',
                                              help="The total value of your investments at the start of the simulation.")
    initial_portfolio = parse_number(initial_portfolio)

    initial_expenses = st.sidebar.text_input('Initial Annual Expenses (€)', 
                                             value='80,000',
                                             help="Your expected annual living expenses at the start of the simulation.")
    initial_expenses = parse_number(initial_expenses)

    inflation_rate = st.sidebar.slider('Inflation Rate', 
                                       0.0, 0.10, 0.02, 0.001,
                                       help="The expected annual increase in the cost of goods and services.")
    tax_rate = st.sidebar.slider('Tax Rate', 
                                 0.0, 0.50, 0.26, 0.01,
                                 help="The rate at which investment gains are taxed.")
    current_year = st.sidebar.number_input('Current Year', 
                                           value=2025, step=1,
                                           help="The year in which the simulation starts.")
    current_age = st.sidebar.number_input('Current Age', 
                                          value=46, step=1,
                                          help="Your current age at the start of the simulation.")

    st.sidebar.subheader('Income Sources')
    income1_amount = st.sidebar.text_input('Income 1 Amount (€)', 
                                           value='100,000',
                                           help="The annual amount of your first additional income source.")
    income1_amount = parse_number(income1_amount)
    income1_duration = st.sidebar.number_input('Income 1 Duration (years)', 
                                               value=3, step=1,
                                               help="The number of years you expect to receive Income 1.")

    income2_amount = st.sidebar.text_input('Income 2 Amount (€)', 
                                           value='50,000',
                                           help="The annual amount of your second additional income source.")
    income2_amount = parse_number(income2_amount)
    income2_duration = st.sidebar.number_input('Income 2 Duration (years)', 
                                               value=10, step=1,
                                               help="The number of years you expect to receive Income 2.")

    income1 = {"amount": income1_amount, "duration": income1_duration}
    income2 = {"amount": income2_amount, "duration": income2_duration}

    st.sidebar.subheader('Expense Multipliers')
    st.sidebar.markdown("Adjust your expenses for different life stages:")
    expense_multipliers = [
        (st.sidebar.slider('Multiplier 1', 0.5, 2.0, 1.15, help="Expense multiplier for the first period"),
         st.sidebar.number_input('Duration 1', value=5, step=1, help="Duration of the first expense period")),
        (st.sidebar.slider('Multiplier 2', 0.5, 2.0, 1.05, help="Expense multiplier for the second period"),
         st.sidebar.number_input('Duration 2', value=15, step=1, help="Duration of the second expense period")),
        (st.sidebar.slider('Multiplier 3', 0.5, 2.0, 0.95, help="Expense multiplier for the third period"),
         st.sidebar.number_input('Duration 3', value=10, step=1, help="Duration of the third expense period")),
        (st.sidebar.slider('Multiplier 4', 0.5, 2.0, 0.85, help="Expense multiplier for the remaining years"), None)
    ]

    st.sidebar.subheader('Investment Allocations')
    investments = [
        {"amount": parse_number(st.sidebar.text_input('Cash Amount (€)', value='500,000', help="Amount invested in low-risk, low-return assets")),
         "return": st.sidebar.slider('Cash Return', 0.0, 0.10, 0.02, 0.001, help="Expected annual return for cash investments"),
         "volatility": st.sidebar.slider('Cash Volatility', 0.0, 0.20, 0.01, 0.001, help="Expected annual volatility for cash investments"),
         "type": "cash"},
        {"amount": parse_number(st.sidebar.text_input('Cash Plus Amount (€)', value='500,000', help="Amount invested in slightly higher-risk, higher-return assets")),
         "return": st.sidebar.slider('Cash Plus Return', 0.0, 0.10, 0.03, 0.001, help="Expected annual return for cash plus investments"),
         "volatility": st.sidebar.slider('Cash Plus Volatility', 0.0, 0.20, 0.04, 0.001, help="Expected annual volatility for cash plus investments"),
         "type": "cash plus"},
        {"amount": parse_number(st.sidebar.text_input('RP Invest Amount (€)', value='4,000,000', help="Amount invested in higher-risk, higher-return assets")),
         "return": st.sidebar.slider('RP Invest Return', 0.0, 0.20, 0.06, 0.001, help="Expected annual return for RP investments"),
         "volatility": st.sidebar.slider('RP Invest Volatility', 0.0, 0.30, 0.10, 0.001, help="Expected annual volatility for RP investments"),
         "type": "RP Invest"}
    ]

    if run_simulation:
        expected_return = calculate_expected_return(investments)

        monte_carlo_portfolios, monte_carlo_returns, monte_carlo_after_tax_returns = run_monte_carlo(
            num_simulations, years, initial_portfolio=initial_portfolio, initial_expenses=initial_expenses,
            inflation_rate=inflation_rate, tax_rate=tax_rate, income1=income1, income2=income2,
            expense_multipliers=expense_multipliers, investments=investments
        )

        avg_portfolio = np.mean(monte_carlo_portfolios, axis=0)
        std_portfolio = np.std(monte_carlo_portfolios, axis=0)
        lower_portfolio_1sd = avg_portfolio - std_portfolio
        upper_portfolio_1sd = avg_portfolio + std_portfolio
        lower_portfolio_2sd = avg_portfolio - 2*std_portfolio
        upper_portfolio_2sd = avg_portfolio + 2*std_portfolio

        avg_returns = np.mean(monte_carlo_returns, axis=0)
        avg_after_tax_returns = np.mean(monte_carlo_after_tax_returns, axis=0)

        # Plot 1: Portfolio Value Projection
        st.subheader('Portfolio Value Projection')
        st.write("""
        This chart shows the projected growth of your portfolio over time.
        The blue line represents the average portfolio value across all simulations.
        The shaded areas represent the range of possible outcomes within one and two standard deviations.
        A wider range indicates more uncertainty in the projections.
        The chart helps visualize the potential growth and variability of your investments.
        """)
        df_portfolio = pd.DataFrame({
            'Year': range(current_year, current_year + years + 1),
            'Average': avg_portfolio / 1000,
            'Lower 1SD': lower_portfolio_1sd / 1000,
            'Upper 1SD': upper_portfolio_1sd / 1000,
            'Lower 2SD': lower_portfolio_2sd / 1000,
            'Upper 2SD': upper_portfolio_2sd / 1000
        })
        st.line_chart(df_portfolio.set_index('Year'))

        # Plot 2: Nominal vs Real Portfolio Value
        st.subheader('Nominal vs Real Portfolio Value')
        st.write("""
        This chart compares the nominal (not adjusted for inflation) and real (inflation-adjusted) portfolio values.
        The blue line shows the nominal value, which is the actual amount you'd see in your account.
        The green line shows the real value, representing the purchasing power of your portfolio over time.
        The gap between the lines illustrates the impact of inflation on your wealth.
        This comparison helps you understand the true growth of your portfolio in today's terms.
        """)
        real_portfolio = avg_portfolio / (1 + inflation_rate) ** np.arange(years + 1)
        df_nominal_real = pd.DataFrame({
            'Year': range(current_year, current_year + years + 1),
            'Nominal Value': avg_portfolio / 1000,
            'Real Value': real_portfolio / 1000
        })
        st.line_chart(df_nominal_real.set_index('Year'))

        # Plot 3: Returns + Income vs Expenses
        st.subheader('Annual Returns + Income vs Expenses')
        st.write("""
        This chart breaks down your annual financial flows.
        The blue portion of each bar represents investment returns.
        The green portion shows additional income from other sources.
        The red bars indicate your annual expenses.
        Comparing the height of the blue+green bars to the red bars shows your net financial position each year.
        This visualization helps you understand how your income, returns, and expenses interact over time.
        """)
        returns = avg_portfolio[1:] - avg_portfolio[:-1]
        income = np.zeros(years)
        income[:income1['duration']] += income1['amount'] * (1 + inflation_rate) ** np.arange(income1['duration'])
        income[:income2['duration']] += income2['amount'] * (1 + inflation_rate) ** np.arange(income2['duration'])
        
        expenses = initial_expenses * (1 + inflation_rate) ** np.arange(years)
        for i, (multiplier, duration) in enumerate(expense_multipliers):
            if duration is None:
                expenses[sum(d for _, d in expense_multipliers if d is not None):] *= multiplier
            else:
                start = sum(d for _, d in expense_multipliers[:i] if d is not None)
                expenses[start:start+duration] *= multiplier

        df_cashflow = pd.DataFrame({
            'Year': range(current_year, current_year + years),
            'Returns': returns / 1000,
            'Income': income / 1000,
            'Expenses': expenses / 1000
        })
        st.bar_chart(df_cashflow.set_index('Year'))

        st.subheader('Simulation Results')
        final_avg_portfolio = avg_portfolio[-1] / 1000
        final_lower_2sd = lower_portfolio_2sd[-1] / 1000
        final_upper_2sd = upper_portfolio_2sd[-1] / 1000

        st.write(f"After {years} years:")
        st.write(f"Average final portfolio value: {format_number(final_avg_portfolio)}k€")
        st.write(f"Lower bound (-2 std dev): {format_number(final_lower_2sd)}k€")
        st.write(f"Upper bound (+2 std dev): {format_number(final_upper_2sd)}k€")

        total_return = (avg_portfolio[-1] / initial_portfolio) ** (1/years) - 1
        after_tax_total_return = total_return * (1 - tax_rate)
        st.write(f"Average annual return over {years} years (before tax): {total_return*100:.2f}%")
        st.write(f"Average annual return over {years} years (after tax): {after_tax_total_return*100:.2f}%")

        depletion_probability = (monte_carlo_portfolios[:, -1] <= 0).mean() * 100
        st.write(f"Probability of portfolio depletion: {depletion_probability:.2f}%")

        st.write(f"Expected annual return of the portfolio (before tax): {expected_return*100:.2f}%")
        st.write(f"Expected annual return of the portfolio (after tax): {expected_return*(1-tax_rate)*100:.2f}%")

if __name__ == "__main__":
    main()




