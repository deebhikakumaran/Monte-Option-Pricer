# app.py
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import math
from nselib import capital_market

st.set_page_config(page_title="Indian Stock Option Pricing", layout="wide")
st.title("📈 Option Pricing Dashboard")
st.subheader("Simulate option prices using Monte Carlo for NSE stocks")
st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------
# Data Fetching Functions
# -----------------------------

@st.cache_data(ttl=60*60*24) 
def get_all_nse_symbols():
    """Fetches the list of all valid NSE stock symbols."""
    try:
        # CORRECTED LINE: Removed 'index=False'
        df = capital_market.equity_list() 
        
        # Extract the symbol column and convert to a sorted list
        # Note: Depending on the nselib version, the column name might be 'SYMBOL' or 'Symbol'.
        # We'll stick to 'SYMBOL' as in your original code, but be aware of this potential variation.
        symbols = sorted(df['SYMBOL'].tolist())
        
        # Append .NS for yfinance compatibility
        yfinance_symbols = [s + ".NS" for s in symbols]
        
        # Add a default value if needed, or ensure TCS.NS is in the list
        if "TCS.NS" not in yfinance_symbols:
             yfinance_symbols.insert(0, "TCS.NS")
             
        return yfinance_symbols
        
    except Exception as e:
        # Added a clearer message for debugging
        st.error(f"Error fetching NSE symbol list. Please check nselib version and column name. Original error: {e}")
        return ["TCS.NS", "RELIANCE.NS", "INFY.NS"] # Fallback

# -----------------------------
# Fetch stock data
# -----------------------------
@st.cache_data(ttl=3600)
def fetch_stock_prices(symbol):
    try:
        # Use a longer period for historical volatility calculation
        start_date = (datetime.date.today() - datetime.timedelta(days=5*365)).strftime('%Y-%m-%d')
        df = yf.download(symbol, start=start_date, progress=False)["Close"].dropna()
        return df
    except Exception as e:
        return None
    
# -----------------------------
# User inputs
# -----------------------------

# Fetch the list once and use it for the selectbox
nse_symbols = get_all_nse_symbols()
default_index = nse_symbols.index("TCS.NS") if "TCS.NS" in nse_symbols else 0
stock_symbol = st.selectbox(
    "Select NSE Stock Symbol", 
    options=nse_symbols, 
    index=default_index
)

# stock_symbol = st.text_input("Enter NSE stock symbol", value="TCS.NS")

# Strike price
strike_price = st.number_input("Strike Price (K)", min_value=1.0, value=1000.0, step=1.0)

# Current date
today = datetime.date.today()

# Let user select expiry date
expiry_date = st.date_input("Select Option Expiry Date", min_value=today)

# Monte Carlo parameters
simulations = st.number_input(
    "Number of Monte Carlo Simulations",
    min_value=1000,
    max_value=500000,
    value=50000,
    step=1000,
)

# Risk-free rate
r = st.number_input(
    "Risk-Free Rate (Annual, 0.05 for 5%)", min_value=0.0, max_value=1.0, value=0.05, step=0.005
)

# Allow user to override volatility (optional)
use_manual_vol = st.checkbox("Manually set annual volatility instead of using historical estimate?", value=False)
manual_sigma = None
if use_manual_vol:
    manual_sigma = st.number_input("Annual Volatility, 0.25 for 25%", min_value=0.001, max_value=5.0, value=0.25, step=0.01)

# Option: choose simulation type
simulation_type = st.selectbox("Simulation Type", ["Single-step to expiry (European)", "Multi-step daily paths (American-capable)"])

st.markdown("---")

prices = fetch_stock_prices(stock_symbol)

if prices is None or len(prices) < 30:
    st.error(f"Failed to fetch enough historical data for {stock_symbol}. Check the symbol and try again.")
    st.stop() # Stop execution if data fetch failed

if prices is not None:
    S0 = float(prices.iloc[-1])
    st.subheader("Contract Information")
    st.write(f"Current Stock Price: ₹{S0:.2f}")

    # Daily log returns
    log_returns = np.log(prices / prices.shift(1)).dropna()
    sigma_daily = float(log_returns.std())

    # Time to expiry in trading days
    expiry_days = int(np.busday_count(today, expiry_date))
    if expiry_days <= 0:
        st.error("Expiry must be at least one business day after today.")
        st.stop()

    T_years = expiry_days / 252

    # Choose sigma
    if use_manual_vol and manual_sigma is not None:
        sigma_annual = float(manual_sigma)
        sigma_daily_used = sigma_annual / np.sqrt(252)
        st.write(f"Using Manual Annualized Volatility: {sigma_annual:.2%}")
    else:
        sigma_annual = sigma_daily * np.sqrt(252)
        sigma_daily_used = sigma_daily
        st.write(f"Estimated Annualized Volatility: {sigma_annual:.2%}")

    st.write(f"Time to expiry: {expiry_days} trading days ≈ {T_years:.3f} years")

    # -----------------------------
    # Monte Carlo single step simulation (simulate S_T directly)
    # -----------------------------
    np.random.seed(42)
    N_sim = int(simulations)

    # risk-neutral single-step drift (use r for pricing)
    drift_rn = (r - 0.5 * sigma_annual**2) * T_years
    diffusion_rn = sigma_annual * np.sqrt(T_years) * np.random.normal(0, 1, N_sim)
    final_prices_single = S0 * np.exp(drift_rn + diffusion_rn)

    # -----------------------------
    # Monte Carlo: multi step simulation (daily)
    # -----------------------------
    N_steps = expiry_days
    dt = 1.0 / 252.0

    # simulate daily steps using risk-neutral drift (r)
    # shape: (N_steps, N_sim)
    Z = np.random.normal(0, 1, size=(N_steps, N_sim))
    # use sigma_annual and dt in formula: S_t+1 = S_t * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    increments = (r - 0.5 * sigma_annual**2) * dt + sigma_annual * np.sqrt(dt) * Z
    # cumulative log returns
    log_price_paths = np.cumsum(increments, axis=0)
    # price paths
    price_paths = S0 * np.exp(log_price_paths)
    price_paths = np.vstack([np.full((1, N_sim), S0), price_paths])
    final_prices_multi = price_paths[-1, :]

    # Select final_prices depending on simulation_type
    if simulation_type.startswith("Single-step"):
        final_simulated_prices = final_prices_single
    else:
        final_simulated_prices = final_prices_multi

    # Option payoffs
    K = float(strike_price)
    call_payoffs = np.maximum(final_simulated_prices - K, 0.0)
    put_payoffs = np.maximum(K - final_simulated_prices, 0.0)

    # Monte Carlo premium (risk-neutral)
    discount = math.exp(-r * T_years)
    call_premium_mc = discount * np.mean(call_payoffs)
    put_premium_mc = discount * np.mean(put_payoffs)

    # -----------------------------
    # Black-Scholes analytic prices (European) - requires sigma annual and T_years
    # -----------------------------
    def norm_cdf(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def black_scholes(S, K, T, r, sigma):
        if T <= 0:
            # immediate: payoff
            call = max(S - K, 0.0)
            put = max(K - S, 0.0)
            return call, put
        sqrtT = math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT
        call = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
        put = K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
        return call, put

    call_premium_bs, put_premium_bs = black_scholes(S0, K, T_years, r, sigma_annual)

    # Parity checks
    parity_diff_mc = (call_premium_mc - put_premium_mc) - (S0 - strike_price * np.exp(-r * T_years))
    parity_diff_bs = (call_premium_bs - put_premium_bs) - (S0 - strike_price * np.exp(-r * T_years))

    # -----------------------------
    # Display results
    # -----------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Option Premiums (Monte Carlo vs Black-Scholes)")

    # Create a comparison DataFrame
    premium_df = pd.DataFrame(
        {
            "Model": ["Monte Carlo", "Black-Scholes"],
            "Call (₹)": [call_premium_mc, call_premium_bs],
            "Put (₹)": [put_premium_mc, put_premium_bs],
            "Discount Factor": [discount, discount],
            "Call - Put (C-P)": [call_premium_mc - put_premium_mc, call_premium_bs - put_premium_bs],
            "Parity Difference": [parity_diff_mc, parity_diff_bs],
        }
    )

    # Apply number formatting only to numeric columns
    numeric_cols = premium_df.select_dtypes(include=["float", "int"]).columns
    st.dataframe(
        premium_df.style.format(subset=numeric_cols, formatter="{:.4f}")
    )

    st.caption("Note: Both Monte Carlo and Black-Scholes models use the same discount factor e^(-rT) to convert expected future payoffs to present value.")

    # -----------------------------
    # Plot simulated prices
    # -----------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Simulated Stock Price Distribution at Expiry")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.hist(final_simulated_prices, bins='auto', color='skyblue', edgecolor='black')
    ax1.axvline(K, color='red', linestyle='--', label=f"Strike (K={K:.2f})")
    ax1.axvline(S0, color='blue', linestyle='--', label=f"S0={S0:.2f}")
    ax1.set_xlabel("Stock Price at Expiry")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Simulated Final Prices")
    ax1.legend()
    st.pyplot(fig1)

    # -----------------------------
    # Plot few sample paths from multi-step (only if multi-step was computed)
    # -----------------------------
    if simulation_type.startswith("Multi-step"):
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Simulated Price Paths (sample)")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sample_n = min(50, N_sim)
        idx = np.random.choice(N_sim, size=sample_n, replace=False)
        tgrid = np.arange(0, N_steps + 1)
        for i in idx[:min(10, sample_n)]:  # show up to 10 paths for clarity
            ax2.plot(tgrid, price_paths[:, i], linewidth=0.8, alpha=0.8)
        ax2.set_xlabel("Day")
        ax2.set_ylabel("Price")
        ax2.set_title("Sample Simulated Price Paths (multi-step)")
        st.pyplot(fig2)

    # -----------------------------
    # Plot option payoff functions
    # -----------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Option Payoff vs Price at Expiry")
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    # price_grid = np.linspace(min(final_simulated_prices), max(final_simulated_prices), 1000)
    price_grid = np.linspace(max(0.01, final_simulated_prices.min()), final_simulated_prices.max(), 1000)
    call_payoff_grid = np.maximum(price_grid - K, 0)
    put_payoff_grid = np.maximum(K - price_grid, 0)
    ax3.plot(price_grid, call_payoff_grid, label="Call Payoff", linewidth=1.5, color="green")
    ax3.plot(price_grid, put_payoff_grid, label="Put Payoff", linewidth=1.5, color="orange")
    ax3.axvline(S0, color="blue", linestyle="--", label=f"S0={S0:.2f}")
    ax3.set_xlabel("Price at Expiry")
    ax3.set_ylabel("Payoff")
    ax3.set_title("Option Payoffs")
    ax3.legend()
    st.pyplot(fig3)

    st.success("Simulation complete. Use the controls to change symbol, strike, expiry, volatility, or simulation type.")

