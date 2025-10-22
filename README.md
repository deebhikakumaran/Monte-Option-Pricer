# Indian Stock Option Pricing Dashboard

[![Live Site](https://img.shields.io/badge/Visit-Deployed%20App-brightgreen?style=for-the-badge)](https://delta-dice.streamlit.app)

> Monte Carlo & Black-Scholes Models for NSE Equities

---

## Overview

This project is a **quantitative finance dashboard** designed to calculate and visualize the price of **options** on Indian National Stock Exchange equities.  

It demonstrates two fundamental option pricing methodologies:

- **Black-Scholes-Merton Model** – Analytical, closed-form solution.
- **Monte Carlo Simulation** – Numerical method using **Geometric Brownian Motion** to estimate expected payoffs.

---

### Key Quantitative Features

- **Dual-Model Comparison** – View option prices from both Monte Carlo and Black-Scholes models side-by-side.  

- **Geometric Brownian Motion** – Core stochastic model used for Monte Carlo path simulation.  

- **Volatility Estimation** – Automatically computes **historical annual volatility (σ)** from 5 years of NSE stock data using log returns.  

- **Customizable Simulation Paths** – Supports both **single-step** (European payoff) and **multi-step** (American-capable) simulations.  

- **Put-Call Parity Check** – Validates the arbitrage relationship

    **Formula:**
        `C - P - (S_0 - K e^{-rT}) = 0`

    **Where**
    - `C` = Call option price 
    - `P` = Put option price 
    - `S₀` = Current stock price
    - `K` = Strike price  
    - `r` = Risk-free rate  
    - `T` = Time to maturity 

- **Interactive Visualizations** – Built with **Plotly** and **Matplotlib** for intuitive price paths, histograms, and comparisons.

---

## Installation

### Local Setup

1. **Clone the repository**:
```bash
git clone https://github.com/deebhikakumaran/Monte-Option-Pricer.git
cd Monte-Option-Pricer
```

2. **Create and Activate a Virtual Environment**:
```bash
python -m venv venv
source venv/bin/activate    # macOS/Linux
.\venv\Scripts\activate     # Windows
```

3. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

4. **Run the Application**:
```bash
streamlit run app.py
```

---

## Dashboard User Guide

### Sidebar Inputs

| Parameter                 | Input Type   | Role in Pricing                                                   |
| ------------------------- | ------------ | ----------------------------------------------------------------- |
| **NSE Stock Symbol**      | Dropdown     | Selects the underlying stock (`TCS.NS`, `RELIANCE.NS`).     |
| **Strike Price**      | Number Input | Sets the exercise price.                                          |
| **Option Expiry Date**    | Date Picker  | Defines the time to maturity.                               |
| **Risk-Free Rate**    | Number Input | Used for discounting expected payoffs.                            |
| **Annual Volatility** | Auto/Manual  | The most critical parameter for both models.                      |
| **MC Simulations**    | Number Input | Determines sample size for averaging.                             |
| **MC Path Type**          | Selectbox    | Choose between direct calculation or multi-step simulation. |

### Main Panel Sections

| Section                          | Description                                      | Quantitative Insight                                     |
| -------------------------------- | ------------------------------------------------ | -------------------------------------------------------- |
| **Contract Info**                | Shows current price, strike price, time to expiry, and annualized volatility.     | Confirms input parameters used.                          |
| **Option Premium Comparison**    | Table with Call and Put prices from both models. | Validates MC convergence to BS theoretical value.        |
| **Parity Check Difference**            |  C - P - (S_0 - K e^{-rT}).    | Should be near zero for arbitrage-free consistency.      |
| **Simulated Price Distribution** | Interactive Plotly histogram.                    | Illustrates log-normal distribution of simulated prices. |
| **Sample Simulated Paths**       | Matplotlib plot for Multi-Step Simulation.                    | Shows random GBM paths and path-dependence.              |

---

## Author

Deebhika Kumaran - Building projects that bridge theory & practice in finance

---

## License

This project is licensed under the MIT License — free for personal and educational use.

---