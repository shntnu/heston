import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from math import log, sqrt, exp
    from scipy.stats import norm
    return exp, log, mo, norm, np, sqrt


@app.cell
def _(exp, log, norm, np, sqrt):
    # ----------------------------
    # Core: simulate GBM (Black–Scholes) paths
    # ----------------------------
    def simulate_gbm_paths(S0, r, q, sigma, T, n_steps, n_paths, seed=0):
        """
        Returns:
          S: array shape (n_paths, n_steps+1) with prices including S0 at t=0
        """
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        drift = (r - q - 0.5 * sigma**2) * dt
        diff  = sigma * sqrt(dt)

        # Pre-allocate and fill
        S = np.empty((n_paths, n_steps + 1), dtype=float)
        S[:, 0] = S0

        # Vectorized simulation in log-space (exact step for GBM)
        Z = rng.standard_normal(size=(n_paths, n_steps))
        log_increments = drift + diff * Z
        log_paths = np.cumsum(log_increments, axis=1)  # cumulative over time
        S[:, 1:] = S0 * np.exp(log_paths)
        return S

    # ----------------------------
    # Black–Scholes analytic price for a European call
    # ----------------------------
    def bs_call_price(S0, K, r, q, sigma, T):
        if sigma <= 0 or T <= 0:
            # handle edge cases conservatively
            forward = S0 * exp((r - q) * T)
            return max(forward - K, 0) * exp(-r * T)
        d1 = (log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        return S0 * exp(-q * T) * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    return bs_call_price, simulate_gbm_paths


@app.cell
def _(mo):
    # Parameters (risk-neutral)
    S0_slider = mo.ui.slider(50, 150, value=100, step=1, label="S0 (initial price)")
    r_slider = mo.ui.slider(0, 0.10, value=0.03, step=0.01, label="r (risk-free rate)")
    q_slider = mo.ui.slider(0, 0.10, value=0.00, step=0.01, label="q (dividend yield)")
    sigma_slider = mo.ui.slider(0.05, 0.50, value=0.20, step=0.05, label="σ (volatility)")
    T_slider = mo.ui.slider(0.25, 2.0, value=1.0, step=0.25, label="T (years)")
    K_slider = mo.ui.slider(50, 150, value=100, step=1, label="K (strike price)")

    mo.md(f"""
    ## Black-Scholes Monte Carlo Simulation

    Adjust the parameters below:

    {S0_slider}
    {r_slider}
    {q_slider}
    {sigma_slider}
    {T_slider}
    {K_slider}
    """)
    return K_slider, S0_slider, T_slider, q_slider, r_slider, sigma_slider


@app.cell
def _(
    K_slider,
    S0_slider,
    T_slider,
    bs_call_price,
    exp,
    np,
    q_slider,
    r_slider,
    sigma_slider,
    simulate_gbm_paths,
):
    # Fixed simulation parameters
    steps = 252
    paths = 100_000

    # Get parameter values
    S0 = S0_slider.value
    r = r_slider.value
    q = q_slider.value
    sigma = sigma_slider.value
    T = T_slider.value
    K = K_slider.value

    # Run simulation
    S = simulate_gbm_paths(S0, r, q, sigma, T, steps, paths, seed=42)
    ST = S[:, -1]

    # 1) Forward/mean check: E[S_T] = S0 * exp((r - q) * T)
    mean_ST_mc = ST.mean()
    mean_ST_theory = S0 * exp((r - q) * T)

    # 2) Price a European call and compare to analytic Black–Scholes
    disc = exp(-r * T)
    call_payoff_mc = np.maximum(ST - K, 0.0)
    call_price_mc = disc * call_payoff_mc.mean()
    call_price_bs = bs_call_price(S0, K, r, q, sigma, T)
    # Also estimate MC standard error
    se = disc * call_payoff_mc.std(ddof=1) / np.sqrt(paths)
    return call_price_bs, call_price_mc, mean_ST_mc, mean_ST_theory, se


@app.cell
def _(call_price_bs, call_price_mc, mean_ST_mc, mean_ST_theory, mo, se):
    mo.md(f"""
    ### Results

    **Forward/Mean Check:**
    - Mean S_T (MC): {mean_ST_mc:.4f}
    - Mean S_T (Theory): {mean_ST_theory:.4f}

    **Call Option Pricing:**
    - Call price (MC): {call_price_mc:.4f} ± {1.96*se:.4f} (95% CI)
    - Call price (BS): {call_price_bs:.4f}
    """)
    return


if __name__ == "__main__":
    app.run()
