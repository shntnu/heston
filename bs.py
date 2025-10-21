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
    S0_slider = mo.ui.slider(50, 150, value=100, step=1, label="S0 (initial price)", full_width=True)
    r_slider = mo.ui.slider(0, 0.10, value=0.03, step=0.01, label="r (risk-free rate)", full_width=True)
    q_slider = mo.ui.slider(0, 0.10, value=0.00, step=0.01, label="q (dividend yield)", full_width=True)
    sigma_slider = mo.ui.slider(0.05, 2.00, value=0.20, step=0.05, label="σ (volatility)", full_width=True)
    T_slider = mo.ui.slider(0.25, 2.0, value=1.0, step=0.25, label="T (years)", full_width=True)
    K_slider = mo.ui.slider(50, 150, value=100, step=1, label="K (strike price)", full_width=True)

    mo.vstack([
        mo.md("## Black-Scholes Monte Carlo Simulation"),
        mo.md("### Price Parameters"),
        mo.hstack([S0_slider, K_slider], widths=[1, 1]),
        mo.md("### Market Parameters"),
        mo.hstack([r_slider, q_slider], widths=[1, 1]),
        mo.md("### Model Parameters"),
        mo.hstack([sigma_slider, T_slider], widths=[1, 1])
    ])
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
    return (
        K,
        S,
        ST,
        T,
        call_price_bs,
        call_price_mc,
        mean_ST_mc,
        mean_ST_theory,
        se,
        steps,
    )


@app.cell
def _(call_price_bs, call_price_mc, mean_ST_mc, mean_ST_theory, mo, se):
    mo.md(
        f"""
    ### Results

    **Forward/Mean Check:**
    - Mean S_T (MC): {mean_ST_mc:.4f}
    - Mean S_T (Theory): {mean_ST_theory:.4f}

    **Call Option Pricing:**
    - Call price (MC): {call_price_mc:.4f} ± {1.96*se:.4f} (95% CI)
    - Call price (BS): {call_price_bs:.4f}
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ### Visualizations

    Below are visualizations of the Monte Carlo simulation results.
    """
    )
    return


@app.cell
def _(K, S, ST, T, np, steps):
    import matplotlib.pyplot as plt

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Plot sample price paths
    n_paths_to_plot = 100
    time_grid = np.linspace(0, T, steps + 1)

    ax1 = axes[0]
    for i in range(n_paths_to_plot):
        ax1.plot(time_grid, S[i, :], alpha=0.3, linewidth=0.5, color='steelblue')
    ax1.axhline(y=K, color='red', linestyle='--', linewidth=2, label=f'Strike K={K}')
    ax1.set_xlabel('Time (years)')
    ax1.set_ylabel('Price')
    ax1.set_title(f'Sample Price Paths (n={n_paths_to_plot})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Distribution of final prices
    ax2 = axes[1]
    ax2.hist(ST, bins=100, alpha=0.7, color='steelblue', edgecolor='black', density=True)
    ax2.axvline(x=ST.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean={ST.mean():.2f}')
    ax2.axvline(x=K, color='red', linestyle='--', linewidth=2, label=f'Strike={K}')
    ax2.set_xlabel('Final Price S_T')
    ax2.set_ylabel('Density')
    ax2.set_title('Distribution of Final Prices')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Call option payoff distribution
    ax3 = axes[2]
    payoffs = np.maximum(ST - K, 0.0)
    ax3.hist(payoffs, bins=100, alpha=0.7, color='green', edgecolor='black', density=True)
    ax3.axvline(x=payoffs.mean(), color='darkgreen', linestyle='--', linewidth=2, 
                label=f'Mean Payoff={payoffs.mean():.2f}')
    ax3.set_xlabel('Call Payoff')
    ax3.set_ylabel('Density')
    ax3.set_title('Call Option Payoff Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.gca()
    return


if __name__ == "__main__":
    app.run()
