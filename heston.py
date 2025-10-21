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
    # Heston (Euler Full Truncation) path simulator
    # ----------------------------
    def simulate_heston_paths_euler_ft(
        S0, v0, r, q, kappa, theta, sigma, rho,
        T, n_steps, n_paths, seed=0
    ):
        """
        Simulate Heston under risk-neutral measure using Euler Full Truncation (FT).

        S dynamics: dS = (r - q) S dt + sqrt(v) S dW_S
        v dynamics: dv = kappa (theta - v) dt + sigma sqrt(v) dW_v
        corr(dW_S, dW_v) = rho

        Returns
        -------
        S : (n_paths, n_steps+1) array of prices
        v : (n_paths, n_steps+1) array of variances
        """
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        sqdt = sqrt(dt)

        S = np.empty((n_paths, n_steps + 1), dtype=float)
        v = np.empty((n_paths, n_steps + 1), dtype=float)
        S[:, 0] = S0
        v[:, 0] = v0

        # Generate correlated shocks
        Z_v = rng.standard_normal(size=(n_paths, n_steps))
        Z_p = rng.standard_normal(size=(n_paths, n_steps))  # independent
        Z_s = rho * Z_v + sqrt(1.0 - rho**2) * Z_p  # correlated with Z_v

        # Work arrays
        S_curr = np.full(n_paths, S0, dtype=float)
        v_curr = np.full(n_paths, v0, dtype=float)

        for t in range(n_steps):
            v_plus = np.maximum(v_curr, 0.0)  # clamp for sqrt

            # Variance update (Euler FT): clamp inside sqrt and after the step
            v_next = v_curr + kappa * (theta - v_plus) * dt + sigma * np.sqrt(v_plus) * sqdt * Z_v[:, t]
            v_next = np.maximum(v_next, 0.0)

            # Log-price exact-style step using v_plus
            drift = (r - q - 0.5 * v_plus) * dt
            diff = np.sqrt(v_plus) * sqdt * Z_s[:, t]
            S_curr = S_curr * np.exp(drift + diff)

            # Store and roll
            S[:, t+1] = S_curr
            v[:, t+1] = v_next
            v_curr = v_next

        return S, v

    # ----------------------------
    # Black–Scholes analytic price for comparison
    # ----------------------------
    def bs_call_price(S0, K, r, q, sigma, T):
        if sigma <= 0 or T <= 0:
            forward = S0 * exp((r - q) * T)
            return max(forward - K, 0) * exp(-r * T)
        d1 = (log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        return S0 * exp(-q * T) * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)

    return bs_call_price, simulate_heston_paths_euler_ft


@app.cell
def _(mo):
    # Parameters
    S0_slider = mo.ui.slider(50, 150, value=100, step=1, label="S0 (initial price)", full_width=True)
    K_slider = mo.ui.slider(50, 150, value=100, step=1, label="K (strike price)", full_width=True)
    r_slider = mo.ui.slider(0, 0.10, value=0.03, step=0.01, label="r (risk-free rate)", full_width=True)
    q_slider = mo.ui.slider(0, 0.10, value=0.00, step=0.01, label="q (dividend yield)", full_width=True)
    T_slider = mo.ui.slider(0.25, 2.0, value=1.0, step=0.25, label="T (years)", full_width=True)

    # Heston-specific parameters
    v0_slider = mo.ui.slider(0.01, 0.20, value=0.04, step=0.01, label="v0 (initial variance)", full_width=True)
    kappa_slider = mo.ui.slider(0.5, 5.0, value=1.5, step=0.1, label="κ (mean reversion speed)", full_width=True)
    theta_slider = mo.ui.slider(0.01, 0.20, value=0.04, step=0.01, label="θ (long-run variance)", full_width=True)
    sigma_slider = mo.ui.slider(0.1, 2.0, value=0.5, step=0.1, label="σ (vol-of-vol)", full_width=True)
    rho_slider = mo.ui.slider(-1.0, 1.0, value=-0.7, step=0.1, label="ρ (correlation)", full_width=True)

    mo.vstack([
        mo.md("## Heston Model Monte Carlo Simulation"),
        mo.md("### Price Parameters"),
        mo.hstack([S0_slider, K_slider], widths=[1, 1]),
        mo.md("### Market Parameters"),
        mo.hstack([r_slider, q_slider, T_slider], widths=[1, 1, 1]),
        mo.md("### Heston Model Parameters"),
        mo.hstack([v0_slider, kappa_slider, theta_slider], widths=[1, 1, 1]),
        mo.hstack([sigma_slider, rho_slider], widths=[1, 1]),
    ])
    return (
        K_slider,
        S0_slider,
        T_slider,
        kappa_slider,
        q_slider,
        r_slider,
        rho_slider,
        sigma_slider,
        theta_slider,
        v0_slider,
    )


@app.cell
def _(
    K_slider,
    S0_slider,
    T_slider,
    exp,
    kappa_slider,
    np,
    q_slider,
    r_slider,
    rho_slider,
    sigma_slider,
    simulate_heston_paths_euler_ft,
    theta_slider,
    v0_slider,
):
    # Fixed simulation parameters
    steps = 252
    paths = 100_000

    # Get parameter values
    S0 = S0_slider.value
    K = K_slider.value
    r = r_slider.value
    q = q_slider.value
    T = T_slider.value
    v0 = v0_slider.value
    kappa = kappa_slider.value
    theta = theta_slider.value
    sigma = sigma_slider.value
    rho = rho_slider.value

    # Run Heston simulation
    S, v = simulate_heston_paths_euler_ft(
        S0, v0, r, q, kappa, theta, sigma, rho,
        T, steps, paths, seed=42
    )
    ST = S[:, -1]
    vT = v[:, -1]

    # Forward/mean check: E[S_T] = S0 * exp((r - q) * T)
    mean_ST_mc = ST.mean()
    mean_ST_theory = S0 * exp((r - q) * T)

    # Variance checks
    variance_negative_share = (v < 0).mean()
    mean_vT = vT.mean()

    # Price a European call
    disc = exp(-r * T)
    call_payoff_mc = np.maximum(ST - K, 0.0)
    call_price_mc = disc * call_payoff_mc.mean()
    se = disc * call_payoff_mc.std(ddof=1) / np.sqrt(paths)

    return (
        K,
        S,
        ST,
        T,
        call_payoff_mc,
        call_price_mc,
        disc,
        kappa,
        mean_ST_mc,
        mean_ST_theory,
        mean_vT,
        paths,
        r,
        rho,
        se,
        sigma,
        steps,
        theta,
        v,
        vT,
        v0,
        variance_negative_share,
    )


@app.cell
def _(bs_call_price, call_price_mc, mean_ST_mc, mean_ST_theory, mean_vT, mo, se, theta, variance_negative_share, sqrt):
    # BS price using sqrt(theta) as reference
    from math import sqrt as _sqrt
    call_price_bs_theta = bs_call_price(
        mean_ST_theory,  # forward price
        mean_ST_theory,  # ATM strike
        0,  # r=0 since we're using forward
        0,  # q=0
        _sqrt(theta),
        1.0  # normalized time
    )

    mo.md(
        f"""
    ### Results

    **Forward/Mean Check:**
    - Mean S_T (MC): {mean_ST_mc:.4f}
    - Mean S_T (Theory): {mean_ST_theory:.4f}

    **Variance Process:**
    - Share of v_t < 0: {variance_negative_share:.6f} (should be 0 with FT)
    - Mean v_T: {mean_vT:.6f} (θ = {theta:.4f})

    **Call Option Pricing:**
    - Call price (MC): {call_price_mc:.4f} ± {1.96*se:.4f} (95% CI)
    - BS price @√θ: {call_price_bs_theta:.4f} (reference only)
    """
    )
    return (call_price_bs_theta,)


@app.cell
def _(mo):
    mo.md(
        """
    ### Visualizations

    Below are visualizations of the Heston Monte Carlo simulation results.
    """
    )
    return


@app.cell
def _(K, S, ST, T, call_payoff_mc, np, steps, v):
    import matplotlib.pyplot as plt

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Sample variance paths
    n_paths_to_plot = 100
    time_grid = np.linspace(0, T, steps + 1)

    ax1 = axes[0, 0]
    for i in range(n_paths_to_plot):
        ax1.plot(time_grid, v[i, :], alpha=0.3, linewidth=0.5, color='coral')
    ax1.axhline(y=v[0, 0], color='green', linestyle='--', linewidth=2, label=f'v0={v[0, 0]:.4f}')
    ax1.set_xlabel('Time (years)')
    ax1.set_ylabel('Variance')
    ax1.set_title(f'Sample Variance Paths (n={n_paths_to_plot})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Sample price paths
    ax2 = axes[0, 1]
    for i in range(n_paths_to_plot):
        ax2.plot(time_grid, S[i, :], alpha=0.3, linewidth=0.5, color='steelblue')
    ax2.axhline(y=K, color='red', linestyle='--', linewidth=2, label=f'Strike K={K}')
    ax2.set_xlabel('Time (years)')
    ax2.set_ylabel('Price')
    ax2.set_title(f'Sample Price Paths (n={n_paths_to_plot})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Distribution of final prices
    ax3 = axes[1, 0]
    ax3.hist(ST, bins=100, alpha=0.7, color='steelblue', edgecolor='black', density=True)
    ax3.axvline(x=ST.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean={ST.mean():.2f}')
    ax3.axvline(x=K, color='red', linestyle='--', linewidth=2, label=f'Strike={K}')
    ax3.set_xlabel('Final Price S_T')
    ax3.set_ylabel('Density')
    ax3.set_title('Distribution of Final Prices')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Call option payoff distribution
    ax4 = axes[1, 1]
    ax4.hist(call_payoff_mc, bins=100, alpha=0.7, color='green', edgecolor='black', density=True)
    ax4.axvline(x=call_payoff_mc.mean(), color='darkgreen', linestyle='--', linewidth=2,
                label=f'Mean Payoff={call_payoff_mc.mean():.2f}')
    ax4.set_xlabel('Call Payoff')
    ax4.set_ylabel('Density')
    ax4.set_title('Call Option Payoff Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.gca()
    return ax1, ax2, ax3, ax4, fig, n_paths_to_plot, plt, time_grid


if __name__ == "__main__":
    app.run()
