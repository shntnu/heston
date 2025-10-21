# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a quantitative finance project focused on **stochastic volatility modeling**, specifically implementing Black-Scholes and Heston model simulations using Monte Carlo methods. The project uses **Marimo** for interactive notebooks and **NumPy** for numerical computation.

## Key Technologies

- **Python 3.12+** (managed with `uv`)
- **Marimo** (0.17.0+): Interactive computational notebooks
- **NumPy** (2.3.4+): Numerical arrays and computation
- **SciPy**: Statistical distributions (norm, etc.)

## Project Structure

- `bs.py`: Marimo notebook implementing Black-Scholes GBM simulation with interactive sliders for parameters (S0, r, q, σ, T, K). Includes Monte Carlo path generation and analytic pricing comparison.
- `heston.py`: Marimo notebook implementing Heston stochastic volatility model using Euler Full Truncation scheme. Interactive sliders for 10 parameters (S0, K, r, q, T, v0, κ, θ, σ, ρ). Includes correlated price/variance simulation and validation checks.
- `main.py`: Simple entry point (placeholder)
- `heston.md`: Comprehensive theoretical documentation covering:
  - Heston model fundamentals and intuition
  - Parameter requirements and Feller condition
  - Discretization schemes (Euler-FT, QE, exact CIR)
  - Implementation recipes and code examples
  - Black-Scholes vs Heston comparison
- `PROGRESS.md`: Compact implementation tracker showing completed work and optional enhancement roadmap

## Running the Project

### Run the Black-Scholes Marimo notebook

```bash
marimo edit bs.py
```

This launches an interactive environment with sliders to explore how parameters affect:

- Monte Carlo price paths
- Risk-neutral forward mean (E[S_T])
- European call option pricing vs analytic Black-Scholes formula

### Run the Heston Marimo notebook

```bash
marimo edit heston.py
```

This launches an interactive environment with sliders for 10 Heston parameters to explore:

- Stochastic variance paths (mean reversion, vol-of-vol)
- Correlated price paths with time-varying volatility
- Variance positivity validation (Euler Full Truncation)
- European call option pricing with Monte Carlo standard errors

### Run as standalone Python

```bash
python bs.py
python heston.py
```

## Architecture & Design

### Monte Carlo Simulation Pattern

All simulations follow this structure:

1. **Exact log-space evolution** for GBM (Black-Scholes):
   - Uses `np.exp(drift + diffusion * Z)` to guarantee price positivity
   - Vectorized across all paths using NumPy array operations
   - Cumulative sum over time dimension for efficiency

2. **Parameter handling**:
   - S0: initial price
   - r: risk-free rate (drift under risk-neutral measure)
   - q: dividend yield
   - σ: volatility (constant in BS, stochastic in Heston)
   - T: time horizon
   - K: strike price (for option payoffs)

3. **Validation approach**:
   - Compare Monte Carlo mean to theoretical forward: `S0 * exp((r - q) * T)`
   - Compare MC option prices to closed-form Black-Scholes formula
   - Report 95% confidence intervals using standard error

### Code Style in `bs.py`

- **Marimo cell structure**: Each `@app.cell` defines dependencies explicitly via function arguments
- **Reactive execution**: Changing slider values automatically re-runs dependent cells
- **Vectorized NumPy**: All path simulation done in vectorized form (n_paths × n_steps arrays)
- **Exact GBM stepping**: Uses exact solution in log-space, not Euler approximation

## Development Workflow

### Adding new models

When implementing additional stochastic volatility models, follow the established patterns from `bs.py` and `heston.py`:

1. Create a `simulate_*_paths()` function returning (n_paths, n_steps+1) arrays
2. Implement analytical pricing function for validation (if available)
3. Use Marimo sliders for interactive parameter exploration
4. Include both forward mean check and option pricing comparison
5. Report Monte Carlo standard errors for statistical rigor
6. For stochastic volatility: ensure variance/volatility stays non-negative via truncation or exact schemes

### Key implementation details

- **Reproducibility**: Always use seeded RNG (`np.random.default_rng(seed)`)
- **Vectorization**: Simulate all paths simultaneously using NumPy broadcasting
- **Log-space simulation**: For price processes, work in log-space to avoid negative prices
- **Risk-neutral measure**: Use (r - q) drift for pricing, not historical drift

## Heston Model Implementation

The `heston.py` implementation follows the theoretical foundation in `heston.md`:

1. **Variance process** (CIR dynamics): `dv_t = κ(θ - v_t)dt + σ√v_t dW_v`
2. **Price process**: `dS_t = (r - q)S_t dt + √v_t S_t dW_S`
3. **Correlation**: `corr(dW_S, dW_v) = ρ`

**Current scheme**: Euler with Full Truncation (FT)

- Clamps variance to non-negative: `v_plus = max(v, 0)` before sqrt
- Clamps after step: `v_next = max(v_next, 0)` to ensure positivity
- Uses correlated normals: `Z_S = ρ*Z_v + √(1-ρ²)*Z_perp`
- Log-space price evolution for guaranteed positivity

**Parameters** (with default values from `heston.md`):

- `v0 = 0.04` (initial variance, i.e., σ₀ ≈ 20%)
- `κ = 1.5` (mean reversion speed, κ > 0)
- `θ = 0.04` (long-run variance, θ > 0)
- `σ = 0.5` (vol-of-vol)
- `ρ = -0.7` (correlation, ρ ∈ [-1, 1])
- Feller condition check: `2κθ ≥ σ²` (ensures variance stays positive)

**Future enhancements** (see `PROGRESS.md`):

- QE (Quadratic-Exponential) scheme for lower bias
- Exact CIR variance via noncentral χ² transition
- Heston semi-analytic pricing via characteristic function

## Testing & Validation

Always include these sanity checks:

1. **Forward mean check**: Monte Carlo mean vs theoretical `E[S_T] = S0 * exp((r-q)*T)`
2. **Analytical comparison**: MC option price vs closed-form (BS formula or Heston characteristic function)
3. **Statistical rigor**: Report standard errors and confidence intervals
4. **Convergence tests**: Verify results stable as n_paths increases

## Notes

- The project uses `uv` for Python package management
- All Monte Carlo implementations should support fixed seeds for reproducibility
- Interactive notebooks (Marimo) are the primary interface for exploration
- Theoretical foundation documented extensively in `heston.md`
