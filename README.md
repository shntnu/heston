# Heston

Monte Carlo simulations for Black-Scholes and Heston stochastic volatility models.

## Run

```bash
marimo edit bs.py       # Black-Scholes
marimo edit heston.py   # Heston stochastic volatility
```

## What's here

- `bs.py` - Interactive Black-Scholes simulation with sliders
- `heston.py` - Interactive Heston model with stochastic variance
- `THEORY.md` - Comprehensive theory, discretization schemes, implementation recipes
- `CLAUDE.md` - AI assistant conventions

## Progress

### ✅ Completed

- **Black-Scholes Monte Carlo** (`bs.py`)
  - Exact log-space GBM simulation
  - Interactive Marimo notebook with sliders
  - Validation: forward mean check, analytic BS pricing comparison
  - Visualizations: paths, final distribution, payoff distribution

- **Heston Model** (`heston.py`)
  - Euler Full Truncation (FT) scheme
  - Correlated shocks (price ↔ variance)
  - 10 interactive parameters (S0, K, r, q, T, v0, κ, θ, σ, ρ)
  - Validation: forward mean, variance positivity, MC standard errors
  - Visualizations: variance paths, price paths, distributions

### 🔄 Next Steps (Optional Enhancements)

**Heston Improvements:**

- [ ] QE (Quadratic-Exponential) scheme - lower bias at coarse steps
- [ ] Exact CIR variance - noncentral χ² transition
- [ ] Heston semi-analytic pricing - characteristic function benchmark
- [ ] Antithetic variates - variance reduction

**Analysis & Calibration:**

- [ ] Greeks computation - Delta, Gamma, Vega via finite differences or pathwise
- [ ] Implied volatility surface - generate and compare BS vs Heston
- [ ] Parameter calibration - fit to market option prices
- [ ] Convergence analysis - bias vs step size

**Extensions:**

- [ ] Barrier options - path-dependent payoffs
- [ ] Asian options - average price payoffs
- [ ] Quasi-Monte Carlo - Sobol sequences for variance reduction
- [ ] Multi-asset Heston - correlated variance factors
- [ ] Bates model - Heston + jumps
