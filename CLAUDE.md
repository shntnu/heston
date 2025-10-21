# CLAUDE.md

AI assistant guidance for working with this codebase.

## Project Type

Quantitative finance: stochastic volatility modeling (Black-Scholes, Heston) via Monte Carlo simulation.

## Tech Stack

- **Python 3.12+** (managed with `uv`)
- **Marimo** (0.17.0+) - interactive computational notebooks
- **NumPy** (2.3.4+) - numerical computation
- **SciPy** - statistical distributions

## File Map

- `bs.py` - Black-Scholes interactive notebook (Marimo)
- `heston.py` - Heston stochastic volatility notebook (Marimo)
- `main.py` - Placeholder entry point
- `README.md` - User-facing overview, usage, progress tracker
- `THEORY.md` - Comprehensive theory, discretization schemes, implementation recipes
- `CLAUDE.md` - This file (AI assistant conventions)

## Key Patterns

### Simulation Structure

All Monte Carlo simulations follow this pattern:

1. **Log-space price evolution**: `S_next = S * exp(drift + diffusion * Z)` (guarantees positivity)
2. **Vectorized NumPy**: Simulate all paths simultaneously `(n_paths, n_steps+1)` arrays
3. **Seeded RNG**: Always `np.random.default_rng(seed)` for reproducibility
4. **Risk-neutral drift**: Use `(r - q)` for pricing, not historical measure

### Validation Requirements

Every model implementation must include:

1. **Forward mean check**: MC mean vs theory `E[S_T] = S0 * exp((r-q)*T)`
2. **Analytical comparison**: MC price vs closed-form when available
3. **Statistical rigor**: Report standard errors and 95% confidence intervals
4. **Positivity**: For stochastic vol, ensure variance stays non-negative (truncation/exact schemes)

### Marimo Notebook Conventions

- Each `@app.cell` defines dependencies via function arguments
- Interactive sliders for all model parameters
- Reactive execution on parameter changes
- Include visualizations: paths, distributions, payoffs

## Implementation Details

**Heston specifics:**

- Euler Full Truncation (FT) scheme for variance process
- Correlated shocks: `Z_S = ρ*Z_v + √(1-ρ²)*Z_perp`
- Clamp variance before and after stepping: `v_plus = max(v, 0)`
- See `THEORY.md` for detailed equations and discretization schemes

**Adding new models:**

- Create `simulate_*_paths()` returning `(n_paths, n_steps+1)` arrays
- Follow validation pattern above
- Use log-space for price processes to avoid negativity

## Critical Conventions

- Never hardcode parameters - use sliders in notebooks
- Always work in log-space for asset prices
- Use exact GBM stepping (not Euler) for Black-Scholes
- For Heston: clamp variance, use correlated normals
- Report MC standard errors alongside point estimates

## References

- `README.md` - Quick start and usage
- `THEORY.md` - Detailed theory, math, implementation guidance
