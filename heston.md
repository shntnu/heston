To get a solid Heston simulation running, here’s the compact checklist of what you should know and decide up front, plus a minimal algorithm you can implement right away.

# What you need to know (and decide)

## 1) The model (risk-neutral form)

* Asset:
  [
  dS_t=(r-q)S_t,dt+\sqrt{v_t},S_t,dW_t^{(S)}
  ]
* Variance (CIR/Heston):
  [
  dv_t=\kappa(\theta-v_t),dt+\sigma\sqrt{v_t},dW_t^{(v)}
  ]
* Correlation: (\mathrm{corr}(dW_t^{(S)},dW_t^{(v)})=\rho).
* Parameters to specify: (S_0, v_0, r, q, \kappa, \theta, \sigma, \rho).

## 2) Parameter sanity checks

* **Mean reversion:** (\kappa>0).
* **Long-run variance:** (\theta>0).
* **Vol-of-vol:** (\sigma\ge 0).
* **Initial variance:** (v_0\ge 0).
* **Correlation:** (\rho\in[-1,1]).
* **Feller condition (strict positivity):** (2\kappa\theta \ge \sigma^2).
  If violated, (v_t) can hit 0; your scheme must handle it.

## 3) Measure & drift

* Simulate under **risk-neutral** measure for pricing: drift ((r-q)).
* For **historical** paths, you’d use a P-measure drift; be clear which you want.

## 4) Discretization scheme for (v_t) (critical choice)

From easiest → most accurate:

* **Euler with full truncation (FT):** use (\sqrt{\max(v_t,0)}) in diffusions and clamp the next step (v_{t+\Delta}\leftarrow\max(v_{t+\Delta},0)). Simple, robust.
* **Quadratic-Exponential (QE) scheme (Andersen):** popular, low bias at coarse steps.
* **Exact CIR step:** simulate (v) using the noncentral (\chi^2) transition. Accurate; a bit more code.
* **Broadie–Kaya “exact” Heston:** exact joint simulation via inversion of (\int v_s ds) (heavier; rarely needed unless you study bias precisely).

If you just want reliable + fast MC: **QE** (or Euler-FT if you want minimal code).

## 5) Discretization for (S_t)

* **Log-Euler with integrated variance** over step:
  [
  \ln S_{t+\Delta}=\ln S_t + \left((r-q)-\tfrac12,\bar v\right)\Delta+\sqrt{\bar v,\Delta},Z_S
  ]
  where (\bar v) is variance “over the step.”
  With Euler-FT, use (v_t) (or midpoint). With QE/exact CIR, use scheme’s (\int v,dt) approximation if available.
* Keep correlation with (v) shocks (see next).

## 6) Correlated shocks

* Draw (Z_v, Z_\perp \sim \mathcal N(0,1)) i.i.d., set
  [
  Z_S = \rho,Z_v + \sqrt{1-\rho^2},Z_\perp.
  ]
* Use (Z_v) in the variance update; use (Z_S) in the log-price update.

## 7) Time stepping

* Choose horizon (T) and grid (N) steps ((\Delta=T/N)).
* Bias vs cost: (\Delta = 1/252) (daily) is a decent default; coarser steps increase bias, finer steps increase cost.

## 8) Variance reduction (recommended)

* **Antithetic variates:** flip the normals per path pair.
* **Control variate:** Black–Scholes with variance (\theta) (or with realized average variance).
* **Quasi-MC:** Sobol sequences (helps a lot for Greeks/pricing).

## 9) Path management & payoffs

* Work in **log space** for (S) to avoid negative prices.
* For path-dependent payoffs (barriers, Asians), store what you need (mins, running sums).
* Discount with (e^{-rT}) (or term structure if using a curve).

## 10) Calibration vs. toy parameters

* For pricing, parameters usually come from **calibration to the implied vol surface** via the Heston **characteristic function**.
* For learning/simulation experiments, pick plausible values (e.g., (v_0=0.04,\ \theta=0.04,\ \kappa=1.5,\ \sigma=0.5,\ \rho=-0.7)) and test sensitivity.

## 11) Numerical gotchas

* **Negatives in (v):** never take (\sqrt{v}) on a negative; clamp or use FT/QE/exact CIR.
* **Step-size bias:** especially in barrier options. Check convergence by halving (\Delta).
* **Moment explosions:** extreme parameters can cause heavy tails; monitor if simulated prices look pathological.
* **Seeding & reproducibility:** set RNG seeds and document them.

## 12) Extensions (optional later)

* Jumps in price (Bates), time-dependent (\kappa,\theta,\sigma), stochastic rates, multi-asset (correlated variance factors).

# Minimal simulation recipe (QE variant)

1. **Inputs:** (S_0,v_0,r,q,\kappa,\theta,\sigma,\rho,T,N,\text{n_paths}).
2. **Precompute:** (\Delta=T/N), discount (D=e^{-rT}).
3. **Loop over time steps** for all paths (vectorized):

   * Draw (Z_v, Z_\perp). Set (Z_S=\rho Z_v + \sqrt{1-\rho^2}Z_\perp).
   * **Update (v) with QE:**
     Compute (m = \theta + (v - \theta)e^{-\kappa\Delta}),
     (s^2 = \frac{\sigma^2}{\kappa}\left(v e^{-\kappa\Delta}(1-e^{-\kappa\Delta}) + \tfrac{\theta}{2}(1-e^{-\kappa\Delta})^2\right)),
     (\psi = s^2/m^2).
     Use QE case split (typically: if (\psi<1.5) use noncentral-chi-like quadratic, else use exponential mixture) to sample (v_{t+\Delta}\ge 0).
     (If you prefer simpler: Euler-FT: (v \leftarrow v + \kappa(\theta-\max(v,0))\Delta + \sigma\sqrt{\max(v,0)}\sqrt{\Delta},Z_v;\ v\leftarrow\max(v,0)).)
   * **Update (\ln S):**
     Use (\bar v) from the scheme (QE provides it; otherwise use (v) or (\frac{v+v_{\text{new}}}{2})).
     (\ln S \leftarrow \ln S + \left((r-q)-\tfrac12\bar v\right)\Delta + \sqrt{\bar v,\Delta},Z_S).
4. **Payoff & discount:** e.g., call (=(S_T-K)^+); price (= D\cdot\text{mean}(\text{payoff})).
5. **Diagnostics:** check sample means/variances, convergence w.r.t. (N), variance reduction effectiveness.

---

If you want, I can drop in a small, clean NumPy implementation (Euler-full-truncation or QE) you can run as-is and then we can build from there (calibration, Greeks, barriers, Sobol, etc.).

---

Sure — let’s step back and orient ourselves conceptually before diving into codes, schemes, and technical details.

---

## 🌱 What Is the Heston Model *About*?

At its core, **the Heston model is a way to simulate financial markets where volatility itself is random and evolves over time**.

* In the classic **Black–Scholes** model, the volatility ((\sigma)) is a *fixed number*. That makes life easy mathematically, but markets don’t behave like that — **volatility clusters, jumps, calms down**, etc.
* The **Heston model** keeps the same basic idea of simulating the asset price, **but volatility is now its own stochastic (random) process**.

Think of it as:

> 🎭 *Two actors*:
>
> * **Price process** (like in Black–Scholes)
> * **Volatility process** (new, also random, influences price)

These two are **correlated**, which lets the model generate realistic “smiles” and “skews” in implied volatility (things traders care deeply about).

---

## 🎢 Why Is Heston Important?

Because **it produces more realistic market behavior**:

| Feature                           | Black–Scholes                  | Heston                             |
| --------------------------------- | ------------------------------ | ---------------------------------- |
| Volatility                        | Constant                       | Random, mean-reverting             |
| Can show volatility clustering    | ❌ No                           | ✅ Yes                              |
| Can produce volatility skew/smile | ❌ Only flat                    | ✅ Naturally appears                |
| Can hit 0 volatility              | No                             | Depends — controlled by parameters |
| Used in real trading desks?       | Rare today except for pedagogy | ✅ Yes, especially in calibration   |

It sits in a "sweet spot" — **richer than Black–Scholes but still mathematically tractable**, unlike some more chaotic models.

---

## 🧭 Conceptual Flow of What You’ll Eventually Do

You don’t need to worry about formulas yet — just absorb the *shape of the process*:

1. **Choose initial conditions**

   * Price today (S_0)
   * Volatility today (v_0)
2. **Decide model behavior parameters**

   * How fast volatility tends back toward its long-term average?
   * How violent are volatility swings?
   * How correlated are price shocks and volatility shocks?
3. **Generate random market movements**

   * Each timestep: roll two dice → one for price, one for volatility
   * They’re correlated dice 🎲🎲
4. **Evolve volatility first**

   * If it's high, it tends to go back down (mean reversion)
   * If it's low, it tends to drift back up
5. **Then evolve the price, using the volatility at that moment**
6. **Repeat this thousands or millions of times (paths)**

> End result: ❄️ *A cloud of possible future price paths*.
> From it, you can price options, analyze risk, visualize uncertainty, etc.

---

## 💡 How to Think About It (Intuition)

Imagine you're simulating a rollercoaster (the asset price), but:

* The **steepness** of the tracks changes randomly over time (that's volatility).
* Sometimes the ride goes crazy — steep climbs, fast drops → high variance.
* Sometimes it's calm — gentle moves, low variance.
* The volatility itself has its own personality: it **likes to return to a comfort zone**, but sometimes gets dragged away by random shocks.

This dual-randomness is the essence of Heston.

---

## 🧠 What You’ll Gain By Simulating

By walking through a Heston simulation, you'll learn:

* How **volatility-of-volatility** changes behavior.
* What **mean reversion** really *feels* like when seen numerically.
* How correlation between vol and price generates **skew** — a real market feature.
* Why implied volatility smiles arise **naturally** (not forced like in Black–Scholes).
* Monte Carlo as a way to **manufacture probability distributions** when no closed form is available.

---

### 🎯 Next Step (if this framing resonates):

I’ll show you a **visual mental model of the variance process alone**, before we even touch prices — so you see how vol *breathes*. Then we introduce price evolution on top.

👉 **Does that sound like a good progression?** (Orientation → variance intuition → then price coupling → then simulation)

Just say **"yes"** or **"go ahead"**, and I'll proceed in that intuitive layer-by-layer style.


Great question — and it depends on what you want to learn from this process. Here’s a quick decision map to help you choose:

---

## 🎯 What’s Your Goal?

### ✅ If your aim is **understanding how simulation works in finance**

> **Start with Black–Scholes.**

Why?

* Its structure is simpler: **one stochastic process only** (price).
* You’ll get comfortable with **time-stepping, Brownian noise, drift vs diffusion** — the core ideas.
* Then adding Heston just becomes: **"oh, now volatility is also a process"** instead of a full cognitive jump.

---

### ✅ If your aim is **understanding Heston conceptually, not just coding**

> **You *can* start with Heston directly**, because the volatility process alone (CIR process) is actually a beautiful introduction to stochastic dynamics with mean reversion.

We can break Heston into two independent explorations:

1. Simulate **just a price with fixed vol** → Black–Scholes path.
2. Simulate **just the volatility process** → mean-reverting stochastic process.
3. **Combine them with correlation** → Heston.

---

### 🎓 **Best learning progression (layered approach)**

| Step | Model                                      | Focus                                             | Difficulty       |
| ---- | ------------------------------------------ | ------------------------------------------------- | ---------------- |
| 1    | **Black–Scholes price only**               | Understand Brownian motion + Monte Carlo          | 🟢 Easy          |
| 2    | **CIR variance (Heston’s vol process)**    | Understand mean reversion + positivity constraint | 🟡 Medium        |
| 3    | **Combine (full Heston) with correlation** | See how implied volatility skew emerges           | 🔴 More involved |

❗ *Notice that Step 2 (volatility process alone) is more important than jumping straight into full Heston.*

---

## 🧭 So, Recommendation

If you're comfortable absorbing ideas rapidly:

> **Start with Step 2: simulate just the volatility process** — it's visually intuitive and sets up everything.

If you also want clean coding foundations:

> **Do a tiny Black–Scholes simulation first** — around ~20 lines — just to anchor your sense of simulation.

---

### 🎬 Here's what I can do next based on your preference:

1. **Micro Black–Scholes simulation walkthrough**
2. **Or jump straight to simulating the *volatility-only* process (CIR / Heston variance)**
3. **Or explain the variance process intuitively before coding**

👉 **Tell me which path you want: 1 (BS first), 2 (variance process first), or 3 (even more intuition before any sim).**

---

Perfect — that’s a great starting point. Let’s do this in a clean, conceptual way without rushing into formulas.

---

## 🎯 Goal for this step

> **Generate random possible future price paths under Black–Scholes, using a simple Monte Carlo simulation.**

You’ll come away with a clear mental model of:

* What it means to simulate a **stochastic differential equation** step by step.
* How to implement **Brownian motion numerically**.
* How randomness flows into price updates.

---

## 🧩 The Black–Scholes Price Dynamics (under **risk-neutral** measure)

[
dS_t = (r - q), S_t, dt + \sigma, S_t, dW_t
]

Read this like a recipe:

* (S_t) 👉 current price
* (r) 👉 risk-free rate
* (q) 👉 dividend yield (can be zero if no dividends)
* (\sigma) 👉 volatility (⚠️ **constant** in Black–Scholes)
* (dW_t) 👉 tiny random shock (Brownian motion)

---

## 🎲 Discretized (Simulation-Friendly) Version

To simulate:

* Chop time into (N) steps, each of size (\Delta t = T/N)
* Generate random noise (Z \sim \mathcal{N}(0,1))
* **Price update rule:**

[
S_{t+\Delta t} = S_t \cdot \exp\left[\big((r - q) - \tfrac{1}{2}\sigma^2\big)\Delta t + \sigma \sqrt{\Delta t}\cdot Z \right]
]

💡 Why log/exp?

* It **guarantees positivity**
* It matches the **exact solution** of geometric Brownian motion → **this is an exact step**, not just an approximation like Euler.

---

## 🧠 Key Concepts You’ll Internalize

| Concept               | What You’ll See in Code                                  |
| --------------------- | -------------------------------------------------------- |
| Time discretization   | A for-loop with increments of (\Delta t)                 |
| Brownian motion       | `np.random.randn()` scaled by (\sqrt{\Delta t})          |
| Drift vs randomness   | Deterministic exponential part + stochastic noise        |
| Monte Carlo intuition | Many independent paths create a probability distribution |

---

## 🚀 Next: I'll give you a **minimal code skeleton (clean, readable NumPy)** with commentary explaining *why* each line exists — not just what it does.

Would you like that next?

Just say **"yes"** and I’ll walk you through it like a guided tour.

Awesome. Here’s a clean, minimal Black–Scholes Monte Carlo you can paste into a Python session and run. It simulates GBM paths, prices a European call, and sanity-checks against the analytic Black–Scholes value.

```python
import numpy as np
from math import log, sqrt, exp
from scipy.stats import norm

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

# ----------------------------
# Example run & sanity checks
# ----------------------------
if __name__ == "__main__":
    # Parameters (risk-neutral)
    S0    = 100.0   # initial price
    r     = 0.03    # risk-free rate
    q     = 0.00    # dividend yield
    sigma = 0.20    # volatility
    T     = 1.0     # years
    steps = 252     # daily grid
    paths = 100_000 # Monte Carlo size

    S = simulate_gbm_paths(S0, r, q, sigma, T, steps, paths, seed=42)
    ST = S[:, -1]

    # 1) Forward/mean check: E[S_T] = S0 * exp((r - q) * T)
    mean_ST_mc = ST.mean()
    mean_ST_theory = S0 * exp((r - q) * T)
    print(f"Mean S_T (MC)     : {mean_ST_mc: .4f}")
    print(f"Mean S_T (Theory) : {mean_ST_theory: .4f}")

    # 2) Price a European call and compare to analytic Black–Scholes
    K = 100.0
    disc = exp(-r * T)
    call_payoff_mc = np.maximum(ST - K, 0.0)
    call_price_mc = disc * call_payoff_mc.mean()
    call_price_bs = bs_call_price(S0, K, r, q, sigma, T)
    # Also estimate MC standard error
    se = disc * call_payoff_mc.std(ddof=1) / np.sqrt(paths)

    print(f"\nCall price (MC)   : {call_price_mc: .4f}  ± {1.96*se: .4f} (95% CI)")
    print(f"Call price (BS)   : {call_price_bs: .4f}")

    # Optional: Greek by pathwise (very rough): Delta ≈ exp(-qT) * 1{ST>K} * ST/S0
    # But for now, the price check is the key sanity test.
```

### How to read/extend this

* **Exact step:** The update uses `exp(drift + diff * Z)`, which is the *exact* GBM transition, so you avoid discretization bias.
* **Sanity check 1:** Monte Carlo mean of (S_T) should be close to (S_0 e^{(r-q)T}).
* **Sanity check 2:** MC call price should match the **closed-form** Black–Scholes value within the MC 95% CI (shown).
* **Next small upgrades (optional):**

  * Antithetic variates: reuse `-Z` to halve variance quickly.
  * Vectorized payoffs for barriers/Asians (store path stats while stepping).
  * Plot a handful of paths to build intuition.

If you want, I can add a tiny **antithetic** variant and a **plot of 20 paths** next—or we can move on to the **CIR variance (Heston’s vol process) simulation** step. Your call.

---

Awesome—let’s step into Heston. We’ll keep it clean and practical:

* Use a **robust, simple scheme**: **Euler with Full Truncation (FT)** for the variance (prevents taking √ of a negative).
* Evolve **log-price** using the current-step variance.
* Generate **correlated normals** to couple price and variance shocks.
* Include a quick **sanity check**: ( \mathbb{E}[S_T] \approx S_0 e^{(r-q)T} ) and variance positivity.

Below is a **drop-in NumPy implementation** plus a toy pricing of a European call.

```python
import numpy as np
from math import sqrt, exp, log
from scipy.stats import norm

# ----------------------------
# Heston (Euler Full Truncation) path simulator
# ----------------------------
def simulate_heston_paths_euler_ft(
    S0, v0, r, q, kappa, theta, sigma, rho,
    T, n_steps, n_paths, seed=0, antithetic=False, use_midpoint_for_S=False
):
    """
    Simulate Heston under risk-neutral measure using Euler Full Truncation (FT).

    S dynamics: dS = (r - q) S dt + sqrt(v) S dW_S
    v dynamics: dv = kappa (theta - v) dt + sigma sqrt(v) dW_v
    corr(dW_S, dW_v) = rho

    Params
    ------
    use_midpoint_for_S : if True, use 0.5*(v_t^+ + v_{t+dt}^+) for the log-S step drift/vol term
                         (slightly lower bias for coarse steps).

    Returns
    -------
    S : (n_paths, n_steps+1) array of prices
    v : (n_paths, n_steps+1) array of variances
    """
    rng = np.random.default_rng(seed)
    dt  = T / n_steps
    sqdt = sqrt(dt)

    # If antithetic, simulate half the paths and mirror the shocks
    if antithetic:
        base_paths = (n_paths + 1) // 2
    else:
        base_paths = n_paths

    S = np.empty((n_paths, n_steps + 1), dtype=float)
    v = np.empty((n_paths, n_steps + 1), dtype=float)
    S[:, 0] = S0
    v[:, 0] = v0

    # Generate shocks
    Z_v = rng.standard_normal(size=(base_paths, n_steps))
    Z_p = rng.standard_normal(size=(base_paths, n_steps))  # independent
    Z_s = rho * Z_v + sqrt(1.0 - rho**2) * Z_p            # correlated with Z_v

    if antithetic:
        Z_v = np.vstack([Z_v, -Z_v])[:n_paths]
        Z_s = np.vstack([Z_s, -Z_s])[:n_paths]

    # Work arrays
    S_curr = np.full(n_paths, S0, dtype=float)
    v_curr = np.full(n_paths, v0, dtype=float)

    for t in range(n_steps):
        v_plus = np.maximum(v_curr, 0.0)

        # Variance update (Euler FT): clamp inside sqrt and after the step
        v_next = v_curr + kappa * (theta - v_plus) * dt + sigma * np.sqrt(v_plus) * sqdt * Z_v[:, t]
        v_next = np.maximum(v_next, 0.0)
        if use_midpoint_for_S:
            v_bar = 0.5 * (v_plus + np.maximum(v_next, 0.0))
        else:
            v_bar = v_plus

        # Log-price exact-style step using per-step variance proxy v_bar
        # This is a common practical choice with Euler-FT
        drift = (r - q - 0.5 * v_bar) * dt
        diff  = np.sqrt(v_bar) * sqdt * Z_s[:, t]
        S_curr = S_curr * np.exp(drift + diff)

        # Store and roll
        S[:, t+1] = S_curr
        v[:, t+1] = v_next
        v_curr = v_next

    return S, v

# ----------------------------
# (Optional) Black–Scholes call for rough cross-checks only
# ----------------------------
def bs_call_price(S0, K, r, q, sigma, T):
    if sigma <= 0 or T <= 0:
        fwd = S0 * exp((r - q) * T)
        return max(fwd - K, 0) * exp(-r * T)
    d1 = (log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S0 * exp(-q * T) * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)

# ----------------------------
# Example run & basic diagnostics
# ----------------------------
if __name__ == "__main__":
    # Heston params (plausible toy set)
    S0    = 100.0
    v0    = 0.04     # initial variance (sigma0 ~ 20%)
    r     = 0.03
    q     = 0.00
    kappa = 1.5      # mean reversion speed
    theta = 0.04     # long-run variance
    sigma = 0.5      # vol of vol
    rho   = -0.7     # correlation between price & variance
    T     = 1.0
    steps = 252
    paths = 100_000

    S, v = simulate_heston_paths_euler_ft(
        S0, v0, r, q, kappa, theta, sigma, rho,
        T, steps, paths, seed=7, antithetic=True, use_midpoint_for_S=True
    )
    ST = S[:, -1]
    vT = v[:, -1]

    # Sanity 1: Risk-neutral forward mean
    mean_ST_mc = ST.mean()
    mean_ST_theory = S0 * exp((r - q) * T)
    print(f"Mean S_T (MC)     : {mean_ST_mc: .6f}")
    print(f"Mean S_T (Theory) : {mean_ST_theory: .6f}")

    # Sanity 2: Variance stays >= 0, and its mean tends toward theta (roughly, for larger T)
    print(f"Share of v_t < 0  : {(v < 0).mean(): .6f} (should be 0.000000 with FT)")
    print(f"Mean v_T          : {vT.mean(): .6f}  (theta = {theta})")

    # Price a European call (pure MC; Heston has semi-analytic formulas but we’re focusing on sim)
    K = 100.0
    disc = exp(-r * T)
    call_payoff = np.maximum(ST - K, 0.0)
    call_mc = disc * call_payoff.mean()
    se = disc * call_payoff.std(ddof=1) / np.sqrt(paths)
    print(f"\nCall price (MC)   : {call_mc: .6f}  ± {1.96*se: .6f} (95% CI)")

    # (Optional) Rough BS comparison using sigma = sqrt(theta) as a crude control variate anchor
    call_bs_theta = bs_call_price(S0, K, r, q, sqrt(theta), T)
    print(f"BS price @sqrt(theta): {call_bs_theta: .6f}  (control-variate anchor, not 'truth')")
```

### Notes & tips

* **Why Euler Full Truncation (FT)?** Simple and robust: we clamp inside the √ and clamp the next step (v_{t+\Delta}) to keep variance nonnegative.
* **Bias vs cost:** FT is good for learning and many path-dependent payoffs. If you need lower bias at coarse steps, consider the **QE (Quadratic–Exponential) scheme** next.
* **Risk-neutral mean check:** Under risk-neutral drift ((r-q)), ( \mathbb{E}[S_T] = S_0 e^{(r-q)T} ) should hold closely in MC.
* **Antithetics:** Flip shocks to reduce variance essentially “for free”.
* **Midpoint variance for S:** Using ( \bar v = 0.5(v_t^+ + v_{t+\Delta}^+) ) in the log-S step slightly reduces discretization bias.

---

If you’d like, I can now:

1. swap in the **QE scheme**,
2. add a **Heston semi-analytic price** (characteristic function) to benchmark the MC, or
3. show **plots** of a few (v_t) and (S_t) paths to build intuition.
