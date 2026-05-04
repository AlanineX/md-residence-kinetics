# Two ways to compute residence time — a methods note

We have two analyses that both report a "residence time" but **measure
different things**. They are complementary, not redundant. This note
walks through each method **stepwise on real data** (EDDA chain F,
500 ns, 707 binding events) so the difference is concrete.

---

## TL;DR

| | **Population SP(τ) fit** | **Dwell-time MLE** |
|---|---|---|
| Input | binary contact matrix → SP(τ) curve | binary contact matrix → list of event durations |
| What's a "data point" | a value of SP at lag τ | one binding event |
| Weighting | **time-weighted** (samples t₀ uniformly in time) | **event-weighted** (each event counted once) |
| Output | apparent τ_off (and bi-exp components) | k_off, k_on, K_d per component |
| Bias | toward **long** events (they occupy more time) | none, but flickers dominate by count |

Both views are correct — they answer different questions. Cross-check:
`Σ(dwell_durations) / T_total ≈ ⟨n⟩` (steady-state population), satisfied
to 4 sig figs in our data.

---

## Worked example — EDDA chain F, 500 ns

Numbers are from `results_v3/T1_per_chain_pocket_EDDA/sp_chain_F.csv` and
`results_v3/dwell_kinetics/`.

### Shared input — the contact matrix

For each frame *i* (5001 frames, dt = 0.1 ns), record which ADP molecules
are within 3.5 Å of the 25 chain-F pocket residues.

Stored sparse in `_intermediate/contacts/chain_F.npz` as CSR:
- `frame_indices[i]` = absolute frame number
- `resindices[offsets[i]:offsets[i+1]]` = ADP resids in contact at frame i

Chain F sees **6 unique ADP molecules** across 500 ns; a single
molecule may bind, leave, and rebind many times, giving 707 events total.

---

## Method 1 — Population autocorrelation SP(τ)

### Step 1: build origin-grid sets

For each origin frame t₀ in `[0, 1.0 ns, 2.0 ns, …, T_total]` (501
origins, t0_step = 1 ns), let *n₀* = number of ADPs bound at t₀.

### Step 2: compute survival fraction at each lag

For each lag τ ∈ {0, 0.1, 0.2, …, 50.0 ns}, count how many of those
*n₀* molecules are **still continuously bound** at frame t₀+τ
(intermittency = 0):

  SP_origin(τ; t₀) = |alive at t₀ AND at t₀+τ| / n₀

(Implementation: `phase_b_worker.compute_origin_sp` —
`alive &= list_of_sets[t₀+τ]`, walking τ forward.)

### Step 3: average across origins

  SP(τ) = ⟨ SP_origin(τ; t₀) ⟩ over all t₀ where *n₀* > 0.

The result `sp_chain_F.csv` starts at SP(0) = 1.0 and decays toward a
plateau *c* (the long-residence / "immobilized" fraction).

### Step 4: fit a bi-exponential decay

Constrained model (`fit_bi_exp` in `fitting.py`):

  P(t) = α₁·exp(-t/τ₁) + α₂·exp(-t/τ₂) + c,
  where α₁ = u(1-c), α₂ = (1-u)(1-c)

Fit by **least-squares on SP(τ)** with `scipy.optimize.curve_fit`.

For chain F EDDA the fit gives (from `bi_exp_fitting_results.csv`):

```
α₁ = 0.20, τ₁ = 0.84 ns           (fast component)
α₂ = 0.33, τ₂ = 19.86 ns          (slow component)
c  = 0.46                          (long-residence plateau)
fitted_res_time = α₁·τ₁ + α₂·τ₂ = 7.81 ns
```

### Why this is **time-weighted**

Each origin t₀ is uniformly distributed in time. If a 10-ns event covers
10 consecutive origins, it is **counted 10 times**. A 0.3-ns flicker is
counted ≤ 1 time. So a single long event contributes as many SP-data
points as ~30 flickers of equal-time-share.

→ The fitted τ_off reflects "**given that you find an ADP bound at a
random time, how long until it leaves?**". This is the right answer for
comparing to MS hold-up times (MS averages over time, not events).

---

## Method 2 — Dwell-time MLE

### Step 1: per-molecule occupancy series

For each ADP resid that ever touched chain F, build a binary array
`occ[resid]` of length 5001. `occ[resid][i] = 1` iff resid was bound at
frame *i*.

### Step 2: extract events (runs of 1s)

A "binding event" = a maximal consecutive run of `occ = 1`. For
intermittency = 0, no merging. Result: 707 events for chain F.

First few events (resid 4132):
```
event 1: bound from t=  3.9 ns → t=  4.4 ns,  dwell = 0.50 ns
event 2: bound from t=  5.0 ns → t=  5.4 ns,  dwell = 0.40 ns
event 3: bound from t=  5.6 ns → t=  6.2 ns,  dwell = 0.60 ns
event 4: bound from t=  6.3 ns → t=  6.8 ns,  dwell = 0.50 ns
event 5: bound from t=  7.0 ns → t=  7.1 ns,  dwell = 0.10 ns
…
```

Categorise:
- **Uncensored (full)**: event ends inside the trajectory. Chain F: **704**.
- **Right-censored**: still bound at t = 500 ns. Chain F: **3**.
- **Left-censored**: already bound at t = 0 ns. Chain F: **0**.

Σ(uncensored dwells) = **857.20 ns**;
Σ(censored dwells) = 13.00 ns;
mean dwell (uncensored only) = **1.218 ns**.

### Step 3: count arrivals (for k_on)

An *arrival* = a 0→1 transition (excluding events that started
left-censored, since they have no observed start). For chain F:
**N_arrivals = 707**.

### Step 4: maximum-likelihood fit

The likelihood for a bi-exponential mixture with right-censoring:

  log L(π, τ₁, τ₂) =  Σ_full log f(tᵢ; π, τ₁, τ₂)
                     + Σ_cens log S(tⱼ; π, τ₁, τ₂)

where
  f(t) = π/τ₁·exp(-t/τ₁) + (1-π)/τ₂·exp(-t/τ₂)
  S(t) = π·exp(-t/τ₁) + (1-π)·exp(-t/τ₂)

Maximise log L (equivalently, minimise –log L) with `L-BFGS-B` from
several random starts. SEs from the inverse Hessian of –log L at the
maximum (observed Fisher information).

For chain F EDDA:

```
π       = 0.914 ± 0.013     (fast-component amplitude)
τ_fast  = 0.246 ± 0.013 ns
τ_slow  = 11.92 ± 1.79 ns

k_off_fast = 1/τ_fast = 4.06 ns⁻¹  = 4.06 × 10⁹ s⁻¹
k_off_slow = 1/τ_slow = 0.084 ns⁻¹ = 8.4 × 10⁷ s⁻¹
```

### Step 5: closed-form k_on from arrival counting

  k_on = (N_arrivals / T_total) / [bulk]   (units: M⁻¹·ns⁻¹)

For chain F EDDA:

```
[ADP]_bulk = 300 / (V × N_A) = 300 / (19.1³ nm³ × 6.022×10²³) = 71.49 mM
k_on(F)    = (707 / 500.0 ns) / 0.07149 M
           = 1.414 arrivals/ns / 0.0715 M
           = 19.78 M⁻¹·ns⁻¹  = 1.98 × 10¹⁰ M⁻¹·s⁻¹
```

### Step 6: K_d = k_off / k_on

```
K_d_slow = 0.0839 / 19.78 = 4.24 × 10⁻³ M = 4.24 mM
K_d_fast = 4.06   / 19.78 = 0.205 M       = 205 mM
```

The slow-component K_d is the physically meaningful affinity (the fast
component is contact-flicker noise from the 3.5 Å cutoff).

### Why this is **event-weighted**

Each event contributes one likelihood term, regardless of duration. A
0.3-ns flicker counts the same as a 10-ns binder. So the dwell-time
distribution gives the **per-event** kinetic rate constants k_off,
not the time-weighted residence.

→ The right answer for "**what is the rate constant of dissociation**?"
in mass-action kinetics, and the only way to compute k_on (autocorrelation
can't see arrivals).

---

## Cross-validation

Both methods must agree on the integrated quantity ⟨n⟩, which is
model-free:

```
⟨n_F⟩ = (Σ uncensored dwells + Σ censored dwells) / T_total
      = (857.20 + 13.00) / 500.00
      = 1.7404
```

vs. **direct time-average** of the binary occupancy: 1.7401 ✓

vs. T1 `per_chain_summary.csv` reports: 1.7400 ✓

If this consistency check fails, either the events were extracted
incorrectly or the contact matrix was mis-summed.

---

## Side-by-side numerical comparison (chain F EDDA)

| Quantity | Population SP fit | Dwell-time MLE |
|---|---|---|
| Slow τ ± SE | 19.86 ns | **11.92 ± 1.79 ns** |
| Fast τ ± SE | 0.84 ns  | **0.25 ± 0.01 ns** |
| Slow amplitude | α₂ = 0.33 | (1-π) = 0.086 |
| Plateau term *c* | 0.46 | none (mixture sums to 1) |
| Apparent τ_off | 7.81 ns (= α₁τ₁+α₂τ₂) | 1.30 ns (= ⟨τ⟩_event) |
| k_off,slow | — | 8.4 × 10⁷ s⁻¹ |
| k_on | — | 1.98 × 10¹⁰ M⁻¹s⁻¹ |
| K_d (slow) | — | 4.24 mM |
| ⟨n⟩ steady state | 1.74 | 1.74 |

Numbers differ by design — the population fit's τ_off is a
time-weighted residence (≈ 6 × the event-weighted mean), while MLE
returns true rate constants.

---

## When to use which

**Population fit τ_off** when reporting:
- A single residence time for comparison with mass-spec hold-up time
- Behaviour averaged over the trajectory ("if I look in at random,
  what do I find?")
- Plots of SP(τ) decay, model-free metrics RMST(t*), S(t*)

**Dwell MLE** when reporting:
- Per-event rate constants k_on, k_off in s⁻¹
- Dissociation constant K_d in M / mM
- Distribution of binding-event durations (histogram + fit)
- Direct comparison of buffer effects on **affinity** (vs.
  population, which conflates affinity with accessibility)

For the GroEL §3 paper:
- §3.4 (ADP residence): keep T1 SP(τ) fits (matches MS framing)
- §3.4-supplement (kinetics decomposition): add dwell MLE table with
  k_on / k_off / K_d to make the affinity claim falsifiable

---

## Files & code

```
scripts_kinetics/
  kinetics/phase_b_worker.py    SP(τ) per-origin computation
  fitting.py                    SP-curve bi-exp curve_fit (Method 1)
  dwell_time_kinetics.py        per-event MLE + k_on (Method 2)
results_v3/
  T1_per_chain_pocket_<BUF>/    Method 1 outputs (SP curves + fits)
  dwell_kinetics/               Method 2 outputs
    summary_<BUF>.txt
    bi_exp_fitting_results.csv
    single_exp_fitting_results.csv
    kon_koff_summary.csv
    plots/dwell_<BUF>.{svg,png}
```
