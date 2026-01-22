# Squiggle Scoring (v0)

This document defines the v0 scoring system for ranking candidate learning events (“squiggle events”).

## Goal

Given a detected event window and its associated metric deltas, compute a score that prioritizes:
- **Magnitude**: the size of the geometric change
- **Coherence**: repeatability/structure (across seeds/layers/markers)
- **Novelty**: dissimilarity from previously-seen events

Core scoring factorization:

**Score = Magnitude × Coherence × Novelty**

Volatility is treated as a **context modifier** (not baked into Magnitude),
because “big change on a stable channel” differs from “big change on a volatile channel.”

---

## Definitions

### Event
An event is a candidate change detected at:
- `run_id`, `seed_id`, `layer_id`, `stream` (embedding/residual/attention/MLP/etc)
- `t_start`, `t_end` (training step indices; inclusive/exclusive is implementation-specific)
- `metrics`: a mapping from metric name → (time series or window summary)

This scoring system assumes each event provides:
- **Event sizes** per metric (e.g., max absolute delta in window)
- **Baseline distributions** per metric (to normalize sizes robustly)

---

## 1) Magnitude

Magnitude captures "How big is the geometric change?"

### Per-metric event size
For each metric `i`, define an event size `m_i`. Typical choices:
- `m_i = max_abs_delta_in_window(metric)`
- `m_i = abs(metric[t_end] - metric[t_start])`
- `m_i = integral(abs(Δmetric))` over window

The default recommended v0 is:

**m_i = max(|Δmetric|) within [t_start, t_end]**

### Robust normalization
Because metrics have different scales and non-Gaussian distributions, we compute a robust z-score:

Let:
- `med_i` = median of baseline `m_i` samples
- `mad_i` = median absolute deviation (MAD) of baseline `m_i` samples

MAD:
`mad_i = median(|x - med_i|)`

Robust z:
`z_i = (m_i - med_i) / (mad_i + eps)`

Clip to prevent dominance:
`z_i = clip(z_i, 0, z_max)`

### Combining metrics
With optional weights `w_i` (default equal weights):

`raw = Σ_i (w_i * z_i)`

Squash to (0,1) for interpretability:

`Magnitude = sigmoid(raw)`

Sigmoid:
`sigmoid(x) = 1 / (1 + exp(-x))`

---

## 2) Volatility Modifier (Structure Context)

Volatility provides context: a large magnitude on a volatile metric is less structurally informative.

We compute a **stability ratio** per metric:

- `V_baseline_i`: baseline volatility of metric `i`
- `V_event_i`: event-window volatility of metric `i`

Default volatility measure:
- `V = median(|Δmetric|)` within the specified window

Stability ratio:
`R_i = (V_event_i) / (V_baseline_i + eps)`

Aggregate ratios across metrics (default median):
`R = median_i(R_i)`

Convert to a discount factor:

`structure_modifier = exp(-lambda * max(0, R - 1))`

So:
- If `R <= 1`, modifier ≈ 1 (structured / not noisier than baseline)
- If `R >> 1`, modifier decreases (spiky / turbulent)

Recommended v0:
- Use this modifier on Magnitude:

`Magnitude_eff = Magnitude * structure_modifier`

---

## 3) Coherence

Coherence captures “Is this event repeatable / structured?”

In v0, coherence is a product (or weighted combination) of available components.

### Components (each in [0,1])

- **Seed coherence**: how consistently this event matches across seeds
- **Layer coherence**: whether it appears in contiguous layers or layer bands
- **Marker/task coherence**: whether it aligns with curriculum markers or repeats across task families
- **Volatility coherence** (optional): penalize highly noisy events

v0 supports any subset; missing components default to 1.

Recommended v0 default:
`Coherence = seed_coherence` (start simple; add others as you build matching)

---

## 4) Novelty

Novelty captures “Is this event unlike what we’ve already seen?”

Two common novelty channels:

- **Intra-run novelty**: distance to previously logged event signatures in the same run
- **Prototype novelty**: distance to known prototypes (collapse/migration/etc)

v0 supports a single scalar novelty in [0,1].

Recommended v0 default:
- novelty provided by your matching system (or set to 1 if not yet implemented)

---

## 5) Final Score

Core:

`Score = Magnitude_eff × Coherence × Novelty`

All terms must be in [0,1].

---

## Required Inputs (v0)

Per event:
- event metric sizes: `{metric_name: m_i}`
- baseline metric distributions for sizes OR precomputed baseline `(med_i, mad_i)`
- optional event and baseline metric volatilities to compute structure modifier
- coherence scalar in [0,1] (seed/layer/marker)
- novelty scalar in [0,1]

---

## Suggested Output Schema

A scored event should log:

- `score`
- `magnitude`
- `structure_modifier`
- `magnitude_eff`
- `coherence`
- `novelty`
- per-metric:
  - `m_i`
  - `z_i`
  - `V_event_i`, `V_baseline_i`, `R_i` (if available)

This supports later tuning and ablation analysis.

---

## Notes

- Do not bake volatility directly into Magnitude; treat it as context.
- Clip z-scores to avoid single-metric dominance.
- Prefer robust statistics (median/MAD) over mean/std for heavy-tailed metrics.
