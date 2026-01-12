from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple, List
import math
import statistics


# -----------------------------
# Utilities
# -----------------------------

def sigmoid(x: float) -> float:
    # numerically stable-ish for typical ranges; clamp if you expect huge x
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def median(xs: Iterable[float]) -> float:
    xs = list(xs)
    if not xs:
        raise ValueError("median() requires at least 1 value")
    return statistics.median(xs)


def mad(xs: Iterable[float], center: Optional[float] = None) -> float:
    """
    Median absolute deviation: median(|x - median(x)|)
    Robust scale estimate; does not assume Gaussian distribution.
    """
    xs = list(xs)
    if not xs:
        raise ValueError("mad() requires at least 1 value")
    c = center if center is not None else statistics.median(xs)
    deviations = [abs(x - c) for x in xs]
    return statistics.median(deviations)


def robust_z(x: float, med: float, mad_val: float, eps: float = 1e-8) -> float:
    return (x - med) / (mad_val + eps)


def safe_exp_neg(x: float) -> float:
    # exp(-x) with basic clamp
    if x > 700:
        return 0.0
    return math.exp(-x)


# -----------------------------
# Data Structures
# -----------------------------

@dataclass(frozen=True)
class MetricBaseline:
    """
    Baseline stats for a metric's event-size distribution.
    Provide either (median, mad) directly or raw_samples to compute them.
    """
    median: float
    mad: float


@dataclass
class ScoringConfig:
    # Robust z-score behavior
    z_max: float = 6.0
    eps: float = 1e-8

    # Magnitude aggregation
    metric_weights: Dict[str, float] = field(default_factory=dict)  # default: equal weights
    magnitude_squash: str = "sigmoid"  # future: "tanh", etc.

    # Volatility / structure modifier
    use_structure_modifier: bool = True
    structure_lambda: float = 0.7  # how strongly to discount high volatility ratios
    volatility_aggregate: str = "median"  # "median" or "mean"

    # Coherence/novelty defaults if not provided
    default_coherence: float = 1.0
    default_novelty: float = 1.0


@dataclass
class ScoreBreakdown:
    score: float
    magnitude: float
    structure_modifier: float
    magnitude_eff: float
    coherence: float
    novelty: float

    metric_sizes: Dict[str, float]
    metric_z: Dict[str, float]

    # Optional volatility diagnostics
    volatility_event: Dict[str, float] = field(default_factory=dict)
    volatility_baseline: Dict[str, float] = field(default_factory=dict)
    volatility_ratio: Dict[str, float] = field(default_factory=dict)
    volatility_ratio_agg: Optional[float] = None


# -----------------------------
# Core Scoring
# -----------------------------

def compute_magnitude(
    metric_sizes: Dict[str, float],
    baselines: Dict[str, MetricBaseline],
    cfg: ScoringConfig,
) -> Tuple[float, Dict[str, float]]:
    """
    Magnitude in (0,1), plus per-metric robust z values.
    - metric_sizes: {metric_name: m_i}
    - baselines: {metric_name: MetricBaseline(median, mad)}
    """
    if not metric_sizes:
        raise ValueError("compute_magnitude: metric_sizes is empty")

    # Determine weights: equal by default over metrics present
    weights: Dict[str, float] = {}
    for k in metric_sizes.keys():
        weights[k] = cfg.metric_weights.get(k, 1.0)

    # Normalize weights (optional but nice; keeps scale stable when metric count changes)
    w_sum = sum(weights.values())
    if w_sum <= 0:
        raise ValueError("compute_magnitude: sum of weights must be > 0")
    for k in weights:
        weights[k] /= w_sum

    z_vals: Dict[str, float] = {}
    raw = 0.0

    for name, m_i in metric_sizes.items():
        if name not in baselines:
            raise KeyError(f"Missing baseline for metric '{name}'")
        b = baselines[name]
        z = robust_z(m_i, b.median, b.mad, eps=cfg.eps)
        z = clip(z, 0.0, cfg.z_max)
        z_vals[name] = z
        raw += weights[name] * z

    if cfg.magnitude_squash == "sigmoid":
        mag = sigmoid(raw)
    else:
        # default fallback
        mag = sigmoid(raw)

    return mag, z_vals


def aggregate(values: List[float], method: str) -> float:
    if not values:
        raise ValueError("aggregate: no values")
    if method == "median":
        return median(values)
    if method == "mean":
        return sum(values) / len(values)
    raise ValueError(f"Unknown aggregate method: {method}")


def compute_structure_modifier(
    volatility_event: Dict[str, float],
    volatility_baseline: Dict[str, float],
    cfg: ScoringConfig,
) -> Tuple[float, Dict[str, float], float]:
    """
    Computes an exponential discount based on volatility ratio R:
    R_i = V_event_i / (V_baseline_i + eps)
    R = aggregate_i(R_i)
    structure_modifier = exp(-lambda * max(0, R - 1))

    Returns:
      (structure_modifier, per_metric_ratios, R_agg)
    """
    ratios: Dict[str, float] = {}
    ratio_list: List[float] = []

    # Only compare metrics where both event and baseline volatility exist
    common = set(volatility_event.keys()).intersection(volatility_baseline.keys())
    if not common:
        # If no volatility inputs, no discount
        return 1.0, {}, 1.0

    for name in common:
        v_e = volatility_event[name]
        v_b = volatility_baseline[name]
        r = v_e / (v_b + cfg.eps)
        ratios[name] = r
        ratio_list.append(r)

    R = aggregate(ratio_list, cfg.volatility_aggregate)
    excess = max(0.0, R - 1.0)
    modifier = safe_exp_neg(cfg.structure_lambda * excess)
    modifier = clip(modifier, 0.0, 1.0)
    return modifier, ratios, R


def compute_event_score(
    metric_sizes: Dict[str, float],
    baselines: Dict[str, MetricBaseline],
    cfg: ScoringConfig,
    *,
    coherence: Optional[float] = None,
    novelty: Optional[float] = None,
    volatility_event: Optional[Dict[str, float]] = None,
    volatility_baseline: Optional[Dict[str, float]] = None,
) -> ScoreBreakdown:
    """
    Compute Score = Magnitude_eff × Coherence × Novelty

    Inputs:
      - metric_sizes: per-metric event sizes (m_i)
      - baselines: per-metric MetricBaseline(median, mad) for m_i
      - coherence: scalar [0,1] (seed/layer/marker composite)
      - novelty: scalar [0,1] (intra-run/prototype novelty composite)
      - volatility_event: per-metric volatility in event window (e.g., median(|Δmetric|))
      - volatility_baseline: per-metric baseline volatility (same measure, baseline window)

    Returns a detailed breakdown for logging and tuning.
    """
    mag, z_vals = compute_magnitude(metric_sizes, baselines, cfg)

    coh = cfg.default_coherence if coherence is None else coherence
    nov = cfg.default_novelty if novelty is None else novelty
    coh = clip(coh, 0.0, 1.0)
    nov = clip(nov, 0.0, 1.0)

    structure_modifier = 1.0
    ratios: Dict[str, float] = {}
    R_agg: Optional[float] = None

    vol_e = volatility_event or {}
    vol_b = volatility_baseline or {}

    if cfg.use_structure_modifier and vol_e and vol_b:
        structure_modifier, ratios, R = compute_structure_modifier(vol_e, vol_b, cfg)
        R_agg = R

    mag_eff = clip(mag * structure_modifier, 0.0, 1.0)
    score = clip(mag_eff * coh * nov, 0.0, 1.0)

    return ScoreBreakdown(
        score=score,
        magnitude=mag,
        structure_modifier=structure_modifier,
        magnitude_eff=mag_eff,
        coherence=coh,
        novelty=nov,
        metric_sizes=dict(metric_sizes),
        metric_z=z_vals,
        volatility_event=dict(vol_e),
        volatility_baseline=dict(vol_b),
        volatility_ratio=ratios,
        volatility_ratio_agg=R_agg,
    )


# -----------------------------
# Baseline builders (optional helpers)
# -----------------------------

def build_baselines_from_samples(
    samples: Dict[str, List[float]],
    *,
    eps: float = 1e-8
) -> Dict[str, MetricBaseline]:
    """
    Build per-metric baseline (median, MAD) from raw samples of event sizes.
    samples: {metric_name: [m_i_sample1, m_i_sample2, ...]}
    """
    baselines: Dict[str, MetricBaseline] = {}
    for name, xs in samples.items():
        if not xs:
            raise ValueError(f"No samples for metric '{name}'")
        med = median(xs)
        m = mad(xs, center=med)
        # If MAD is 0, we'll still store it; robust_z uses eps to avoid div0.
        baselines[name] = MetricBaseline(median=med, mad=m)
    return baselines


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Example: you have an event with metric sizes (already computed from window)
    metric_sizes = {
        "delta_effective_rank": 2.4,
        "delta_subspace_angle": 0.18,
        "delta_drift_norm": 0.75,
        "delta_top_singular_ratio": 0.11,
    }

    # Baseline stats computed offline (or via build_baselines_from_samples)
    baselines = {
        "delta_effective_rank": MetricBaseline(median=0.6, mad=0.25),
        "delta_subspace_angle": MetricBaseline(median=0.05, mad=0.02),
        "delta_drift_norm": MetricBaseline(median=0.20, mad=0.10),
        "delta_top_singular_ratio": MetricBaseline(median=0.03, mad=0.015),
    }

    # Optional volatility context (same metric names, but volatility measures)
    volatility_event = {
        "delta_effective_rank": 0.90,
        "delta_subspace_angle": 0.12,
        "delta_drift_norm": 0.40,
        "delta_top_singular_ratio": 0.09,
    }
    volatility_baseline = {
        "delta_effective_rank": 0.60,
        "delta_subspace_angle": 0.08,
        "delta_drift_norm": 0.25,
        "delta_top_singular_ratio": 0.06,
    }

    cfg = ScoringConfig(
        z_max=6.0,
        structure_lambda=0.7,
        use_structure_modifier=True,
        metric_weights={},  # equal weights
    )

    breakdown = compute_event_score(
        metric_sizes=metric_sizes,
        baselines=baselines,
        cfg=cfg,
        coherence=0.8,  # plug in from your cross-seed/layer matcher
        novelty=0.6,    # plug in from novelty calc / prototype distance
        volatility_event=volatility_event,
        volatility_baseline=volatility_baseline,
    )

    print(breakdown)
