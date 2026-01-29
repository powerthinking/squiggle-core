"""Squiggle context builder for LLM prompts."""


def build_squiggle_context() -> str:
    """Return canonical Squiggle framework definitions for LLM context.

    This provides the LLM with necessary vocabulary and concepts to understand
    Squiggle analysis reports and provide meaningful interpretations.
    """
    return """
SQUIGGLE FRAMEWORK CONTEXT

A **squiggle** is a compressed, basis-invariant representation of the temporal
evolution of geometric descriptors in a neural representation, capturing state,
dynamics, and events as a matchable signature. It represents how representations
reorganize over time, not their final state.

**Core Question:** "What changed, when did it change, and what caused it?"

## Three-Layer Ontology

### Layer A - Geometric State (Snapshot)
Basis-invariant descriptors at a given time:
- **effective_rank**: Dimensionality utilization (participation ratio of singular values)
- **sv_entropy**: Singular value distribution entropy (higher = more uniform)
- **topk_mass**: Fraction of variance captured by top-k singular values

### Layer B - Dynamics (Change Over Time)
How geometry evolves:
- Velocity (rate of change of descriptors)
- Drift (directional movement)
- Volatility (magnitude of fluctuations)

### Layer C - Events (Discrete Structure)
Compressed symbolic transitions:
- **change_point**: Significant change in a single metric
- **change_point_composite**: Coordinated changes across multiple metrics at same step
- Phase transitions, rank collapse, bifurcation events

## Event Detection System

### Adaptive Thresholding
```
threshold = median(deltas) + k × MAD(deltas)
```
Where k (default 2.5) controls sensitivity. Higher k = fewer events.

### Peak Selection with Suppression
- Candidates (deltas exceeding threshold) are sorted by magnitude
- Top candidate selected, then candidates within suppression_radius are suppressed
- Prevents detecting the same transition multiple times

### Warmup-Aware Budgeting
- Separate budgets for pre-warmup and post-warmup candidates
- Pre-warmup often has large but uninformative transients
- `max_pre_warmup` controls how many early events can be selected
- Pre and post candidates never compete for the same slots

## Retention Metrics

**Candidates:** Deltas exceeding adaptive threshold
**Selected:** Events that survive peak selection
**Suppression skips:** Candidates too close to a stronger candidate
**TopK skips:** Candidates exceeding budget after selection

**Retention ratio** = selected / candidates

High suppression rate indicates clustering; warmup cap indicates noisy early training.

## Seed Invariance

Events appearing consistently across multiple random seeds indicate genuine
learning dynamics rather than random fluctuations.

**Metrics:**
- **Jaccard similarity**: |intersection| / |union| of event keys
- **Trajectory correlation**: Pairwise correlation of geometry trajectories
- **Common events**: Events appearing in all runs (step tolerance allowed)

**Interpretation:**
- Jaccard > 50% = strong invariance
- Jaccard < 25% = weak invariance
- Trajectory correlation > 0.9 = consistent dynamics
- Common events with low step std = temporally aligned learning

## Detection Config Fingerprint

Analysis parameters that affect event detection:
- `adaptive_k`: Threshold multiplier (default 2.5)
- `suppression_radius`: Steps to suppress around each peak (default 15)
- `max_events_per_series`: Budget per (layer, metric) series (default 5)
- `warmup_end_step`: Step where warmup ends
- `max_pre_warmup`: Budget for pre-warmup events (default 0-1)

**Important:** Only compare runs with matching config fingerprints.

## Event Phases

Events are classified by training phase:
- **Shaping phase**: Early period (first 10-20% of training)
- **Transition phase**: Middle period where main learning occurs
- **Locking phase**: Final period where representations stabilize

Boundary events (at phase boundaries) may be artifacts of phase definitions.

## Diversity Metrics

Measure how distributed events are across dimensions:
- Score variance across layers
- Temporal dispersion (std of event steps)
- Metric participation (entropy across metrics)
- Combined diversity score (0-1, higher = more distributed)
""".strip()
