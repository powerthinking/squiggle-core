from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


RunMode = Literal["probe_micro_finetune", "train_full", "eval_only"]


class SquiggleModel(BaseModel):
    """Strict base model: forbid unknown fields to prevent silent schema drift."""
    model_config = ConfigDict(extra="forbid")


# ----------------------------
# Identity / Capture Reference
# ----------------------------

class ProbeIdentity(SquiggleModel):
    family_id: str
    model_id: str
    base_checkpoint: str
    run_mode: RunMode = "probe_micro_finetune"
    steps: int = Field(ge=1)
    seed: int
    dtype: Optional[str] = None
    device: Optional[str] = None
    timestamp_utc: datetime

    @field_validator("timestamp_utc")
    @classmethod
    def _tz_utc(cls, v: datetime) -> datetime:
        # Require timezone-aware UTC
        if v.tzinfo is None:
            raise ValueError("timestamp_utc must be timezone-aware (UTC)")
        # normalize to UTC
        return v.astimezone(timezone.utc)


class CaptureNotes(SquiggleModel):
    dropped_steps: List[int] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class ProbeCaptureRef(SquiggleModel):
    run_id: str
    probe_name: str
    probe_config_hash: str
    captures_manifest_path: str
    captures_index_path: str
    capture_steps_used: List[int] = Field(default_factory=list)
    notes: CaptureNotes = Field(default_factory=CaptureNotes)

    @field_validator("capture_steps_used")
    @classmethod
    def _sorted_unique_steps(cls, v: List[int]) -> List[int]:
        # deterministic + compact
        return sorted(set(v))


# ----------------------------
# Layer A (State)
# ----------------------------

class LayerAMetricSeries(SquiggleModel):
    pre: List[float]
    post: List[float]
    delta: List[float]


class LayerAStateMetrics(SquiggleModel):
    effective_rank: LayerAMetricSeries
    sv_entropy: LayerAMetricSeries
    sparsity_proxy: LayerAMetricSeries
    principal_angle_post_vs_pre: List[float]


class LayerAAggregateSummary(SquiggleModel):
    mean_abs: float
    median_abs: float
    p95_abs: float


class LayerAAggregations(SquiggleModel):
    summary_pre: LayerAAggregateSummary
    summary_delta: LayerAAggregateSummary


class LayerAState(SquiggleModel):
    description: str
    layers_covered: List[int]
    metrics: LayerAStateMetrics
    aggregations: LayerAAggregations

    @field_validator("layers_covered")
    @classmethod
    def _sorted_unique_layers(cls, v: List[int]) -> List[int]:
        if not v:
            raise ValueError("layers_covered must be non-empty")
        # preserve user order if they want, but ensure uniqueness
        # We’ll enforce uniqueness without sorting to keep alignment stable.
        seen = set()
        out = []
        for x in v:
            if x in seen:
                raise ValueError("layers_covered must not contain duplicates")
            seen.add(x)
            out.append(x)
        return out

    @model_validator(mode="after")
    def _validate_lengths(self) -> "LayerAState":
        n = len(self.layers_covered)

        def chk(name: str, arr: List[Any]) -> None:
            if len(arr) != n:
                raise ValueError(f"{name} length {len(arr)} != layers_covered length {n}")

        m = self.metrics
        chk("effective_rank.pre", m.effective_rank.pre)
        chk("effective_rank.post", m.effective_rank.post)
        chk("effective_rank.delta", m.effective_rank.delta)

        chk("sv_entropy.pre", m.sv_entropy.pre)
        chk("sv_entropy.post", m.sv_entropy.post)
        chk("sv_entropy.delta", m.sv_entropy.delta)

        chk("sparsity_proxy.pre", m.sparsity_proxy.pre)
        chk("sparsity_proxy.post", m.sparsity_proxy.post)
        chk("sparsity_proxy.delta", m.sparsity_proxy.delta)

        chk("principal_angle_post_vs_pre", m.principal_angle_post_vs_pre)
        return self


# ----------------------------
# Layer B (Dynamics)
# ----------------------------

class DynamicsWindowing(SquiggleModel):
    type: Literal["steps", "tokens", "seconds"] = "steps"
    dt: int = Field(ge=1, default=1)
    smoothing: Literal["none", "ema", "ma"] = "ema"
    ema_alpha: Optional[float] = Field(default=0.2, ge=0.0, le=1.0)


class LayerBByLayerAndGlobal(SquiggleModel):
    by_layer: List[float]
    global_: float = Field(alias="global")

    @field_validator("global_")
    @classmethod
    def _finite(cls, v: float) -> float:
        # place for future finiteness checks if desired
        return v


class LayerBDynamicsMetrics(SquiggleModel):
    drift_velocity: LayerBByLayerAndGlobal
    drift_acceleration: LayerBByLayerAndGlobal
    volatility: LayerBByLayerAndGlobal
    alignment_velocity: LayerBByLayerAndGlobal


class AffectedLayers(SquiggleModel):
    method: str
    k: int = Field(ge=1)
    layers: List[int]


class LayerBDynamics(SquiggleModel):
    description: str
    windowing: DynamicsWindowing
    metrics: LayerBDynamicsMetrics
    affected_layers: AffectedLayers

    @model_validator(mode="after")
    def _validate_lengths(self) -> "LayerBDynamics":
        # We require all by_layer arrays align with Layer A's layers_covered;
        # this cross-check is done at ProbeSummary level (see below).
        return self


# ----------------------------
# Layer C (Candidates) - Optional
# ----------------------------

class EventCandidate(SquiggleModel):
    event_id: str
    type: str
    t_step: int = Field(ge=0)
    layers: List[int]
    strength: float
    supporting_signals: Dict[str, float] = Field(default_factory=dict)


class LayerCEventCandidates(SquiggleModel):
    enabled: bool = True
    detector_version: str
    timewarp_tolerance_steps: int = Field(ge=0, default=0)
    candidates: List[EventCandidate] = Field(default_factory=list)
    notes: Optional[str] = None


# ----------------------------
# Signature + Ranking
# ----------------------------

class Signature(SquiggleModel):
    signature_version: str
    construction: Dict[str, Any] = Field(default_factory=dict)
    vector: List[float]
    vector_semantics: List[str] = Field(default_factory=list)
    vector_norm: float = Field(ge=0.0)


class RankingComponents(SquiggleModel):
    magnitude: float
    coherence: float
    novelty: float


class Ranking(SquiggleModel):
    score_version: str
    formula: str = "Magnitude × Coherence × Novelty"
    components: RankingComponents
    total: float
    interpretation: Optional[str] = None


# ----------------------------
# Top-level: Probe Summary
# ----------------------------

class ProbeSummary(SquiggleModel):
    schema_version: Literal["probe_summary@2.0"] = "probe_summary@2.0"

    # Interpretation tracking
    analysis_id: str  # hash(canonical_analysis_spec)

    identity: ProbeIdentity
    capture_ref: ProbeCaptureRef

    layer_A_state: LayerAState
    layer_B_dynamics: LayerBDynamics
    layer_C_event_candidates: Optional[LayerCEventCandidates] = None

    signature: Signature
    ranking: Ranking

    created_at_utc: datetime

    @field_validator("created_at_utc")
    @classmethod
    def _tz_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("created_at_utc must be timezone-aware (UTC)")
        return v.astimezone(timezone.utc)

    @model_validator(mode="after")
    def _cross_validate(self) -> "ProbeSummary":
        # Identity vs capture sanity
        if self.capture_ref.capture_steps_used:
            if max(self.capture_ref.capture_steps_used) > self.identity.steps:
                raise ValueError("capture_steps_used contains step > identity.steps")

        # Enforce Layer B by_layer alignment with Layer A layers_covered
        n = len(self.layer_A_state.layers_covered)
        bm = self.layer_B_dynamics.metrics

        def chk(name: str, arr: List[float]) -> None:
            if len(arr) != n:
                raise ValueError(f"{name} length {len(arr)} != layers_covered length {n}")

        chk("B.drift_velocity.by_layer", bm.drift_velocity.by_layer)
        chk("B.drift_acceleration.by_layer", bm.drift_acceleration.by_layer)
        chk("B.volatility.by_layer", bm.volatility.by_layer)
        chk("B.alignment_velocity.by_layer", bm.alignment_velocity.by_layer)

        # family_id consistency checks (optional but useful)
        if self.identity.family_id != self.identity.family_id:
            raise ValueError("identity.family_id mismatch (unexpected)")

        return self


# ----------------------------
# Mapping helpers (to Parquet rows)
# ----------------------------

def probe_summary_to_probe_summaries_row(ps: ProbeSummary) -> dict:
    """Flatten ProbeSummary into a single row for the probe_summaries Parquet table."""
    A = ps.layer_A_state.metrics
    B = ps.layer_B_dynamics.metrics

    return {
        # spine
        "run_id": ps.capture_ref.run_id,
        "analysis_id": ps.analysis_id,
        "schema_version": ps.schema_version,
        "created_at_utc": ps.created_at_utc,

        # identity
        "family_id": ps.identity.family_id,
        "seed": ps.identity.seed,
        "probe_name": ps.capture_ref.probe_name,
        "probe_config_hash": ps.capture_ref.probe_config_hash,
        "model_id": ps.identity.model_id,
        "base_checkpoint": ps.identity.base_checkpoint,
        "run_mode": ps.identity.run_mode,
        "steps": ps.identity.steps,
        "capture_steps_used": ps.capture_ref.capture_steps_used,

        # traceability
        "captures_manifest_path": ps.capture_ref.captures_manifest_path,
        "captures_index_path": ps.capture_ref.captures_index_path,

        # Layer A
        "layers_covered": ps.layer_A_state.layers_covered,
        "A_eff_rank_pre": A.effective_rank.pre,
        "A_eff_rank_post": A.effective_rank.post,
        "A_eff_rank_delta": A.effective_rank.delta,
        "A_sv_entropy_pre": A.sv_entropy.pre,
        "A_sv_entropy_post": A.sv_entropy.post,
        "A_sv_entropy_delta": A.sv_entropy.delta,
        "A_sparsity_pre": A.sparsity_proxy.pre,
        "A_sparsity_post": A.sparsity_proxy.post,
        "A_sparsity_delta": A.sparsity_proxy.delta,
        "A_principal_angle_post_vs_pre": A.principal_angle_post_vs_pre,

        # Layer B
        "B_drift_velocity_by_layer": B.drift_velocity.by_layer,
        "B_drift_accel_by_layer": B.drift_acceleration.by_layer,
        "B_volatility_by_layer": B.volatility.by_layer,
        "B_alignment_velocity_by_layer": B.alignment_velocity.by_layer,
        "B_drift_velocity_global": B.drift_velocity.global_,
        "B_drift_accel_global": B.drift_acceleration.global_,
        "B_volatility_global": B.volatility.global_,
        "B_alignment_velocity_global": B.alignment_velocity.global_,

        "affected_layers": ps.layer_B_dynamics.affected_layers.layers,

        # signature + ranking
        "signature_version": ps.signature.signature_version,
        "signature_vector": ps.signature.vector,
        "signature_norm": ps.signature.vector_norm,

        "score_version": ps.ranking.score_version,
        "magnitude": ps.ranking.components.magnitude,
        "coherence": ps.ranking.components.coherence,
        "novelty": ps.ranking.components.novelty,
        "DIS": ps.ranking.total,
    }


def probe_summary_to_probe_events_candidate_rows(ps: ProbeSummary) -> List[dict]:
    """Explode optional Layer C candidates into rows for probe_events_candidates table."""
    if ps.layer_C_event_candidates is None or not ps.layer_C_event_candidates.enabled:
        return []

    out: List[dict] = []
    c = ps.layer_C_event_candidates
    for ev in c.candidates:
        # store supporting signals as key/value lists to fit Parquet easily
        keys = list(ev.supporting_signals.keys())
        vals = [float(ev.supporting_signals[k]) for k in keys]
        out.append({
            "run_id": ps.capture_ref.run_id,
            "analysis_id": ps.analysis_id,
            "schema_version": ps.schema_version,
            "created_at_utc": ps.created_at_utc,

            "family_id": ps.identity.family_id,
            "seed": ps.identity.seed,
            "probe_name": ps.capture_ref.probe_name,
            "probe_config_hash": ps.capture_ref.probe_config_hash,

            "detector_version": c.detector_version,
            "timewarp_tolerance_steps": c.timewarp_tolerance_steps,

            "event_id": ev.event_id,
            "event_type": ev.type,
            "t_step": ev.t_step,
            "layers": ev.layers,
            "strength": ev.strength,
            "supporting_key": keys,
            "supporting_val": vals,
        })
    return out
