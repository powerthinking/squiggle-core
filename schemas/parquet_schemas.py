"""
PyArrow schemas for Squiggle Matching Parquet datasets.

Usage:
  from parquet_schemas import SCHEMAS, schema_for
  schema = schema_for("geometry_state")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pyarrow as pa


# -------------------------
# Helpers / shared types
# -------------------------

T_STRING = pa.string()
T_I16 = pa.int16()
T_I32 = pa.int32()
T_I64 = pa.int64()
T_F32 = pa.float32()
T_F64 = pa.float64()
T_TS_US = pa.timestamp("us")

LIST_I32 = pa.list_(T_I32)
LIST_I64 = pa.list_(T_I64)
LIST_F16 = pa.list_(pa.float16())
LIST_F32 = pa.list_(T_F32)
LIST_STR = pa.list_(T_STRING)

# A canonical "vector" type (choose float32 unless you are sure)
VECTOR_F32 = LIST_F32


def _field(name: str, typ: pa.DataType, nullable: bool = False) -> pa.Field:
    return pa.field(name, typ, nullable=nullable)


@dataclass(frozen=True)
class DatasetSpec:
    """
    DatasetSpec holds schema + recommended partition columns (informational).
    """
    name: str
    schema: pa.Schema
    partitions: tuple[str, ...] = ()


# -------------------------
# Schemas
# -------------------------

RUNS = pa.schema(
    [
        _field("run_id", T_STRING),
        _field("created_at", T_TS_US),
        _field("git_commit", T_STRING),
        _field("code_version", T_STRING),
        _field("hostname", T_STRING),
        _field("user", T_STRING),
        _field("notes", T_STRING, nullable=True),
        _field("model_name", T_STRING),
        _field("model_family", T_STRING),
        _field("vocab_size", T_I32),
        _field("n_layers", T_I16),
        _field("d_model", T_I32),
        _field("n_heads", T_I16),
        _field("d_head", T_I16),
        _field("d_mlp", T_I32),
        _field("precision", T_STRING),
        _field("optimizer", T_STRING),
        _field("seed", T_I32),
        _field("dataset_name", T_STRING),
        _field("tokenizer_name", T_STRING),
        _field("config_json", T_STRING),
    ]
)

CHECKPOINTS = pa.schema(
    [
        _field("run_id", T_STRING),
        _field("step", T_I64),
        _field("wall_time", T_TS_US),
        _field("epoch", T_F32),
        _field("tokens_seen", T_I64),
        _field("lr", T_F32),
        _field("loss_train", T_F32, nullable=True),
        _field("loss_val", T_F32, nullable=True),
        _field("ckpt_path", T_STRING),
        _field("weights_hash", T_STRING, nullable=True),
    ]
)

SAMPLES = pa.schema(
    [
        _field("sample_id", T_STRING),
        _field("sample_set", T_STRING),
        _field("split", T_STRING),
        _field("task", T_STRING),
        _field("family_id", T_STRING, nullable=True),
        _field("difficulty", T_I16, nullable=True),
        _field("prompt_text", T_STRING, nullable=True),
        _field("target_text", T_STRING, nullable=True),
        _field("prompt_tokens", LIST_I32, nullable=True),
        _field("target_tokens", LIST_I32, nullable=True),
        _field("metadata_json", T_STRING, nullable=True),
    ]
)

METRICS_SCALAR = pa.schema(
    [
        _field("run_id", T_STRING),
        _field("step", T_I64),
        _field("wall_time", T_TS_US),
        _field("metric_name", T_STRING),
        _field("value", T_F64),
    ]
)

GEOMETRY_STATE = pa.schema(
    [
        _field("run_id", T_STRING),
        _field("step", T_I64),
        _field("sample_id", T_STRING),
        _field("sample_set", T_STRING),
        _field("split", T_STRING),
        _field("layer", T_I16),
        _field("module", T_STRING),
        _field("d", T_I32),
        _field("l2_norm", T_F32),
        _field("mean", T_F32),
        _field("var", T_F32),
        _field("sparsity_l0_frac", T_F32, nullable=True),
        _field("kurtosis", T_F32, nullable=True),
        _field("skew", T_F32, nullable=True),
        _field("eff_rank", T_F32),
        _field("sv_entropy", T_F32),
        _field("top_sv_frac_1", T_F32, nullable=True),
        _field("top_sv_frac_4", T_F32, nullable=True),
        _field("top_sv_frac_16", T_F32, nullable=True),
        _field("participation_ratio", T_F32, nullable=True),
        _field("subspace_id", T_STRING, nullable=True),
        _field("proj_norm_topk", T_F32, nullable=True),
        _field("spectrum_bins", LIST_F32, nullable=True),
    ]
)

GEOMETRY_DYNAMICS = pa.schema(
    [
        _field("run_id", T_STRING),
        _field("step", T_I64),
        _field("step_prev", T_I64),
        _field("sample_id", T_STRING),
        _field("sample_set", T_STRING),
        _field("split", T_STRING),
        _field("layer", T_I16),
        _field("module", T_STRING),
        _field("dt_steps", T_I32),
        _field("delta_l2_norm", T_F32, nullable=True),
        _field("delta_eff_rank", T_F32, nullable=True),
        _field("delta_sv_entropy", T_F32, nullable=True),
        _field("drift_cosine", T_F32, nullable=True),
        _field("drift_l2", T_F32, nullable=True),
        _field("subspace_principal_angle_mean", T_F32, nullable=True),
        _field("subspace_principal_angle_max", T_F32, nullable=True),
        _field("velocity", T_F32, nullable=True),
        _field("acceleration", T_F32, nullable=True),
        _field("volatility", T_F32, nullable=True),
    ]
)

EVENTS = pa.schema(
    [
        _field("run_id", T_STRING),
        _field("event_id", T_STRING),
        _field("step", T_I64),
        _field("step_start", T_I64, nullable=True),
        _field("step_end", T_I64, nullable=True),
        _field("sample_id", T_STRING, nullable=True),
        _field("sample_set", T_STRING),
        _field("layer", T_I16, nullable=True),
        _field("module", T_STRING, nullable=True),
        _field("event_type", T_STRING),
        _field("score", T_F32),
        _field("threshold", T_F32, nullable=True),
        _field("detector_version", T_STRING, nullable=True),
        _field("payload_json", T_STRING, nullable=True),
    ]
)

SIGNATURES = pa.schema(
    [
        _field("run_id", T_STRING),
        _field("signature_id", T_STRING),
        _field("sample_id", T_STRING, nullable=True),
        _field("sample_set", T_STRING),
        _field("split", T_STRING),
        _field("layer", T_I16, nullable=True),
        _field("module", T_STRING, nullable=True),
        _field("step_start", T_I64),
        _field("step_end", T_I64),
        _field("resolution", T_I16),
        _field("features", LIST_STR),
        _field("signature_vec", VECTOR_F32),
        _field("event_seq", LIST_STR, nullable=True),
        _field("hash", T_STRING, nullable=True),
        _field("version", T_STRING),
    ]
)

MATCHES = pa.schema(
    [
        _field("match_id", T_STRING),
        _field("run_id_a", T_STRING),
        _field("signature_id_a", T_STRING),
        _field("run_id_b", T_STRING),
        _field("signature_id_b", T_STRING),
        _field("match_score", T_F32),
        _field("method", T_STRING),
        _field("alignment_cost", T_F32, nullable=True),
        _field("time_warp_path", LIST_I32, nullable=True),
        _field("notes", T_STRING, nullable=True),
    ]
)

# Optional heavy datasets
EMBEDDINGS = pa.schema(
    [
        _field("run_id", T_STRING),
        _field("step", T_I64),
        _field("sample_id", T_STRING),
        _field("sample_set", T_STRING),
        _field("split", T_STRING),
        _field("layer", T_I16),
        _field("module", T_STRING),
        _field("token_idx", T_I16, nullable=True),
        _field("d", T_I32),
        # choose float32 by default; swap to float16/int8 if you add quantization
        _field("vec", VECTOR_F32),
        _field("quant", T_STRING, nullable=True),
    ]
)

ATTENTION_SUMMARY = pa.schema(
    [
        _field("run_id", T_STRING),
        _field("step", T_I64),
        _field("sample_id", T_STRING),
        _field("sample_set", T_STRING),
        _field("split", T_STRING),
        _field("layer", T_I16),
        _field("head", T_I16),
        _field("attn_entropy", T_F32, nullable=True),
        _field("top1_mass", T_F32, nullable=True),
        _field("topk_mass_5", T_F32, nullable=True),
        _field("diagonal_mass", T_F32, nullable=True),
        _field("band_mass_w2", T_F32, nullable=True),
        _field("band_mass_w8", T_F32, nullable=True),
    ]
)


# -------------------------
# Registry
# -------------------------

SCHEMAS: Dict[str, DatasetSpec] = {
    "runs": DatasetSpec("runs", RUNS, partitions=()),
    "checkpoints": DatasetSpec("checkpoints", CHECKPOINTS, partitions=("run_id",)),
    "samples": DatasetSpec("samples", SAMPLES, partitions=("sample_set", "split")),
    "metrics_scalar": DatasetSpec("metrics_scalar", METRICS_SCALAR, partitions=("run_id", "metric_name")),
    "geometry_state": DatasetSpec("geometry_state", GEOMETRY_STATE, partitions=("run_id", "sample_set", "module")),
    "geometry_dynamics": DatasetSpec("geometry_dynamics", GEOMETRY_DYNAMICS, partitions=("run_id", "sample_set", "module")),
    "events": DatasetSpec("events", EVENTS, partitions=("run_id", "event_type")),
    "signatures": DatasetSpec("signatures", SIGNATURES, partitions=("run_id", "sample_set")),
    "matches": DatasetSpec("matches", MATCHES, partitions=("method",)),
    # optional
    "embeddings": DatasetSpec("embeddings", EMBEDDINGS, partitions=("run_id", "module", "sample_set")),
    "attention_summary": DatasetSpec("attention_summary", ATTENTION_SUMMARY, partitions=("run_id", "sample_set", "layer")),
}


def schema_for(name: str) -> pa.Schema:
    """
    Get a schema by dataset name.

    Example:
      schema = schema_for("geometry_state")
    """
    if name not in SCHEMAS:
        raise KeyError(f"Unknown dataset '{name}'. Known: {sorted(SCHEMAS.keys())}")
    return SCHEMAS[name].schema


def partitions_for(name: str) -> tuple[str, ...]:
    """
    Informational: recommended partition columns.
    """
    if name not in SCHEMAS:
        raise KeyError(f"Unknown dataset '{name}'. Known: {sorted(SCHEMAS.keys())}")
    return SCHEMAS[name].partitions
