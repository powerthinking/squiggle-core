import pyarrow as pa


UTC_TS = pa.timestamp("us", tz="UTC")

# ----------------------------
# 12 — probe_captures_index (facts)
# ----------------------------

probe_captures_index_schema = pa.schema([
    ("run_id", pa.string()),
    ("probe_name", pa.string()),
    ("probe_config_hash", pa.string()),
    ("schema_version", pa.string()),

    ("capture_type", pa.string()),        # e.g. activations_sketch, attn_summary
    ("step", pa.int64()),                 # nullable if shard spans steps
    ("shard_id", pa.string()),
    ("path", pa.string()),
    ("bytes", pa.int64()),
    ("checksum", pa.string()),            # nullable allowed by storing empty or using nulls
    ("created_at_utc", UTC_TS),
])

# ----------------------------
# 13 — probe_summaries (interpretation)
# ----------------------------

probe_summaries_schema = pa.schema([
    # spine
    ("run_id", pa.string()),
    ("analysis_id", pa.string()),
    ("schema_version", pa.string()),
    ("created_at_utc", UTC_TS),

    # identity
    ("family_id", pa.string()),
    ("seed", pa.int32()),
    ("probe_name", pa.string()),
    ("probe_config_hash", pa.string()),
    ("model_id", pa.string()),
    ("base_checkpoint", pa.string()),
    ("run_mode", pa.string()),
    ("steps", pa.int32()),
    ("capture_steps_used", pa.list_(pa.int64())),

    # traceability
    ("captures_manifest_path", pa.string()),
    ("captures_index_path", pa.string()),

    # alignment base
    ("layers_covered", pa.list_(pa.int16())),

    # Layer A (float32 arrays aligned to layers_covered)
    ("A_eff_rank_pre", pa.list_(pa.float32())),
    ("A_eff_rank_post", pa.list_(pa.float32())),
    ("A_eff_rank_delta", pa.list_(pa.float32())),

    ("A_sv_entropy_pre", pa.list_(pa.float32())),
    ("A_sv_entropy_post", pa.list_(pa.float32())),
    ("A_sv_entropy_delta", pa.list_(pa.float32())),

    ("A_sparsity_pre", pa.list_(pa.float32())),
    ("A_sparsity_post", pa.list_(pa.float32())),
    ("A_sparsity_delta", pa.list_(pa.float32())),

    ("A_principal_angle_post_vs_pre", pa.list_(pa.float32())),

    # Layer B (float32 arrays aligned to layers_covered)
    ("B_drift_velocity_by_layer", pa.list_(pa.float32())),
    ("B_drift_accel_by_layer", pa.list_(pa.float32())),
    ("B_volatility_by_layer", pa.list_(pa.float32())),
    ("B_alignment_velocity_by_layer", pa.list_(pa.float32())),

    # Layer B globals
    ("B_drift_velocity_global", pa.float32()),
    ("B_drift_accel_global", pa.float32()),
    ("B_volatility_global", pa.float32()),
    ("B_alignment_velocity_global", pa.float32()),

    # affected layers (subset; not necessarily same length as layers_covered)
    ("affected_layers", pa.list_(pa.int16())),

    # signature + ranking
    ("signature_version", pa.string()),
    ("signature_vector", pa.list_(pa.float32())),
    ("signature_norm", pa.float32()),

    ("score_version", pa.string()),
    ("magnitude", pa.float32()),
    ("coherence", pa.float32()),
    ("novelty", pa.float32()),
    ("DIS", pa.float32()),
])

# ----------------------------
# 14 — probe_events_candidates (interpretation)
# ----------------------------

probe_events_candidates_schema = pa.schema([
    ("run_id", pa.string()),
    ("analysis_id", pa.string()),
    ("schema_version", pa.string()),
    ("created_at_utc", UTC_TS),

    ("family_id", pa.string()),
    ("seed", pa.int32()),
    ("probe_name", pa.string()),
    ("probe_config_hash", pa.string()),

    ("detector_version", pa.string()),
    ("timewarp_tolerance_steps", pa.int32()),

    ("event_id", pa.string()),
    ("event_type", pa.string()),
    ("t_step", pa.int64()),
    ("layers", pa.list_(pa.int16())),
    ("strength", pa.float32()),

    # supporting signals as parallel lists
    ("supporting_key", pa.list_(pa.string())),
    ("supporting_val", pa.list_(pa.float32())),
])
