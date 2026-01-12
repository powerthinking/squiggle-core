from __future__ import annotations

import os
from pathlib import Path


def repo_root_from_here() -> Path:
    """
    Best-effort: squiggle-core/src/squiggle_core/paths.py -> squiggle-core -> (parent) squiggle/
    """
    return Path(__file__).resolve().parents[3]


def data_root() -> Path:
    """
    Returns the root data directory. Defaults to <repo_root>/data unless overridden by
    the SQUIGGLE_DATA_ROOT environment variable.
    """
    env = os.getenv("SQUIGGLE_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return (repo_root_from_here() / "data").resolve()


# --- Canonical run artifact locations (Epic 0 contract) ---

# There are two classes of artifacts:
#   (1) Run-scoped files under: runs_root()/runs/<run_id>/...
#       - meta.json, probes, reports, captures (tensors + manifests)
#
#   (2) Global per-run Parquet tables under: runs_root()/<artifact>/<run_id>.parquet
#       - metrics_scalar, geometry_state, events, etc.
#
# V0 canonical choices:
#   - Reports: runs/<run_id>/reports/report.md
#   - Captures: runs/<run_id>/captures/step_<N>/*
#   - Geometry: geometry_state/<run_id>.parquet (long-form: run_id, step, layer, metric, value)
#   - Events: events/<run_id>.parquet (single-run candidate events in v0)
#   - analysis_id: NOT part of paths in v0; may appear as a column later.

def runs_root() -> Path:
    return data_root() / "runs"


def run_dir(run_id: str) -> Path:
    return runs_root() / "runs" / run_id


def captures_dir(run_id: str) -> Path:
    return run_dir(run_id) / "captures"


def metrics_scalar_path(run_id: str) -> Path:
    return runs_root() / "metrics_scalar" / f"{run_id}.parquet"


def metrics_wide_path(run_id: str) -> Path:
    return runs_root() / "metrics_wide" / f"{run_id}.parquet"


def geometry_state_path(run_id: str) -> Path:
    return runs_root() / "geometry_state" / f"{run_id}.parquet"


def geometry_dynamics_path(run_id: str) -> Path:
    return runs_root() / "geometry_dynamics" / f"{run_id}.parquet"


def events_path(run_id: str) -> Path:
    return runs_root() / "events" / f"{run_id}.parquet"


def signatures_path(run_id: str) -> Path:
    return runs_root() / "signatures" / f"{run_id}.parquet"


def embeddings_path(run_id: str) -> Path:
    return runs_root() / "embeddings" / f"{run_id}.parquet"


def matches_path(run_id: str) -> Path:
    return runs_root() / "matches" / f"{run_id}.parquet"


def attention_summary_path(run_id: str) -> Path:
    return runs_root() / "attention_summary" / f"{run_id}.parquet"


def reports_dir(run_id: str) -> Path:
    return run_dir(run_id) / "reports"


def report_md_path(run_id: str) -> Path:
    return reports_dir(run_id) / "report.md"

def probe_fixed_path(run_id: str) -> Path:
    return run_dir(run_id) / "probe_fixed.pt"
    