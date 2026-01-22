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


# --- Canonical run artifact locations (v2 contract) ---

# There are two classes of artifacts:
#   (1) Run-scoped files under: runs_root()/runs/<run_id>/...
#       - meta.json, probes, reports, captures (tensors + manifests)
#
#   (2) Global per-run Parquet tables under: runs_root()/<artifact>/<run_id>.parquet
#       - metrics_scalar, geometry_state, events, etc.

# Canonical choices:
#   - Reports: runs/<run_id>/reports/report.md
#   - Captures: runs/<run_id>/captures/step_<N>/*
#   - Geometry: geometry_state/<run_id>.parquet (analysis artifacts; include v2 spine columns)
#   - Event candidates (single-run): events_candidates/<run_id>.parquet
#   - Events (consensus, cross-seed): events/<test_id>.parquet
#   - analysis_id: stored as a column (not encoded into v2 filenames).

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


def events_candidates_path(run_id: str) -> Path:
    return runs_root() / "events_candidates" / f"{run_id}.parquet"


def events_consensus_path(test_id: str) -> Path:
    return runs_root() / "events" / f"{test_id}.parquet"


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


def probes_dir(run_id: str) -> Path:
    return run_dir(run_id) / "probes"


def probe_dir(run_id: str, probe_name: str) -> Path:
    return probes_dir(run_id) / probe_name


def probe_manifest_path(run_id: str, probe_name: str) -> Path:
    return probe_dir(run_id, probe_name) / "manifest.json"


def probe_captures_dir(run_id: str, probe_name: str) -> Path:
    return probe_dir(run_id, probe_name) / "captures"


def probe_index_path(run_id: str, probe_name: str) -> Path:
    return probe_dir(run_id, probe_name) / "index.parquet"


def probe_stats_path(run_id: str, probe_name: str) -> Path:
    return probe_dir(run_id, probe_name) / "stats.json"


def scoring_baselines_dir() -> Path:
    return runs_root() / "scoring_baselines"


def scoring_baseline_path(baseline_id: str) -> Path:
    return scoring_baselines_dir() / f"{baseline_id}.json"