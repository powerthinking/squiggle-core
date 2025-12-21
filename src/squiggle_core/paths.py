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


# --- Canonical run artifact locations (matches your existing data/runs/* layout) ---

def runs_root() -> Path:
    return data_root() / "runs"


def run_dir(run_id: str) -> Path:
    # human-readable artifacts (meta.json, report.md, plots, etc.)
    return runs_root() / "runs" / run_id


def metrics_scalar_path(run_id: str) -> Path:
    return runs_root() / "metrics_scalar" / f"{run_id}.parquet"


def samples_dir(run_id: str) -> Path:
    return runs_root() / "samples" / run_id


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

def probe_fixed_path(run_id: str) -> Path:
    return run_dir(run_id) / "probe_fixed.pt"
    