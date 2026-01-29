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


def events_candidates_path(run_id: str, analysis_id: str | None = None) -> Path:
    """
    Path to events candidates parquet.

    Args:
        run_id: The run identifier
        analysis_id: Optional analysis version. If provided, stores in subdirectory:
                     events_candidates/<run_id>/<analysis_id>.parquet
                     If None, uses legacy path: events_candidates/<run_id>.parquet
    """
    if analysis_id:
        return runs_root() / "events_candidates" / run_id / f"{analysis_id}.parquet"
    return runs_root() / "events_candidates" / f"{run_id}.parquet"


def detection_summary_path(run_id: str, analysis_id: str | None = None) -> Path:
    """
    Path to detection summary parquet.

    Args:
        run_id: The run identifier
        analysis_id: Optional analysis version. If provided, stores in subdirectory:
                     detection_summary/<run_id>/<analysis_id>.parquet
                     If None, uses legacy path: detection_summary/<run_id>.parquet
    """
    if analysis_id:
        return runs_root() / "detection_summary" / run_id / f"{analysis_id}.parquet"
    return runs_root() / "detection_summary" / f"{run_id}.parquet"


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


def reports_dir(run_id: str, analysis_id: str | None = None) -> Path:
    """
    Path to reports directory.

    Args:
        run_id: The run identifier
        analysis_id: Optional analysis version. If provided, stores in subdirectory:
                     runs/<run_id>/reports/<analysis_id>/
    """
    base = run_dir(run_id) / "reports"
    if analysis_id:
        return base / analysis_id
    return base


def report_md_path(run_id: str, analysis_id: str | None = None) -> Path:
    """
    Path to report.md file.

    Args:
        run_id: The run identifier
        analysis_id: Optional analysis version
    """
    return reports_dir(run_id, analysis_id) / "report.md"

def llm_analysis_path(run_id: str, analysis_id: str | None = None) -> Path:
    """
    Path to LLM analysis JSON file.

    Args:
        run_id: The run identifier
        analysis_id: Optional analysis version
    """
    return reports_dir(run_id, analysis_id) / "llm_analysis.json"


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


# --- AIMO / Math Corpus candidate pool paths ---


def candidate_pool_root() -> Path:
    """Root directory for AIMO candidate pool work product (slices, classifications, etc.)."""
    return data_root() / "candidate_pool"


def candidate_pool_slices_dir() -> Path:
    """Directory containing candidate pool slices."""
    return candidate_pool_root() / "slices"


def candidate_pool_slice_dir(slice_id: str) -> Path:
    """Directory for a specific slice."""
    return candidate_pool_slices_dir() / slice_id


def training_splits_dir() -> Path:
    """Directory containing training splits (train/val/val_family)."""
    return candidate_pool_root() / "splits"


def training_split_dir(split_id: str) -> Path:
    """Directory for a specific training split."""
    return training_splits_dir() / split_id


# --- Analysis ID utilities ---


def generate_analysis_id(
    warmup_fraction: float = 0.1,
    max_pre_warmup: int = 1,
    peak_suppression_radius: int = 15,
    max_events_per_series: int = 5,
    adaptive_k: float = 2.5,
    **kwargs,
) -> str:
    """
    Generate a deterministic analysis_id from detection parameters.

    The ID encodes the key parameters that affect event detection results.
    Format: w{warmup_fraction*100}_p{max_pre_warmup}_r{radius}_e{max_events}_k{adaptive_k*10}

    Example: "w10_p1_r15_e5_k25" for default parameters
    """
    # Convert to integers for cleaner IDs
    w = int(warmup_fraction * 100)
    k = int(adaptive_k * 10)

    return f"w{w}_p{max_pre_warmup}_r{peak_suppression_radius}_e{max_events_per_series}_k{k}"


def list_analysis_ids(run_id: str) -> list[str]:
    """
    List all analysis_ids available for a given run.

    Returns analysis IDs found in events_candidates/<run_id>/ directory.
    """
    analysis_dir = runs_root() / "events_candidates" / run_id
    if not analysis_dir.exists():
        return []

    return sorted(
        p.stem for p in analysis_dir.glob("*.parquet")
    )