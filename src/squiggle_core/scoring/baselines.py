from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from squiggle_core import paths
from squiggle_core.scoring.squiggle_scoring import MetricBaseline, build_baselines_from_samples


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _threshold_for_metric(metric_name: str, *, rank_threshold: float, mass_threshold: float) -> float:
    if metric_name == "effective_rank":
        return float(rank_threshold)
    if metric_name.startswith("topk_mass_"):
        return float(mass_threshold)
    return float(rank_threshold)


def _merge_overlapping_windows(windows: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not windows:
        return []
    windows = sorted(windows)
    merged: List[Tuple[int, int]] = []
    cur_s, cur_e = windows[0]
    for s, e in windows[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


def _window_stats_from_deltas(deltas_abs: List[float], start_idx: int, end_idx: int) -> Tuple[float, float]:
    w = deltas_abs[start_idx : end_idx + 1]
    if not w:
        return 0.0, 0.0
    m = float(max(w))
    w_sorted = sorted(w)
    v = float(w_sorted[len(w_sorted) // 2])
    return m, v


def build_windowed_baselines_from_geometry_df(
    geom: pd.DataFrame,
    *,
    rank_threshold: float,
    mass_threshold: float,
    window_radius_steps: int,
) -> Tuple[Dict[str, MetricBaseline], Dict[str, float]]:
    required = {"layer", "metric", "step", "value"}
    missing = required - set(geom.columns)
    if missing:
        raise ValueError(f"Geometry state missing required columns for baseline build: {sorted(missing)}")

    size_samples: Dict[str, List[float]] = {}
    vol_samples: Dict[str, List[float]] = {}

    for (layer, metric), g in geom.groupby(["layer", "metric"], sort=True):
        g = g.sort_values("step")
        values = g["value"].to_numpy()
        steps = g["step"].to_numpy()
        if len(values) < 2:
            continue

        deltas_abs = [float(abs(values[i] - values[i - 1])) for i in range(1, len(values))]
        if not deltas_abs:
            continue

        thr = _threshold_for_metric(str(metric), rank_threshold=rank_threshold, mass_threshold=mass_threshold)
        hit_idxs = [i for i, d in enumerate(deltas_abs) if d > thr]
        if not hit_idxs:
            continue

        raw_windows: List[Tuple[int, int]] = []
        for idx in hit_idxs:
            s = max(0, idx - int(window_radius_steps))
            e = min(len(deltas_abs) - 1, idx + int(window_radius_steps))
            raw_windows.append((s, e))

        windows = _merge_overlapping_windows(raw_windows)
        key = str(metric)

        for s, e in windows:
            m, v = _window_stats_from_deltas(deltas_abs, s, e)
            size_samples.setdefault(key, []).append(m)
            vol_samples.setdefault(key, []).append(v)

    baselines = build_baselines_from_samples(size_samples)
    vol_baseline: Dict[str, float] = {}
    for k, xs in vol_samples.items():
        if not xs:
            continue
        xs_sorted = sorted(xs)
        vol_baseline[k] = float(xs_sorted[len(xs_sorted) // 2])

    return baselines, vol_baseline


def build_metric_baselines_from_run(
    *,
    run_id: str,
    rank_threshold: float,
    mass_threshold: float,
    window_radius_steps: int,
) -> Tuple[Dict[str, MetricBaseline], Dict[str, float]]:
    geom_path = paths.geometry_state_path(run_id)
    if not geom_path.exists():
        raise FileNotFoundError(f"Geometry state parquet not found for run_id='{run_id}'. Expected: {geom_path}")

    geom = pd.read_parquet(geom_path)
    return build_windowed_baselines_from_geometry_df(
        geom,
        rank_threshold=rank_threshold,
        mass_threshold=mass_threshold,
        window_radius_steps=window_radius_steps,
    )


def baseline_id_from_run_id(run_id: str) -> str:
    return f"baseline_run_{run_id}"


def write_baseline_from_run(
    *,
    run_id: str,
    baseline_id: Optional[str] = None,
    rank_threshold: float = 0.2,
    mass_threshold: float = 0.03,
    window_radius_steps: int = 5,
) -> Path:
    bid = baseline_id or baseline_id_from_run_id(run_id)
    out_path = paths.scoring_baseline_path(bid)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    baselines, volatility_baseline = build_metric_baselines_from_run(
        run_id=run_id,
        rank_threshold=rank_threshold,
        mass_threshold=mass_threshold,
        window_radius_steps=window_radius_steps,
    )

    payload = {
        "schema_version": "scoring_baseline@0.2",
        "baseline_id": bid,
        "source_run_id": run_id,
        "created_at_utc": _utc_now_iso(),
        "detector": {
            "type": "change_point",
            "rank_threshold": float(rank_threshold),
            "mass_threshold": float(mass_threshold),
            "window_radius_steps": int(window_radius_steps),
        },
        "metric_baselines": {k: {"median": float(v.median), "mad": float(v.mad)} for k, v in baselines.items()},
        "volatility_baseline": {k: float(v) for k, v in volatility_baseline.items()},
    }

    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def load_baseline(*, baseline_id: str) -> Tuple[str, Dict[str, MetricBaseline]]:
    p = paths.scoring_baseline_path(baseline_id)
    if not p.exists():
        raise FileNotFoundError(f"Scoring baseline not found: {p}")

    obj = json.loads(p.read_text(encoding="utf-8"))
    if obj.get("schema_version") not in {"scoring_baseline@0.1", "scoring_baseline@0.2"}:
        raise ValueError(f"Unexpected scoring baseline schema_version: {obj.get('schema_version')}")

    source_run_id = str(obj.get("source_run_id"))
    mb = obj.get("metric_baselines")
    if not isinstance(mb, dict):
        raise TypeError("metric_baselines must be a dict")

    baselines: Dict[str, MetricBaseline] = {}
    for k, v in mb.items():
        if not isinstance(v, dict):
            raise TypeError(f"metric_baselines[{k!r}] must be a dict")
        baselines[str(k)] = MetricBaseline(median=float(v["median"]), mad=float(v["mad"]))

    return source_run_id, baselines


def load_baseline_with_volatility(
    *,
    baseline_id: str,
) -> Tuple[str, Dict[str, MetricBaseline], Dict[str, float]]:
    p = paths.scoring_baseline_path(baseline_id)
    if not p.exists():
        raise FileNotFoundError(f"Scoring baseline not found: {p}")

    obj = json.loads(p.read_text(encoding="utf-8"))
    schema_version = obj.get("schema_version")
    if schema_version not in {"scoring_baseline@0.1", "scoring_baseline@0.2"}:
        raise ValueError(f"Unexpected scoring baseline schema_version: {schema_version}")

    source_run_id, baselines = load_baseline(baseline_id=baseline_id)
    if schema_version == "scoring_baseline@0.1":
        return source_run_id, baselines, {}

    vb = obj.get("volatility_baseline")
    if vb is None:
        return source_run_id, baselines, {}
    if not isinstance(vb, dict):
        raise TypeError("volatility_baseline must be a dict")

    vol_baseline: Dict[str, float] = {str(k): float(v) for k, v in vb.items()}
    return source_run_id, baselines, vol_baseline


def _cmd_write(args: argparse.Namespace) -> int:
    p = write_baseline_from_run(
        run_id=str(args.run_id),
        baseline_id=(str(args.baseline_id) if args.baseline_id is not None else None),
        rank_threshold=float(args.rank_threshold),
        mass_threshold=float(args.mass_threshold),
        window_radius_steps=int(args.window_radius_steps),
    )
    print(str(p))
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    source_run_id, baselines, vol = load_baseline_with_volatility(baseline_id=str(args.baseline_id))
    print(f"baseline_id={args.baseline_id}")
    print(f"source_run_id={source_run_id}")
    print(f"metrics={len(baselines)}")
    print(f"metrics_with_volatility={len(vol)}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m squiggle_core.scoring.baselines")
    sub = p.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("write")
    w.add_argument("--run-id", required=True)
    w.add_argument("--baseline-id", default=None)
    w.add_argument("--rank-threshold", type=float, default=0.2)
    w.add_argument("--mass-threshold", type=float, default=0.03)
    w.add_argument("--window-radius-steps", type=int, default=5)
    w.set_defaults(func=_cmd_write)

    s = sub.add_parser("show")
    s.add_argument("--baseline-id", required=True)
    s.set_defaults(func=_cmd_show)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
