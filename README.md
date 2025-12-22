# squiggle-core

Core data contracts, schemas, and filesystem conventions for the Squiggle project.

This package defines the *stable interface* between training, analysis, and reporting.
Nothing here is model-specific.

## Responsibilities

- Canonical run directory structure
- Standardized artifact paths
- Parquet schemas and validation
- Shared utilities used across repos

## Key Concepts

### Run layout

Each run is identified by a deterministic `run_id` and produces:

- `runs/<run_id>/meta.json`
- `metrics_scalar/<run_id>.parquet`
- `samples/<run_id>/step_*/`
- `geometry_state/<run_id>.parquet`
- `events/<run_id>.parquet`
- `runs/<run_id>/report.md`

All other repos write *through* these paths.

### Schemas

Parquet schemas live in:
- `squiggle_core.schemas.parquet_schemas`

They define required columns and enforce ordering for:

- Geometry state
- Event detection
- Scalar metrics

Schema validation is intentionally strict to prevent silent drift.

## Philosophy

`squiggle-core` should change *slowly*.

If a downstream component breaks a schema, that’s a signal the pipeline has drifted —
not something to paper over.

## Versioning

This repo is versioned conservatively and should remain backward compatible
across multiple experiment iterations.
