"""Experiment specification schema.

Defines the formal structure for experiments as specified in design_and_contracts.md Section 6.
An Experiment is a structured comparison of 2+ Tests.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EventConsensusRules:
    """Rules for determining event consensus across seeds."""

    step_tolerance: int = 5
    """Maximum step difference for events to be considered aligned."""

    min_seed_fraction: float = 1.0
    """Minimum fraction of seeds that must show event (1.0 = all seeds required)."""


@dataclass
class ArmSpec:
    """Specification for a single arm of an experiment.

    An arm corresponds to one Test (multi-seed runs with identical config except seed).
    """

    test_config: str
    """Path to the test config YAML (relative to spec/ directory)."""

    curriculum: str | None = None
    """Path to curriculum YAML (relative to spec/ directory), if using curriculum override."""

    description: str | None = None
    """Optional description of this arm's purpose."""


@dataclass
class ExperimentSpec:
    """Formal specification for an experiment.

    An experiment compares 2+ Tests (arms) while holding invariants constant.
    Each arm is run with multiple seeds to enable consensus analysis.

    Structure on disk:
        experiments/<exp_id>/
          README.md
          spec/
            experiment.yaml      # This spec
            <arm>_test.yaml      # Per-arm test configs
          outputs/
            compare.md           # Cross-arm comparison report
    """

    version: int = 1
    """Schema version."""

    exp_id: str = ""
    """Unique experiment identifier (e.g., 'exp_curriculum_ab')."""

    hypothesis: str = ""
    """Falsifiable claim being tested."""

    isolates: str = ""
    """The factor being tested (what differs between arms)."""

    invariants: list[str] = field(default_factory=list)
    """What stays constant across all arms."""

    arms: dict[str, ArmSpec] = field(default_factory=dict)
    """Arm configurations keyed by arm name (e.g., 'iid', 'blocked')."""

    seeds: list[int] = field(default_factory=lambda: [42, 123, 456])
    """Seeds to run for each arm."""

    primary_outcomes: list[str] = field(default_factory=list)
    """What we measure to evaluate the hypothesis."""

    event_consensus_rules: EventConsensusRules = field(default_factory=EventConsensusRules)
    """Rules for determining event consensus."""

    @classmethod
    def from_yaml(cls, path: Path | str) -> ExperimentSpec:
        """Load experiment spec from YAML file."""
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentSpec:
        """Create ExperimentSpec from dictionary."""
        # Parse event consensus rules
        consensus_data = data.pop("event_consensus_rules", {})
        consensus_rules = EventConsensusRules(**consensus_data)

        # Parse arms
        arms_data = data.pop("arms", {})
        arms = {}
        for arm_name, arm_data in arms_data.items():
            if isinstance(arm_data, str):
                # Simple case: just a test config path
                arms[arm_name] = ArmSpec(test_config=arm_data)
            else:
                arms[arm_name] = ArmSpec(**arm_data)

        return cls(
            version=data.get("version", 1),
            exp_id=data.get("exp_id", ""),
            hypothesis=data.get("hypothesis", ""),
            isolates=data.get("isolates", ""),
            invariants=data.get("invariants", []),
            arms=arms,
            seeds=data.get("seeds", [42, 123, 456]),
            primary_outcomes=data.get("primary_outcomes", []),
            event_consensus_rules=consensus_rules,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "version": self.version,
            "exp_id": self.exp_id,
            "hypothesis": self.hypothesis,
            "isolates": self.isolates,
            "invariants": self.invariants,
            "arms": {
                name: {
                    "test_config": arm.test_config,
                    "curriculum": arm.curriculum,
                    "description": arm.description,
                }
                for name, arm in self.arms.items()
            },
            "seeds": self.seeds,
            "primary_outcomes": self.primary_outcomes,
            "event_consensus_rules": {
                "step_tolerance": self.event_consensus_rules.step_tolerance,
                "min_seed_fraction": self.event_consensus_rules.min_seed_fraction,
            },
        }

    def to_yaml(self, path: Path | str) -> None:
        """Write spec to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def compute_spec_hash(self) -> str:
        """Compute a content-based hash of the spec for versioning."""
        canonical = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]

    def validate(self) -> list[str]:
        """Validate the spec, returning list of errors (empty if valid)."""
        errors = []

        if not self.exp_id:
            errors.append("exp_id is required")

        if not self.hypothesis:
            errors.append("hypothesis is required")

        if len(self.arms) < 2:
            errors.append("At least 2 arms required for an experiment")

        if not self.seeds:
            errors.append("At least 1 seed required")

        if not self.primary_outcomes:
            errors.append("At least 1 primary outcome required")

        for arm_name, arm in self.arms.items():
            if not arm.test_config:
                errors.append(f"Arm '{arm_name}' missing test_config")

        return errors
