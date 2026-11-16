"""Modular agent framework for closed-set VR-TriRef referent selection."""

from .context import AgentContext, Candidate, Evidence, load_contexts
from .pipeline import ExperimentConfig, run_experiment

__all__ = ["AgentContext", "Candidate", "Evidence", "ExperimentConfig", "load_contexts", "run_experiment"]
