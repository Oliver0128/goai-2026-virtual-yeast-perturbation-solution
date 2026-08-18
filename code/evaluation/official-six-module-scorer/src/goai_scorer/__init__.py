"""GOAI official-faithful provisional six-module scorer."""

__version__ = "1.1.0"

from .evaluate import EvaluationArtifacts, evaluate_validation

__all__ = ["EvaluationArtifacts", "evaluate_validation"]
