"""
Utility modules for the knowledge distillation training workflow.

This package exposes a small, testable surface area so our entrypoint script
`train_student.py` can stay tiny and declarative.
"""

from .trainer import run_trial
from .results import save_results

__all__ = ["run_trial", "save_results"]

