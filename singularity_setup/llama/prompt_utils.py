"""Compatibility import for the prompt utilities shared by all models."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from singularity_setup.common.prompt_utils import SYSTEM_PROMPT, build_tabular_prompt

__all__ = ["SYSTEM_PROMPT", "build_tabular_prompt"]
