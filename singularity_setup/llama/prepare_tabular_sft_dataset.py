"""Compatibility wrapper; the SFT splitter is shared across model families."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from singularity_setup.common.prepare_tabular_sft_dataset import main


if __name__ == "__main__":
    main()
