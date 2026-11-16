#!/usr/bin/env python3
"""Host entry point; does not require a GPU Python environment."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_harness.runtime.separated import main

if __name__ == "__main__":
    raise SystemExit(main())
