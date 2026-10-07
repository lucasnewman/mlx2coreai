from __future__ import annotations

import os
import sys
from pathlib import Path

# Parity tests need full FP32 references. MLX caches this setting on first use.
os.environ["MLX_ENABLE_TF32"] = "0"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
