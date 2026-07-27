"""Root conftest: make the repo root importable (pipeline/)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
