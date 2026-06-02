"""Make the package importable when running the suite without an editable
install (e.g. `pytest` from a host venv). Harmless under `uv run` (the path is
already present). The tests/ dir is added to sys.path by pytest's prepend import
mode, which makes `import _helpers` resolve.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
