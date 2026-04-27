from __future__ import annotations

import sys
from pathlib import Path


def add_project_src_to_path() -> None:
    src_path = Path(__file__).resolve().parents[1] / "src"
    src_path_string = str(src_path)
    if src_path_string not in sys.path:
        sys.path.insert(0, src_path_string)
