import sys
from pathlib import Path

pkg_root = Path(__file__).parent / "satellite-texture-analysis"
if str(pkg_root.resolve()) not in sys.path:
    sys.path.insert(0, str(pkg_root.resolve()))
