import sys
from pathlib import Path

# Add satellite-texture-analysis folder to sys.path
pkg_root = Path(__file__).parent.resolve()
if str(pkg_root) not in sys.path:
    sys.path.insert(0, str(pkg_root))
