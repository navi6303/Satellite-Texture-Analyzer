"""
Root Application Launcher for Satellite Texture Analyzer.
Delegates to satellite-texture-analysis/app.py so users can launch either from
the root workspace (streamlit run app.py) or from within the subfolder.
"""

import sys
from pathlib import Path

# Add satellite-texture-analysis folder to sys.path
project_dir = Path(__file__).parent / "satellite-texture-analysis"
if str(project_dir.resolve()) not in sys.path:
    sys.path.insert(0, str(project_dir.resolve()))

# Run the app in satellite-texture-analysis/app.py
target_app = project_dir / "app.py"
with open(target_app, "r", encoding="utf-8") as f:
    code = compile(f.read(), str(target_app), "exec")
    exec(code, {"__name__": "__main__", "__file__": str(target_app)})
