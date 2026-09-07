"""Run repository tests against this checkout's source, including in worktrees."""

from pathlib import Path
import sys

# A shared editable install may point to another checkout. Bind source before any
# test module imports the package; individual test import order must not select it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
