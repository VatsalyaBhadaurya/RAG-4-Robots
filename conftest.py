"""Make the repo root importable so `import r4r` works under bare `pytest`."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
