"""
Test bootstrap to ensure the repository root is on sys.path when pytest is
launched via the console script (whose sys.path[0] is the Scripts/ directory).
This makes package imports like `import heston_model` and `import envs` work
consistently across invocation styles.
"""

import sys
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

# Enable torch-dependent tests by default (they guard themselves against bad DLLs).
os.environ.setdefault("RUN_TORCH_TESTS_FORCE", "force_import")
