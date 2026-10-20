"""
Pytest configuration and environment setup.
Ensures repository root is present in sys.path.
"""

import sys
from pathlib import Path

# Guarantee root directory is in sys.path for test discovery
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
