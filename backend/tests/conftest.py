"""
ASTRA — pytest configuration and shared fixtures.
"""

import sys
from pathlib import Path

# Ensure the backend package is importable during tests
sys.path.insert(0, str(Path(__file__).parent.parent))
