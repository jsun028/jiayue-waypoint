"""Pytest configuration and shared fixtures."""

import sys
import pathlib

# Add parent directory to path so we can import NL modules
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

