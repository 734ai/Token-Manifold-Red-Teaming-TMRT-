#!/usr/bin/env python3
"""
TMRT Command Line Interface

Main entry point for running TMRT experiments from command line.
"""

import sys
from pathlib import Path

# Ensure we can import tmrt
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

if __name__ == "__main__":
    from tmrt.demo import main
    main()
