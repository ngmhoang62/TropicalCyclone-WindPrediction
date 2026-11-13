"""Run the research code directly from this repository, without package installation."""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

if __name__ == "__main__":
    importlib.import_module("cyclone_wind.cli").main()
