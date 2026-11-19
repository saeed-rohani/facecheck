"""Run facecheck straight from this folder, without installing it.

Install the only two dependencies first, as ready-made builds:

    pip install --only-binary=:all: opencv-contrib-python numpy

Then, for example:

    python run.py register "Saeed Rohani"    # register a new person
    python run.py check                      # who is this?
    python run.py check --name "Saeed Rohani"  # is this Saeed?
    python run.py --help                     # all commands
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from facecheck.cli import main  # noqa: E402  (needs the path set above)

raise SystemExit(main())
