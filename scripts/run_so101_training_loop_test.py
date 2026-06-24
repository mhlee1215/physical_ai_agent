#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))
    from monitor_so101_training_dashboard import main as run_once

    if "--iterations" not in sys.argv:
        sys.argv.extend(["--iterations", "1"])
    run_once()


if __name__ == "__main__":
    main()
