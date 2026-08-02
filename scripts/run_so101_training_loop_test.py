#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))
    from monitor_so101_training_dashboard import main as run_once

    checkpoint_dir = os.environ.get("SO101_CHECKPOINT_DIR")
    if checkpoint_dir and "--checkpoint-name" not in sys.argv:
        sys.argv.extend(["--checkpoint-name", Path(checkpoint_dir).name])
    if "--skip-validation" not in sys.argv:
        sys.argv.append("--skip-validation")
    if "--iterations" not in sys.argv:
        sys.argv.extend(["--iterations", "1"])
    run_once()


if __name__ == "__main__":
    main()
