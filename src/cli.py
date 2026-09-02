# Entry point. Run with `python src/cli.py`.

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        prog="ppochip",
        description="PPOtatoChip: RL-based chip placement optimization.",
    )
    parser.add_argument("--version", action="store_true", help="Show version")
    args, _ = parser.parse_known_args()

    if args.version:
        print("PPOtatoChip 0.1.0")
        sys.exit(0)

    from src.cli.tui import main as tui_main
    tui_main()


if __name__ == "__main__":
    main()