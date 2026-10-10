"""Command line entry point for Colab and local experiments."""

import argparse
import json
from pathlib import Path

from .runner import RunConfig, run


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train and evaluate UJIIndoorLoc models")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    if "buildings" in settings:
        settings["buildings"] = tuple(settings["buildings"])
    run(args.data_dir, args.output_dir, RunConfig(**settings))


if __name__ == "__main__":
    main()
