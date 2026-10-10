"""Command line entry point for Colab and local experiments."""

import argparse
import json
from pathlib import Path

from ips.algorithms.factory import available_models
from ips.data.preprocessors.factory import available_preprocessors

from .runner import RunConfig, run


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train and evaluate UJIIndoorLoc models")
    parser.add_argument("--list-options", action="store_true", help="Print discovered plugins as JSON")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", help="Override the model in the config")
    parser.add_argument("--preprocess", help="Override the preprocessor in the config")
    parser.add_argument("--preprocess-params", help="JSON object with plugin parameters")
    args = parser.parse_args(argv)
    if args.list_options:
        print(json.dumps({"models": available_models(), "preprocessors": available_preprocessors()}))
        return
    if not all((args.config, args.data_dir, args.output_dir)):
        parser.error("--config, --data-dir and --output-dir are required for training")
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    if args.model:
        settings["model"] = args.model
    if args.preprocess:
        if args.preprocess != settings.get("preprocess", "rss_standard"):
            settings["preprocess_params"] = {}  # Params from another preprocessor must not leak.
        settings["preprocess"] = args.preprocess
    if args.preprocess_params is not None:
        try:
            params = json.loads(args.preprocess_params)
        except json.JSONDecodeError as exc:
            parser.error(f"--preprocess-params is not valid JSON: {exc}")
        if not isinstance(params, dict):
            parser.error("--preprocess-params must be a JSON object")
        settings["preprocess_params"] = params
    if "buildings" in settings:
        settings["buildings"] = tuple(settings["buildings"])
    run(args.data_dir, args.output_dir, RunConfig(**settings))


if __name__ == "__main__":
    main()
