from __future__ import annotations

import argparse
import json

from .config import load_config
from .runner import BacktestRunner



def main() -> None:
    parser = argparse.ArgumentParser(description="Run industry trading simulation")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    runner = BacktestRunner(config, config_path=args.config)
    result = runner.run()

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
