from __future__ import annotations

import argparse

from classical_baselines.pipeline import load_config, train_and_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config for classical baseline training")
    args = ap.parse_args()

    cfg = load_config(args.config)
    train_and_eval(cfg)


if __name__ == "__main__":
    main()

