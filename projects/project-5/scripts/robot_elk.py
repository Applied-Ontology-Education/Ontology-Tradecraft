#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import sys

from common.robot import detect_robot, run, build_elk_robot_command
from util.logger_config import config

logger = logging.getLogger(__name__)
from pathlib import Path

config(logger)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GENERATED_ROOT = PROJECT_ROOT / "generated"
SRC_ROOT = PROJECT_ROOT / "src"
DEFAULT_OUTPUT_FILE = SRC_ROOT / "module_reasoned.ttl"
DEFAULT_INPUT_FILE = SRC_ROOT / "module_augmented.ttl"
ROBOT_DIR = PROJECT_ROOT / "robot"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run ROBOT with the ELK reasoner on an input Turtle ontology and write the reasoned output."
        )
    )
    p.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_FILE),
        help="Input ontology TTL to reason over (default: src/module_augmented.ttl)",
    )
    p.add_argument(
        "--out",
        default=str(DEFAULT_OUTPUT_FILE),
        help="Output TTL path for reasoned ontology (default: src/module_reasoned.ttl)",
    )
    p.add_argument(
        "--robot",
        default=None,
        help="Path to robot executable or robot.jar (if not provided, auto-detect)",
    )
    p.add_argument(
        "--max-mem",
        default=os.getenv("ROBOT_JAVA_MAX_MEM", "6g"),
        help="Max Java heap for ROBOT when using robot.jar (default: 6g)",
    )
    return p.parse_args()


def main(
    input: str = str(DEFAULT_INPUT_FILE),
    out: str = str(DEFAULT_OUTPUT_FILE),
    robot: str | None = None,
    max_mem: str = os.getenv("ROBOT_JAVA_MAX_MEM", "6g"),
) -> None:
    in_path = Path(input)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Input ontology not found: {in_path}")

    robot_cmd = detect_robot(robot, ROBOT_DIR)
    cmd = build_elk_robot_command(in_path, out, robot_cmd, max_mem)
    robot_status = run(cmd)

    if robot_status != 0:
        raise RuntimeError(f"ROBOT ELK reasoner failed with exit code {robot_status}.")

    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"ROBOT reported success but output missing or empty: {out}")

    logger.info("Wrote reasoned ontology: %s", out)

    logger.info("Done.")


if __name__ == "__main__":
    try:
        args = parse_args()
        main(args.input, args.out, args.robot, args.max_mem)
    except Exception as e:
        logger.error("robot_elk failed: %s", e)
        sys.exit(1)
