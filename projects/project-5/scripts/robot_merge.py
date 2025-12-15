#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import sys

from common.robot import detect_robot, run, build_merge_robot_command
from common.settings import build_settings
from util.logger_config import config

logger = logging.getLogger(__name__)
from pathlib import Path

config(logger)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
settings = build_settings(PROJECT_ROOT, DATA_ROOT)
GENERATED_ROOT = PROJECT_ROOT / "generated"
DEFAULT_MERGE_AXIOMS = GENERATED_ROOT / "accepted_el.ttl"
SRC_ROOT = PROJECT_ROOT / "src"
DEFAULT_OUTPUT_FILE = SRC_ROOT / "module_augmented.ttl"
ROBOT_DIR = PROJECT_ROOT / "robot"


def parse_args(settings) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Merge generated/accepted_el.ttl with src/cco-module.ttl using ROBOT, reason with ELK, "
            "and write src/module_augmented.ttl"
        )
    )
    p.add_argument(
        "--base",
        default=str(settings["reference_ontology"]),
        help=f"Base ontology TTL to merge into (default: {settings["reference_ontology"]})",
    )
    p.add_argument(
        "--add",
        action="append",
        help=(
            "Additional TTL file(s) to merge. May be provided multiple times, e.g. "
            "--add generated/accepted_el.ttl --add other.ttl. "
            "If omitted, defaults to generated/accepted_el.ttl"
        ),
    )
    p.add_argument(
        "--out",
        default=str(DEFAULT_OUTPUT_FILE),
        help="Output TTL path (default: src/module_augmented.ttl)",
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
    args = p.parse_args()

    # Default for --add if not provided
    if not args.add:
        args.add = [str(DEFAULT_MERGE_AXIOMS)]

    # Support accidental comma-separated lists in a single --add
    expanded: list[str] = []
    for val in args.add:
        if isinstance(val, str) and "," in val:
            expanded.extend([v.strip() for v in val.split(",") if v.strip()])
        else:
            expanded.append(val)
    args.add = expanded

    return args


def main(
    base: str = str(settings["reference_ontology"]),
    add: list[str] | tuple[str, ...] | str = str(DEFAULT_MERGE_AXIOMS),
    out: str = str(DEFAULT_OUTPUT_FILE),
    robot: str | None = None,
    max_mem: str = os.getenv("ROBOT_JAVA_MAX_MEM", "6g"),
) -> None:

    base = Path(base)
    # Normalize add to a list of Paths
    if isinstance(add, (list, tuple)):
        add_paths = [Path(a) for a in add]
    else:
        add_paths = [Path(add)]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not base.exists():
        raise FileNotFoundError(f"Base ontology not found: {base}")
    missing = [p for p in add_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "One or more generated axioms files not found: " + ", ".join(str(m) for m in missing)
        )

    robot_cmd = detect_robot(robot, ROBOT_DIR)
    cmd = build_merge_robot_command(base, add_paths, out, robot_cmd, max_mem)
    run(cmd)

    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"ROBOT reported success but output missing or empty: {out}")
    logger.info("Wrote augmented ontology: %s", out)


if __name__ == "__main__":
    try:
        args = parse_args(settings)
        main(args.base, args.add, args.out, args.robot, args.max_mem)
    except Exception as e:
        logger.error("robot_merge failed: %s", e)
        sys.exit(1)


