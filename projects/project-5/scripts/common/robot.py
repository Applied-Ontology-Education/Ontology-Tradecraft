from __future__ import annotations

import logging
import shutil
import subprocess

from util.logger_config import config

logger = logging.getLogger(__name__)
from pathlib import Path

config(logger)


def detect_robot(robot_arg: str, robot_dir: Path) -> list[str]:
    """Return the command (argv list) to invoke ROBOT.

    Preference order:
    1) --robot argument (file or command)
    2) robot/robot.jar in repo (launch via `java -Xmx{mem} -jar ...`)
    3) `robot` available on PATH

    We return just the base command; memory flag will be added later if needed.
    """
    # If user provided a path/command, trust it
    if robot_arg:
        return [robot_arg]

    # Local jar inside repo
    jar = robot_dir / "robot.jar"
    if jar.exists():
        # We'll prepend java and -Xmx when building final command
        return [jar.as_posix()]  # marker that it's a jar

    # System robot on PATH
    robot_on_path = shutil.which("robot")
    if robot_on_path:
        return [robot_on_path]

    raise FileNotFoundError(
        "ROBOT not found. Provide --robot, place robot.jar in ./robot, or install 'robot' on PATH."
    )


def run(cmd: list[str]) -> int:
    logger.info("Running: %s", " ".join(cmd))
    # Stream subprocess output directly into our logs as it is produced.
    # Merge stderr into stdout to preserve interleaving and avoid threading.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    from collections import deque

    # Keep a rolling tail of output to summarize on failure
    tail: deque[str] = deque(maxlen=200)  # store last ~200 lines

    assert proc.stdout is not None
    for line in proc.stdout:
        # Log each line immediately; strip trailing newline for cleaner logs
        clean = line.rstrip("\n")
        logger.info(clean)
        tail.append(clean)

    proc.wait()
    rc = proc.returncode
    if rc != 0:
        # Provide a concise tail of combined output to aid debugging
        snippet = "\n".join(tail)
        if len(snippet) > 2000:
            snippet = "..." + snippet[-2000:]
        logger.error("ROBOT failed (exit %d). Last output:\n%s", rc, snippet)
        raise RuntimeError(f"ROBOT failed with exit code {rc}")
    return rc


def build_elk_robot_command(
    in_ttl: Path, out_ttl: Path, robot_cmd: list[str], max_mem: str
) -> list[str]:
    """Construct the ROBOT command to run ELK reasoning on a single input TTL and save result."""
    # We support two invocation modes:
    # - If robot_cmd[0] endswith .jar, we call via java -Xmx{max_mem} -jar robot.jar ...
    # - Else, we call the executable directly
    if robot_cmd and robot_cmd[0].endswith(".jar"):
        cmd = [
            "java",
            f"-Xmx{max_mem}",
            "-jar",
            robot_cmd[0],
            "reason",
            "--reasoner",
            "ELK",
            # Make output as rich as possible for ELK materialization
            "--axiom-generators",
            "SubClass EquivalentClass",
            "--equivalent-classes-allowed",
            "all",
            "--remove-redundant-subclass-axioms",
            "false",
            "--exclude-tautologies",
            "none",
            "--annotate-inferred-axioms",
            "true",
            "--input",
            in_ttl.as_posix(),
            "--output",
            out_ttl.as_posix(),
        ]
    else:
        cmd = [
            robot_cmd[0],
            "reason",
            "--reasoner",
            "ELK",
            # Make output as rich as possible for ELK materialization
            "--axiom-generators",
            "SubClass EquivalentClass",
            "--equivalent-classes-allowed",
            "all",
            "--remove-redundant-subclass-axioms",
            "false",
            "--exclude-tautologies",
            "none",
            "--annotate-inferred-axioms",
            "true",
            "--input",
            in_ttl.as_posix(),
            "--output",
            out_ttl.as_posix(),
        ]
    return cmd


def build_merge_robot_command(
    base_ttl: Path, add_ttls: list[Path] | tuple[Path, ...], out_ttl: Path, robot_cmd: list[str], max_mem: str
) -> list[str]:
    """Construct the ROBOT command to merge one base TTL with one or more additional TTLs,
    then reason with ELK, and save the result.

    Parameters:
    - base_ttl: base ontology
    - add_ttls: one or more TTL files to merge into base
    - out_ttl: output file path
    - robot_cmd: robot invocation as returned by detect_robot
    - max_mem: Java Xmx setting used when invoking via robot.jar
    """
    # We support two invocation modes:
    # - If robot_cmd[0] endswith .jar, we call via java -Xmx{max_mem} -jar robot.jar ...
    # - Else, we call the executable directly
    if robot_cmd and robot_cmd[0].endswith(".jar"):
        cmd = [
            "java",
            f"-Xmx{max_mem}",
            "-jar",
            robot_cmd[0],
            "merge",
            "--input",
            base_ttl.as_posix(),
        ]
    else:
        cmd = [
            robot_cmd[0],
            "merge",
            "--input",
            base_ttl.as_posix(),
        ]

    # Append all additional inputs
    for add in add_ttls:
        cmd.extend(["--input", add.as_posix()])

    # Reason and output
    cmd.extend([
        "reason",
        "--reasoner",
        "ELK",
        "--output",
        out_ttl.as_posix(),
    ])
    return cmd
