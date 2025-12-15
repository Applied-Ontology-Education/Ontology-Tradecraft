#!/usr/bin/env bash

# Run ELK reasoning with ROBOT on a given ontology.
#
# Usage:
#   bash scripts/reason_with_elk.sh [INPUT] [OUTPUT]
#
# If INPUT/OUTPUT are omitted, defaults are taken from .env or:
#   INPUT  = projects/project-5/src/cco-module.ttl
#   OUTPUT = projects/project-5/src/module_reasoned.ttl

set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${THIS_DIR}" && pwd)"

# Load environment overrides from project .env if present
ENV_FILE="${PROJECT_DIR}/.env"
if [[ -f "${ENV_FILE}" ]]; then
  # Export all variables defined in .env while preserving existing env
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

# Defaults (can be overridden via .env or CLI args)
ROBOT_BIN=${ROBOT_BIN:-robot}
ELK_REASONER=${ELK_REASONER:-ELK}
ELK_INPUT_DEFAULT_REL="src/cco-module.ttl"
ELK_OUTPUT_DEFAULT_REL="src/module_reasoned.ttl"
ELK_INPUT=${ELK_INPUT:-"${PROJECT_DIR}/${ELK_INPUT_DEFAULT_REL}"}
ELK_OUTPUT=${ELK_OUTPUT:-"${PROJECT_DIR}/${ELK_OUTPUT_DEFAULT_REL}"}

# If ELK_INPUT/ELK_OUTPUT came from .env and are relative paths, make them
# relative to the project directory so the script can be run from anywhere.
case "${ELK_INPUT}" in
  /*) : ;; # absolute, leave as-is
  *) ELK_INPUT="${PROJECT_DIR}/${ELK_INPUT}" ;;
esac
case "${ELK_OUTPUT}" in
  /*) : ;;
  *) ELK_OUTPUT="${PROJECT_DIR}/${ELK_OUTPUT}" ;;
esac

# CLI overrides
if [[ ${#@} -ge 1 ]]; then
  ELK_INPUT="$1"
  # Make relative path relative to project dir for consistency
  if [[ "${ELK_INPUT}" != /* ]]; then
    ELK_INPUT="${PROJECT_DIR}/$1"
  fi
fi
if [[ ${#@} -ge 2 ]]; then
  ELK_OUTPUT="$2"
  if [[ "${ELK_OUTPUT}" != /* ]]; then
    ELK_OUTPUT="${PROJECT_DIR}/$2"
  fi
fi

usage() {
  cat <<USAGE
Run ELK reasoning with ROBOT.

Usage:
  bash scripts/reason_with_elk.sh [INPUT] [OUTPUT]

Defaults (can be set via .env):
  ROBOT_BIN   = ${ROBOT_BIN}
  ELK_REASONER= ${ELK_REASONER}
  ELK_INPUT   = ${ELK_INPUT}
  ELK_OUTPUT  = ${ELK_OUTPUT}

Examples:
  bash scripts/reason_with_elk.sh
  bash scripts/reason_with_elk.sh src/cco-module.ttl src/module_augmented.ttl
  ROBOT_JAVA_ARGS="-Xms2G -Xmx4G" bash scripts/reason_with_elk.sh
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

# Ensure ROBOT is available, else auto-bootstrap a local copy

# Allow users to control where/how we install ROBOT if missing
ROBOT_VERSION=${ROBOT_VERSION:-1.9.7}
ROBOT_DIR_DEFAULT="${PROJECT_DIR}/robot"
ROBOT_DIR=${ROBOT_DIR:-${ROBOT_DIR_DEFAULT}}

have_robot=false
if command -v "${ROBOT_BIN}" >/dev/null 2>&1; then
  have_robot=true
else
  # If ROBOT_BIN looks like a path (contains a slash) and does not exist, we will
  # place the wrapper and jar at that location (directory of ROBOT_BIN).
  if [[ "${ROBOT_BIN}" == */* ]]; then
    TARGET_BIN="${ROBOT_BIN}"
    TARGET_DIR="$(cd "$(dirname "${TARGET_BIN}")" 2>/dev/null && pwd || echo "$(dirname "${TARGET_BIN}")")"
  else
    TARGET_DIR="${ROBOT_DIR}"
    TARGET_BIN="${TARGET_DIR}/robot"
  fi

  mkdir -p "${TARGET_DIR}"

  ROBOT_JAR_PATH="${TARGET_DIR}/robot.jar"
  ROBOT_WRAPPER_PATH="${TARGET_BIN}"

  if [[ ! -f "${ROBOT_JAR_PATH}" ]]; then
    echo "[reason_with_elk] ROBOT not found. Downloading robot.jar v${ROBOT_VERSION} into ${TARGET_DIR}..."
    ROBOT_URL="https://github.com/ontodev/robot/releases/download/v${ROBOT_VERSION}/robot.jar"
    if command -v curl >/dev/null 2>&1; then
      curl -L -o "${ROBOT_JAR_PATH}" "${ROBOT_URL}"
    elif command -v wget >/dev/null 2>&1; then
      wget -O "${ROBOT_JAR_PATH}" "${ROBOT_URL}"
    else
      echo "Error: Neither curl nor wget is available to download ROBOT (needed automatically)." >&2
      exit 1
    fi
  fi

  if [[ ! -s "${ROBOT_JAR_PATH}" ]]; then
    echo "Error: Failed to download robot.jar to ${ROBOT_JAR_PATH}." >&2
    exit 1
  fi

  # Create wrapper script at desired ROBOT_BIN location
  if [[ ! -f "${ROBOT_WRAPPER_PATH}" ]]; then
    cat > "${ROBOT_WRAPPER_PATH}" <<'WRAP'
#!/usr/bin/env bash
set -euo pipefail
JAVA_BIN="${JAVA_BIN:-java}"
exec "${JAVA_BIN}" ${ROBOT_JAVA_ARGS:-} -jar "$(dirname "$0")/robot.jar" "$@"
WRAP
    chmod +x "${ROBOT_WRAPPER_PATH}"
  fi

  # Point ROBOT_BIN to the wrapper we just created
  ROBOT_BIN="${ROBOT_WRAPPER_PATH}"

  if [[ -x "${ROBOT_BIN}" ]]; then
    have_robot=true
  fi
fi

if [[ ${have_robot} == false ]]; then
  echo "Error: ROBOT is not available and could not be bootstrapped (ROBOT_BIN='${ROBOT_BIN}')." >&2
  exit 1
fi

if [[ ! -f "${ELK_INPUT}" ]]; then
  echo "Error: Input ontology not found: ${ELK_INPUT}" >&2
  exit 2
fi

mkdir -p "$(dirname "${ELK_OUTPUT}")"

echo "[reason_with_elk] ROBOT: ${ROBOT_BIN}"
echo "[reason_with_elk] Reasoner: ${ELK_REASONER}"
echo "[reason_with_elk] Input: ${ELK_INPUT}"
echo "[reason_with_elk] Output: ${ELK_OUTPUT}"

set -x
"${ROBOT_BIN}" reason \
  --reasoner "${ELK_REASONER}" \
  --input "${ELK_INPUT}" \
  --output "${ELK_OUTPUT}"
set +x

echo "[reason_with_elk] Done. Wrote: ${ELK_OUTPUT}"
