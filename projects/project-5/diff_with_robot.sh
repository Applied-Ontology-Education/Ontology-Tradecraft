#!/usr/bin/env bash

# Make a semantic diff between two ontology files using ROBOT.
#
# Usage:
#   bash scripts/diff_with_robot.sh [LEFT] [RIGHT] [OUT]
#
# If arguments are omitted, defaults are taken from .env or:
#   LEFT  = src/InformationEntityOntology.ttl
#   RIGHT = src/module_reasoned.ttl
#   OUT   = reports/elk_diff.md (format configurable via DIFF_FORMAT)

set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# project-5 root (this script lives under project-5/scripts)
PROJECT_DIR="$(cd "${THIS_DIR}" && pwd)"

# Load environment overrides from project .env if present
ENV_FILE="${PROJECT_DIR}/.env"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

# Defaults (can be overridden via .env or CLI args)
ROBOT_BIN=${ROBOT_BIN:-robot}
DIFF_LEFT_DEFAULT_REL="src/InformationEntityOntology.ttl"
DIFF_RIGHT_DEFAULT_REL="src/module_reasoned.ttl"
DIFF_FORMAT=${DIFF_FORMAT:-md}
DIFF_OUTPUT_DEFAULT_REL="reports/elk_diff.${DIFF_FORMAT}"

LEFT=${DIFF_LEFT:-"${PROJECT_DIR}/${DIFF_LEFT_DEFAULT_REL}"}
RIGHT=${DIFF_RIGHT:-"${PROJECT_DIR}/${DIFF_RIGHT_DEFAULT_REL}"}
OUT=${DIFF_OUTPUT:-"${PROJECT_DIR}/${DIFF_OUTPUT_DEFAULT_REL}"}

# Normalize diff format: accept 'md' shorthand and map to 'markdown' for ROBOT
ROBOT_DIFF_FORMAT="${DIFF_FORMAT}"
if [[ "${ROBOT_DIFF_FORMAT}" == "md" ]]; then
  ROBOT_DIFF_FORMAT="markdown"
fi

# Normalize env-provided relative paths to be relative to project dir
case "${LEFT}" in
  /*) : ;;
  *) LEFT="${PROJECT_DIR}/${LEFT}" ;;
esac
case "${RIGHT}" in
  /*) : ;;
  *) RIGHT="${PROJECT_DIR}/${RIGHT}" ;;
esac
case "${OUT}" in
  /*) : ;;
  *) OUT="${PROJECT_DIR}/${OUT}" ;;
esac

# CLI overrides
if [[ ${#@} -ge 1 ]]; then
  LEFT="$1"; [[ "$1" != /* ]] && LEFT="${PROJECT_DIR}/$1"
fi
if [[ ${#@} -ge 2 ]]; then
  RIGHT="$2"; [[ "$2" != /* ]] && RIGHT="${PROJECT_DIR}/$2"
fi
if [[ ${#@} -ge 3 ]]; then
  OUT="$3"; [[ "$3" != /* ]] && OUT="${PROJECT_DIR}/$3"
fi

usage() {
  cat <<USAGE
ROBOT diff between two ontology files.

Usage:
  bash scripts/diff_with_robot.sh [LEFT] [RIGHT] [OUT]

Defaults (can be set via .env):
  ROBOT_BIN   = ${ROBOT_BIN}
  DIFF_LEFT   = ${LEFT}
  DIFF_RIGHT  = ${RIGHT}
  DIFF_FORMAT = ${DIFF_FORMAT}
  DIFF_OUTPUT = ${OUT}

Examples:
  bash scripts/diff_with_robot.sh
  bash scripts/diff_with_robot.sh src/InformationEntityOntology.ttl src/module_reasoned.ttl reports/elk_diff.md
  DIFF_FORMAT=html bash scripts/diff_with_robot.sh
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

# Ensure ROBOT is available, else auto-bootstrap a local copy (same approach as reason_with_elk.sh)
ROBOT_VERSION=${ROBOT_VERSION:-1.9.7}
ROBOT_DIR_DEFAULT="${PROJECT_DIR}/robot"
ROBOT_DIR=${ROBOT_DIR:-${ROBOT_DIR_DEFAULT}}

have_robot=false
if command -v "${ROBOT_BIN}" >/dev/null 2>&1; then
  have_robot=true
else
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
    echo "[diff_with_robot] ROBOT not found. Downloading robot.jar v${ROBOT_VERSION} into ${TARGET_DIR}..."
    ROBOT_URL="https://github.com/ontodev/robot/releases/download/v${ROBOT_VERSION}/robot.jar"
    if command -v curl >/dev/null 2>&1; then
      curl -L -o "${ROBOT_JAR_PATH}" "${ROBOT_URL}"
    elif command -v wget >/dev/null 2>&1; then
      wget -O "${ROBOT_JAR_PATH}" "${ROBOT_URL}"
    else
      echo "Error: Neither curl nor wget is available to download ROBOT automatically." >&2
      exit 1
    fi
  fi

  if [[ ! -s "${ROBOT_JAR_PATH}" ]]; then
    echo "Error: Failed to download robot.jar to ${ROBOT_JAR_PATH}." >&2
    exit 1
  fi

  if [[ ! -f "${ROBOT_WRAPPER_PATH}" ]]; then
    cat > "${ROBOT_WRAPPER_PATH}" <<'WRAP'
#!/usr/bin/env bash
set -euo pipefail
JAVA_BIN="${JAVA_BIN:-java}"
exec "${JAVA_BIN}" ${ROBOT_JAVA_ARGS:-} -jar "$(dirname "$0")/robot.jar" "$@"
WRAP
    chmod +x "${ROBOT_WRAPPER_PATH}"
  fi

  ROBOT_BIN="${ROBOT_WRAPPER_PATH}"
  [[ -x "${ROBOT_BIN}" ]] && have_robot=true || true
fi

if [[ ${have_robot} == false ]]; then
  echo "Error: ROBOT is not available and could not be bootstrapped (ROBOT_BIN='${ROBOT_BIN}')." >&2
  exit 1
fi

if [[ ! -f "${LEFT}" ]]; then
  echo "Error: LEFT ontology not found: ${LEFT}" >&2
  exit 2
fi
if [[ ! -f "${RIGHT}" ]]; then
  echo "Error: RIGHT ontology not found: ${RIGHT}" >&2
  exit 2
fi

mkdir -p "$(dirname "${OUT}")"

echo "[diff_with_robot] ROBOT: ${ROBOT_BIN}"
echo "[diff_with_robot] LEFT:  ${LEFT}"
echo "[diff_with_robot] RIGHT: ${RIGHT}"
echo "[diff_with_robot] OUT:   ${OUT}"
echo "[diff_with_robot] FORMAT:${ROBOT_DIFF_FORMAT}"

set -x
"${ROBOT_BIN}" diff \
  --left  "${LEFT}" \
  --right "${RIGHT}" \
  --format "${ROBOT_DIFF_FORMAT}" \
  --output "${OUT}"
set +x

echo "[diff_with_robot] Done. Wrote: ${OUT}"
