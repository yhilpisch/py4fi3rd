#!/usr/bin/env bash
set -Eeuo pipefail

# Run the complete companion-repository validation from a normal CLI shell.
# Override VALIDATION_LOG, VALIDATION_TIMEOUT, or PYTHON_BIN when needed.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VALIDATION_TIMEOUT="${VALIDATION_TIMEOUT:-300}"
VALIDATION_LOG="${VALIDATION_LOG:-${ROOT_DIR}/colab-validation.log}"

mkdir -p "$(dirname "${VALIDATION_LOG}")"
: > "${VALIDATION_LOG}"

run_logged() {
  local label="$1"
  shift

  printf '\n== %s ==\n' "${label}" | tee -a "${VALIDATION_LOG}"
  set +e
  "$@" 2>&1 | tee -a "${VALIDATION_LOG}"
  local command_status="${PIPESTATUS[0]}"
  set -e

  if (( command_status != 0 )); then
    printf '!! %s failed with exit status %s\n' "${label}" "${command_status}" \
      | tee -a "${VALIDATION_LOG}"
    return "${command_status}"
  fi

  printf '== %s: OK ==\n' "${label}" | tee -a "${VALIDATION_LOG}"
}

cd "${ROOT_DIR}"
printf 'Validation root: %s\n' "${ROOT_DIR}" | tee -a "${VALIDATION_LOG}"
printf 'Validation log: %s\n' "${VALIDATION_LOG}" | tee -a "${VALIDATION_LOG}"

run_logged \
  "Colab-style bootstrap, representative scripts, and all notebooks" \
  "${PYTHON_BIN}" -u tools/validate_colab.py \
  --scripts --all-notebooks --timeout "${VALIDATION_TIMEOUT}"

run_logged \
  "All runnable chapter, figure, and lab scripts" \
  "${PYTHON_BIN}" -u tools/validate_code.py \
  'chapters/ch*.py' 'figures/*.py' 'labs/*.py' \
  --timeout "${VALIDATION_TIMEOUT}"

printf '\nValidation complete: OK\n' | tee -a "${VALIDATION_LOG}"
