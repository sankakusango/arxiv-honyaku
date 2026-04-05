#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_ARXIV_REF="1706.03762"
COLOR_RESET=$'\033[0m'
COLOR_GREEN=$'\033[32m'
COLOR_YELLOW=$'\033[33m'
COLOR_BLUE=$'\033[34m'
COLOR_RED=$'\033[31m'


print_info() {
  printf '%sINFO%s %s\n' "${COLOR_BLUE}" "${COLOR_RESET}" "$1"
}


print_ok() {
  printf '%sOK%s %s\n' "${COLOR_GREEN}" "${COLOR_RESET}" "$1"
}


print_warning() {
  printf '%sWARNING%s %s\n' "${COLOR_YELLOW}" "${COLOR_RESET}" "$1"
}


print_error() {
  printf '%sERROR%s %s\n' "${COLOR_RED}" "${COLOR_RESET}" "$1" >&2
}


handle_error() {
  local exit_code="$1"
  local line_number="$2"
  local command_text="$3"
  print_error \
    "The script stopped at line ${line_number} with exit code ${exit_code}. Command: ${command_text}"
  exit "${exit_code}"
}


usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_ollama_gemma_translate.sh [--force] [ARXIV_REF]

Example:
  bash scripts/run_ollama_gemma_translate.sh 1706.03762
  bash scripts/run_ollama_gemma_translate.sh --force 1706.03762

This script:
1. Detects GPU VRAM size (if available).
2. Chooses a TranslateGemma size that fits the machine.
3. Starts Docker Compose services (web + ollama).
4. Pulls the selected model on ollama.
5. Runs arxiv-honyaku CLI and translates one arXiv paper to Japanese.
USAGE
}


normalize_arxiv_ref_for_path() {
  python3 -c \
    'import sys; sys.path.insert(0, sys.argv[1]); from arxiv_honyaku.core.arxiv_id import normalize_arxiv_ref, path_safe_arxiv_id; print(path_safe_arxiv_id(normalize_arxiv_ref(sys.argv[2])))' \
    "${PROJECT_ROOT}/src" \
    "$1"
}


completed_run_exists() {
  local safe_arxiv_id="$1"
  local run_dir="${PROJECT_ROOT}/runs/${safe_arxiv_id}"
  [[ -d "${run_dir}/translated" ]] || return 1
  compgen -G "${run_dir}/build/*.pdf" >/dev/null
}


detect_total_vram_mb() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "0"
    return
  fi

  local raw
  if ! raw="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null)"; then
    echo "0"
    return
  fi

  local total=0
  local line
  while IFS= read -r line; do
    line="${line//[[:space:]]/}"
    if [[ -z "${line}" ]]; then
      continue
    fi
    if [[ "${line}" =~ ^[0-9]+$ ]]; then
      total=$((total + line))
    fi
  done <<<"${raw}"
  echo "${total}"
}


select_model_candidates() {
  local vram_mb="$1"
  if ((vram_mb >= 24576)); then
    echo "translategemma:27b"
    return
  fi
  if ((vram_mb >= 12288)); then
    echo "translategemma:12b"
    return
  fi
  echo "translategemma:4b"
}


pull_selected_model() {
  local -n compose_cmd_ref="$1"
  shift

  local model
  local compose_output
  for model in "$@"; do
    echo "Trying model: ${model}" >&2
    if compose_output="$("${compose_cmd_ref[@]}" exec -T ollama ollama pull "${model}" 2>&1)"; then
      echo "${compose_output}" >&2
      echo "${model}"
      return
    fi
    echo "${compose_output}" >&2
  done
  return 1
}


wait_for_ollama() {
  local -a compose_cmd=("$@")
  local max_attempts=90
  local attempt
  for ((attempt = 1; attempt <= max_attempts; attempt++)); do
    if "${compose_cmd[@]}" exec -T ollama ollama list >/dev/null 2>&1; then
      return
    fi
    sleep 2
  done
  print_error "Failed to connect to ollama service."
  exit 1
}


start_compose_stack() {
  local -n compose_cmd_ref="$1"
  local compose_output
  if compose_output="$("${compose_cmd_ref[@]}" up --build -d 2>&1)"; then
    echo "${compose_output}" >&2
    return 0
  fi
  echo "${compose_output}" >&2
  return 1
}


trap 'handle_error "$?" "${LINENO}" "${BASH_COMMAND}"' ERR


main() {
  local force=0
  local arxiv_ref=""
  while (($# > 0)); do
    case "$1" in
      -h|--help)
        usage
        return
        ;;
      --force)
        force=1
        shift
        ;;
      -*)
        print_error "Unknown option: $1"
        usage
        exit 1
        ;;
      *)
        if [[ -n "${arxiv_ref}" ]]; then
          print_error "Only one arXiv reference can be provided."
          usage
          exit 1
        fi
        arxiv_ref="$1"
        shift
        ;;
    esac
  done
  if [[ -z "${arxiv_ref}" ]]; then
    arxiv_ref="${DEFAULT_ARXIV_REF}"
  fi

  local safe_arxiv_id
  safe_arxiv_id="$(normalize_arxiv_ref_for_path "${arxiv_ref}")"
  if ((force == 0)) && completed_run_exists "${safe_arxiv_id}"; then
    print_ok \
      "A compiled PDF already exists for ${arxiv_ref}. Skipping all work. Use --force to rerun."
    return
  fi

  local total_vram_mb
  total_vram_mb="$(detect_total_vram_mb)"
  local model_candidates
  model_candidates="$(select_model_candidates "${total_vram_mb}")"
  local -a candidates=()
  read -r -a candidates <<<"${model_candidates}"

  cd "${PROJECT_ROOT}"

  local -a compose_cmd=(
    docker
    compose
    -f
    docker-compose.yml
    -f
    docker-compose.ollama.yml
  )
  if ((total_vram_mb > 0)); then
    print_info \
      "Host GPU detected. If Docker runtime is configured for GPU, Ollama may use acceleration."
  else
    print_warning "No usable GPU was detected on the host. Ollama will run on CPU."
  fi

  print_info "Detected VRAM (MB): ${total_vram_mb}"
  print_info "Model candidates: ${model_candidates}"
  print_info "Target arXiv ref: ${arxiv_ref}"

  if ! start_compose_stack compose_cmd; then
    print_error "Failed to start Docker Compose."
    exit 1
  fi
  wait_for_ollama "${compose_cmd[@]}"

  local selected_model
  print_info "Pulling the selected TranslateGemma model. The first download can take a while."
  if ! selected_model="$(pull_selected_model compose_cmd "${candidates[@]}")"; then
    print_error "Failed to pull the selected TranslateGemma model."
    exit 1
  fi

  print_ok "Selected model: ${selected_model}"
  local -a cli_args=(
    exec
    -T
    web
    env
    "ARXIV_HONYAKU_LLM_MODEL=${selected_model}"
    arxiv-honyaku
    --config
    /app/arxiv-honyaku.toml
  )
  if ((force == 1)); then
    cli_args+=(--force)
  fi
  cli_args+=("${arxiv_ref}")
  if ! "${compose_cmd[@]}" "${cli_args[@]}"; then
    print_error "Translation command failed inside the web container."
    exit 1
  fi

  print_ok "Completed. Check output under: ${PROJECT_ROOT}/runs"
}


main "$@"
