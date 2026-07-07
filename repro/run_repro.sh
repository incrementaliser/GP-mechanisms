#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/repro/outputs"
LOGS="$OUT/logs"
MODEL_NAME="${REPRO_MODEL_NAME:-EleutherAI/pythia-70m-deduped}"
READINGCOMP_MODEL="${REPRO_READINGCOMP_MODEL:-google/gemma-2-2b}"

mkdir -p "$LOGS"
cd "$ROOT" || exit 1
export PYTHONPATH="$ROOT/feature-circuits-gp:${PYTHONPATH:-}"

record_run() {
  local name="$1"
  local state="$2"
  local command_text="$3"
  local message="${4:-}"
  uv run python repro/compare_results.py record-run \
    --result "$name" \
    --state "$state" \
    --run-command "$command_text" \
    --message "$message" >/dev/null
}

refresh_comparison() {
  uv run python repro/compare_results.py compare >/dev/null
}

cuda_available() {
  uv run python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1
}

skip_without_cuda() {
  local name="$1"
  local command_text="$2"
  if cuda_available; then
    return 1
  fi
  record_run "$name" "skipped" "$command_text" \
    "CUDA is not available in this shell; run from a GPU job/session to regenerate."
  refresh_comparison
  echo "Skipping $name: CUDA is not available in this shell."
  return 0
}

run_logged() {
  local name="$1"
  shift
  local command_text="$*"
  echo "==> $name"
  echo "Command: $command_text"
  if "$@" 2>&1 | tee "$LOGS/$name.log"; then
    record_run "$name" "success" "$command_text"
    return 0
  fi
  record_run "$name" "failed" "$command_text" "Command failed; inspect repro/outputs/logs/$name.log."
  return 1
}

run_behavior() {
  skip_without_cuda behavior "uv run python behavioral_evaluation.py --model_name $MODEL_NAME" && return 0
  run_logged behavior uv run python behavioral_evaluation.py --model_name "$MODEL_NAME"
  uv run python repro/compare_results.py collect --target behavior >/dev/null
  refresh_comparison
}

run_causal() {
  skip_without_cuda causal "uv run python causal_analysis.py --model_name $MODEL_NAME" && return 0
  local sae_probe="feature-circuits-gp/dictionaries/pythia-70m-deduped/resid_out_layer0/10_32768/ae.pt"
  if [[ ! -f "$sae_probe" ]]; then
    record_run causal "skipped" "uv run python causal_analysis.py --model_name $MODEL_NAME" \
      "Missing Pythia SAE checkpoint at $sae_probe."
    refresh_comparison
    echo "Skipping causal: missing $sae_probe"
    return 0
  fi
  run_logged causal uv run python causal_analysis.py --model_name "$MODEL_NAME"
  uv run python repro/compare_results.py collect --target causal >/dev/null
  refresh_comparison
}

run_probe() {
  skip_without_cuda probe "uv run python parseprobe_behavior.py" && return 0
  local missing=0
  for probe in standalone_probes/embeddings.pt standalone_probes/layer0.pt standalone_probes/layer1.pt standalone_probes/layer2.pt standalone_probes/layer3.pt standalone_probes/layer4.pt standalone_probes/layer5.pt; do
    if [[ ! -f "$probe" ]]; then
      missing=1
    fi
  done
  if [[ "$missing" -ne 0 ]]; then
    record_run probe "skipped" "uv run python parseprobe_behavior.py" \
      "Missing trained standalone_probes/*.pt files."
    refresh_comparison
    echo "Skipping probe: missing trained standalone_probes/*.pt files."
    return 0
  fi
  run_logged probe uv run python parseprobe_behavior.py
  uv run python repro/compare_results.py collect --target probe >/dev/null
  refresh_comparison
}

run_readingcomp() {
  skip_without_cuda readingcomp "uv run python readingcomp_evaluation.py --model $READINGCOMP_MODEL --dataset data_csv/garden_path_samelen_readingcomp.csv" && return 0
  run_logged readingcomp uv run python readingcomp_evaluation.py --model "$READINGCOMP_MODEL" \
    --dataset data_csv/garden_path_samelen_readingcomp.csv
  refresh_comparison
}

run_compare() {
  uv run python repro/compare_results.py collect --target all >/dev/null
  refresh_comparison
  echo "Wrote repro/outputs/status.json and repro/outputs/comparison_summary.json"
}

usage() {
  echo "Usage: repro/run_repro.sh [behavior|causal|probe|readingcomp|core|all|compare]"
  echo
  echo "Environment variables:"
  echo "  REPRO_MODEL_NAME            Model for behavior/causal scripts (default: $MODEL_NAME)"
  echo "  REPRO_READINGCOMP_MODEL     Model for readingcomp script (default: $READINGCOMP_MODEL)"
}

target="${1:-core}"
case "$target" in
  behavior)
    run_behavior
    ;;
  causal)
    run_causal
    ;;
  probe)
    run_probe
    ;;
  readingcomp)
    run_readingcomp
    ;;
  core)
    run_behavior
    run_causal
    ;;
  all)
    run_behavior
    run_causal
    run_probe
    run_readingcomp
    ;;
  compare)
    run_compare
    ;;
  *)
    usage
    exit 2
    ;;
esac
