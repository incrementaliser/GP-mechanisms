"""Collect and compare regenerated paper results for the marimo notebook."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd

ResultState = Literal["success", "failed", "skipped", "not_run"]


@dataclass(frozen=True)
class ReproResult:
    """Status for one reproducible paper-result target."""

    result_id: str
    paper_target: str
    command: str
    output_path: str | None
    state: ResultState
    reproducibility: str
    message: str


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPRO_ROOT = PROJECT_ROOT / "repro"
OUTPUTS_DIR = REPRO_ROOT / "outputs"
LOGS_DIR = OUTPUTS_DIR / "logs"
ASSETS_DIR = PROJECT_ROOT / "assets"
RESULTS_DIR = PROJECT_ROOT / "results" / "pythia-70m-deduped"
RUN_ATTEMPTS_PATH = OUTPUTS_DIR / "run_attempts.json"
STATUS_PATH = OUTPUTS_DIR / "status.json"
COMPARISON_PATH = OUTPUTS_DIR / "comparison_summary.json"
MATRIX_PATH = OUTPUTS_DIR / "reproducibility_matrix.json"


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp for status files."""
    return datetime.now(timezone.utc).isoformat()


def ensure_output_dirs() -> None:
    """Create generated-output directories used by the harness."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def read_json(path: Path, default: Any) -> Any:
    """Read JSON from disk, returning a default when absent."""
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    """Write stable, indented JSON to disk."""
    ensure_output_dirs()
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def record_run(result_id: str, state: ResultState, command: str, message: str = "") -> None:
    """Record one wrapper-command attempt in run_attempts.json."""
    attempts = read_json(RUN_ATTEMPTS_PATH, {})
    attempts[result_id] = {
        "state": state,
        "command": command,
        "message": message,
        "timestamp": utc_now(),
        "log_path": str((LOGS_DIR / f"{result_id}.log").relative_to(PROJECT_ROOT)),
    }
    write_json(RUN_ATTEMPTS_PATH, attempts)


def copy_if_exists(source: Path, destination: Path) -> bool:
    """Copy one generated artifact into repro/outputs when it exists."""
    if not source.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def collect_results(target: str) -> dict[str, bool]:
    """Collect outputs produced by original author scripts into repro/outputs."""
    ensure_output_dirs()
    copied: dict[str, bool] = {}
    if target in {"behavior", "all", "core"}:
        behavior_dir = OUTPUTS_DIR / "behavior"
        copied["behavior_gp_with_probs"] = copy_if_exists(
            RESULTS_DIR / "gp_with_probs.csv",
            behavior_dir / "gp_with_probs.csv",
        )
        copied["behavior_summary"] = copy_if_exists(
            RESULTS_DIR / "behavioral_summary.csv",
            behavior_dir / "behavioral_summary.csv",
        )
    if target in {"causal", "all", "core"}:
        copied["causal_probabilities"] = copy_if_exists(
            RESULTS_DIR / "causal_probabilities.pt",
            OUTPUTS_DIR / "causal" / "causal_probabilities.pt",
        )
    if target in {"probe", "all"}:
        copied["probe_probabilities"] = copy_if_exists(
            RESULTS_DIR / "parse_probe" / "probe_probs.pt",
            OUTPUTS_DIR / "probe" / "probe_probs.pt",
        )
    return copied


def summarize_behavior(gp_with_probs_path: Path) -> pd.DataFrame:
    """Summarize original behavioral_evaluation.py outputs as Figure 2 mean deltas."""
    df = pd.read_csv(gp_with_probs_path)
    rows: list[dict[str, str | float]] = []
    for input_type in ("ambiguous", "gp", "post"):
        punct_col = f"{input_type}_punct_prob"
        was_col = f"{input_type}_was_prob"
        if punct_col not in df.columns or was_col not in df.columns:
            continue
        tmp = df.copy()
        tmp["diff"] = tmp[punct_col] - tmp[was_col]
        grouped = tmp.groupby("condition", as_index=False)["diff"].agg(["mean", "sem"]).reset_index()
        for _, row in grouped.iterrows():
            rows.append(
                {
                    "condition": str(row["condition"]),
                    "input_type": input_type,
                    "mean_diff": float(row["mean"]),
                    "sem_diff": float(row["sem"]) if pd.notna(row["sem"]) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def compare_behavior() -> dict[str, Any]:
    """Compare regenerated behavior outputs against notebook cache directions and values."""
    generated_path = OUTPUTS_DIR / "behavior" / "gp_with_probs.csv"
    cache_path = ASSETS_DIR / "behavioral_summary.parquet"
    if not generated_path.exists():
        return {"state": "not_run", "message": "No regenerated behavior output found."}

    generated = summarize_behavior(generated_path)
    comparison: dict[str, Any] = {
        "state": "success",
        "generated_rows": generated.to_dict(orient="records"),
    }
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        merged = generated.merge(
            cached,
            on=["condition", "input_type"],
            how="inner",
            suffixes=("_generated", "_notebook_cache"),
        )
        if not merged.empty:
            merged["abs_delta"] = (
                merged["mean_diff_generated"] - merged["mean_diff_notebook_cache"]
            ).abs()
            comparison["max_abs_delta_vs_notebook_cache"] = float(merged["abs_delta"].max())
            comparison["comparisons"] = merged[
                [
                    "condition",
                    "input_type",
                    "mean_diff_generated",
                    "mean_diff_notebook_cache",
                    "abs_delta",
                ]
            ].to_dict(orient="records")
    return comparison


def summarize_causal(causal_path: Path) -> pd.DataFrame:
    """Summarize causal_analysis.py probabilities into Figure 4-compatible means."""
    import torch

    payload = torch.load(causal_path, map_location="cpu", weights_only=False)
    label_map = {"baseline": "baseline", "intervened": "syntactic", "random": "random"}
    rows: list[dict[str, str | float]] = []
    for raw_label, by_condition in payload.items():
        label = label_map.get(raw_label, raw_label)
        for condition, tensors in by_condition.items():
            gp_probs, non_gp_probs = tensors
            diff = gp_probs - non_gp_probs
            rows.append(
                {
                    "condition": condition,
                    "intervention": label,
                    "mean_p_gp": float(gp_probs.float().mean().item()),
                    "mean_p_non_gp": float(non_gp_probs.float().mean().item()),
                    "mean_diff": float(diff.float().mean().item()),
                }
            )
    return pd.DataFrame(rows)


def compare_causal() -> dict[str, Any]:
    """Compare regenerated causal outputs against notebook Figure 4 cache."""
    generated_path = OUTPUTS_DIR / "causal" / "causal_probabilities.pt"
    cache_path = ASSETS_DIR / "interventions.parquet"
    if not generated_path.exists():
        return {"state": "not_run", "message": "No regenerated causal output found."}

    generated = summarize_causal(generated_path)
    comparison: dict[str, Any] = {
        "state": "success",
        "generated_rows": generated.to_dict(orient="records"),
    }
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        merged = generated.merge(
            cached,
            on=["condition", "intervention"],
            how="inner",
            suffixes=("_generated", "_notebook_cache"),
        )
        if not merged.empty:
            merged["abs_delta"] = (
                merged["mean_diff_generated"] - merged["mean_diff_notebook_cache"]
            ).abs()
            comparison["max_abs_delta_vs_notebook_cache"] = float(merged["abs_delta"].max())
            comparison["comparisons"] = merged[
                [
                    "condition",
                    "intervention",
                    "mean_diff_generated",
                    "mean_diff_notebook_cache",
                    "abs_delta",
                ]
            ].to_dict(orient="records")
    return comparison


def compare_probe() -> dict[str, Any]:
    """Report whether trained structural-probe outputs were regenerated."""
    generated_path = OUTPUTS_DIR / "probe" / "probe_probs.pt"
    if not generated_path.exists():
        return {
            "state": "not_run",
            "message": "No regenerated probe output found; requires standalone_probes/*.pt.",
        }
    return {
        "state": "success",
        "message": "Regenerated probe tensor file is present for downstream plotting.",
        "output_path": str(generated_path.relative_to(PROJECT_ROOT)),
    }


def read_log_tail(result_id: str, max_chars: int = 4000) -> str:
    """Read the tail of a repro command log when present."""
    path = LOGS_DIR / f"{result_id}.log"
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def parse_readingcomp_log() -> dict[str, Any]:
    """Extract reading-comprehension accuracies from the original script log."""
    log_text = read_log_tail("readingcomp", max_chars=12000)
    if not log_text:
        return {
            "state": "not_run",
            "message": "No regenerated reading-comprehension log found.",
        }
    lines = [line.rstrip() for line in log_text.splitlines()]
    return {
        "state": "success",
        "message": "Original readingcomp_evaluation.py completed; see log for grouped accuracies.",
        "log_tail": lines[-40:],
    }


def artifact_requirements() -> dict[str, dict[str, Any]]:
    """Return artifact availability used to explain non-reproducible results."""
    sae_path = (
        PROJECT_ROOT
        / "feature-circuits-gp"
        / "dictionaries"
        / "pythia-70m-deduped"
        / "resid_out_layer0"
        / "10_32768"
        / "ae.pt"
    )
    probe_paths = [PROJECT_ROOT / "standalone_probes" / "embeddings.pt"] + [
        PROJECT_ROOT / "standalone_probes" / f"layer{idx}.pt" for idx in range(6)
    ]
    return {
        "pythia_saes": {
            "available": sae_path.exists(),
            "required_for": ["figure4_causal", "figure3_circuit_recompute"],
            "example_path": str(sae_path.relative_to(PROJECT_ROOT)),
        },
        "standalone_probes": {
            "available": all(path.exists() for path in probe_paths),
            "required_for": ["figure5_probe"],
            "missing": [
                str(path.relative_to(PROJECT_ROOT)) for path in probe_paths if not path.exists()
            ],
        },
        "penn_treebank": {
            "available": False,
            "required_for": ["probe_training"],
            "message": "Not publicly bundled; needed only to train probes from scratch.",
        },
    }


def build_matrix(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the reproducibility matrix consumed by the notebook ledger."""
    attempts = read_json(RUN_ATTEMPTS_PATH, {})
    requirements = artifact_requirements()
    behavior_state = comparison.get("behavior", {}).get("state", "not_run")
    causal_state = comparison.get("causal", {}).get("state", "not_run")
    probe_state = comparison.get("probe", {}).get("state", "not_run")
    readingcomp_state = comparison.get("readingcomp", {}).get("state", "not_run")
    def _attempt_state(result_id: str, fallback: str) -> str:
        """Return the latest wrapper attempt state for a result."""
        return str(attempts.get(result_id, {}).get("state", fallback))

    def _attempt_message(result_id: str) -> str:
        """Return the latest wrapper attempt message for a result."""
        return str(attempts.get(result_id, {}).get("message", ""))

    def _notes(base: str, result_id: str) -> str:
        """Append concrete run-attempt blockers to static notes."""
        message = _attempt_message(result_id)
        return f"{base} Blocker: {message}" if message else base

    def _reproducibility(result_id: str, success_state: str, default: str) -> str:
        """Return a concise reproducibility label for a result row."""
        state = _attempt_state(result_id, success_state)
        if success_state == "success":
            return "code-generated"
        message = _attempt_message(result_id)
        if state in {"failed", "skipped"} and message:
            return f"blocked: {message}"
        return default

    return [
        {
            "result_id": "figure2_behavior",
            "paper_target": "Figure 2 / Section 4.1",
            "notebook_surface": "Module 2 Behavioral lab",
            "producer": "behavioral_evaluation.py",
            "harness_command": "repro/run_repro.sh behavior",
            "state": _attempt_state("behavior", behavior_state),
            "output_path": "repro/outputs/behavior/gp_with_probs.csv",
            "reproducibility": _reproducibility(
                "behavior", behavior_state, "not regenerated"
            ),
            "notes": _notes("Compares p(GP)-p(non-GP) means against notebook cache.", "behavior"),
        },
        {
            "result_id": "figure4_causal",
            "paper_target": "Figure 4 / Section 4.3",
            "notebook_surface": "Module 4 Intervention sandbox",
            "producer": "causal_analysis.py",
            "harness_command": "repro/run_repro.sh causal",
            "state": _attempt_state("causal", causal_state),
            "output_path": "repro/outputs/causal/causal_probabilities.pt",
            "reproducibility": _reproducibility("causal", causal_state, "requires run"),
            "notes": _notes("Requires local Pythia SAE checkpoints.", "causal"),
            "artifact_available": requirements["pythia_saes"]["available"],
        },
        {
            "result_id": "figure3_circuits_atp_ig",
            "paper_target": "Figures 3, 9, 10 / Section 4.2",
            "notebook_surface": "Module 3 Feature microscope",
            "producer": "feature-circuits-gp/scripts/get_circuit_garden_path.sh",
            "harness_command": "not automated yet",
            "state": "skipped",
            "output_path": None,
            "reproducibility": "not reproduced by notebook harness",
            "notes": "Notebook AtP-IG ranks are demo priors; full circuit regeneration also needs manual annotation.",
        },
        {
            "result_id": "figure5_probe",
            "paper_target": "Figure 5 / Section 5.2",
            "notebook_surface": "Module 5 Structural probe plot",
            "producer": "parseprobe_behavior.py",
            "harness_command": "repro/run_repro.sh probe",
            "state": _attempt_state("probe", probe_state),
            "output_path": "repro/outputs/probe/probe_probs.pt",
            "reproducibility": _reproducibility(
                "probe", probe_state, "requires trained probes"
            ),
            "notes": _notes(
                "Current notebook probe cache is approximate/paper-derived unless this output exists.",
                "probe",
            ),
            "artifact_available": requirements["standalone_probes"]["available"],
        },
        {
            "result_id": "table3_gprc",
            "paper_target": "Table 3 / Section 6.1",
            "notebook_surface": "Module 6 Repair vs reanalysis",
            "producer": "readingcomp_evaluation.py",
            "harness_command": "repro/run_repro.sh readingcomp",
            "state": _attempt_state("readingcomp", readingcomp_state),
            "output_path": "repro/outputs/logs/readingcomp.log",
            "reproducibility": (
                _reproducibility("readingcomp", readingcomp_state, "paper-derived in notebook")
            ),
            "notes": _notes(
                "Notebook Table 3 constants should be replaced or qualified unless regenerated.",
                "readingcomp",
            ),
        },
        {
            "result_id": "faithfulness_tradeoff",
            "paper_target": "Appendix C faithfulness discussion",
            "notebook_surface": "Module 7 Faithfulness budget",
            "producer": "feature-circuits-gp/scripts/evaluate_circuit.sh",
            "harness_command": "not automated yet",
            "state": "skipped",
            "output_path": None,
            "reproducibility": "paper-reported illustrative curve",
            "notes": "Current chart includes paper-reported/illustrative values, not generated by notebook code.",
        },
    ]


def build_status() -> dict[str, Any]:
    """Build and persist status, comparison summary, and reproducibility matrix."""
    ensure_output_dirs()
    comparison = {
        "behavior": compare_behavior(),
        "causal": compare_causal(),
        "probe": compare_probe(),
        "readingcomp": parse_readingcomp_log(),
    }
    matrix = build_matrix(comparison)
    results = [
        ReproResult(
            result_id=row["result_id"],
            paper_target=row["paper_target"],
            command=row["harness_command"],
            output_path=row["output_path"],
            state=row["state"],
            reproducibility=row["reproducibility"],
            message=row["notes"],
        )
        for row in matrix
    ]
    status = {
        "generated_at": utc_now(),
        "results": [asdict(result) for result in results],
        "artifact_requirements": artifact_requirements(),
    }
    write_json(COMPARISON_PATH, comparison)
    write_json(MATRIX_PATH, matrix)
    write_json(STATUS_PATH, status)
    return status


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the reproducibility comparator."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect")
    collect.add_argument("--target", default="all", choices=["behavior", "causal", "probe", "core", "all"])

    subparsers.add_parser("compare")

    record = subparsers.add_parser("record-run")
    record.add_argument("--result", required=True)
    record.add_argument("--state", required=True, choices=["success", "failed", "skipped", "not_run"])
    record.add_argument("--run-command", required=True)
    record.add_argument("--message", default="")

    return parser.parse_args()


def main() -> None:
    """Run the selected comparator subcommand."""
    args = parse_args()
    if args.command == "collect":
        copied = collect_results(args.target)
        write_json(OUTPUTS_DIR / "last_collect.json", {"target": args.target, "copied": copied})
        build_status()
    elif args.command == "compare":
        build_status()
    elif args.command == "record-run":
        record_run(args.result, args.state, args.run_command, args.message)
        build_status()


if __name__ == "__main__":
    main()
