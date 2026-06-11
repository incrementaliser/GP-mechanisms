"""Causal SAE feature interventions reproducing the paper's Figure 4 setup."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Tuple

import pandas as pd
import torch
from nnsight import LanguageModel
from tqdm import tqdm
from transformers import AutoTokenizer

from gp_notebook.behavior import MODEL_NAME, continuation_tokens_for_condition
from gp_notebook.device import get_torch_device, live_mode_available
from gp_notebook.paths import FEATURE_RESULTS, PROJECT_ROOT

FeatureEdit = Tuple[int, int, float]
FeatureMap = Dict[str, List[FeatureEdit]]


def submodule_name_to_submodule(model_name: str, submodule_name: str, model: LanguageModel):
    """Resolve a submodule label like resid_3 to the underlying nnsight module."""
    if submodule_name == "embed":
        if "pythia" in model_name:
            return model.gpt_neox.embed_in
        if "gemma" in model_name:
            return model.model.embed_tokens
    submod_type, layer_idx = submodule_name.split("_")
    layer_idx = int(layer_idx)
    if "pythia" in model_name:
        if submod_type == "resid":
            return model.gpt_neox.layers[layer_idx]
        if submod_type == "attn":
            return model.gpt_neox.layers[layer_idx].attention
        if submod_type == "mlp":
            return model.gpt_neox.layers[layer_idx].mlp
    raise ValueError(f"Unsupported submodule {submodule_name} for {model_name}")


def _runtime_device() -> torch.device:
    """Return the device used for live SAE interventions."""
    return get_torch_device()


def _dictionary_module():
    """Import dictionary_learning once the feature-circuits-gp submodule is present."""
    submodule_root = PROJECT_ROOT / "feature-circuits-gp"
    if not (submodule_root / "dictionary_learning").exists():
        raise ModuleNotFoundError(
            "dictionary_learning not found. Run: git submodule update --init --recursive"
        )
    sys.path.insert(0, str(submodule_root))
    from dictionary_learning import dictionary

    return dictionary


def load_autoencoder(model_name: str, submodule_name: str):
    """Load a Pythia-70m sparse autoencoder checkpoint for one submodule."""
    dictionary = _dictionary_module()
    device = _runtime_device()
    if submodule_name == "embed":
        ae_path = (
            PROJECT_ROOT
            / "feature-circuits-gp/dictionaries/pythia-70m-deduped/embed/10_32768/ae.pt"
        )
        ae = dictionary.AutoEncoder(512, 32768).to(device)
    else:
        submod_type, layer_idx = submodule_name.split("_")
        ae_path = (
            PROJECT_ROOT
            / f"feature-circuits-gp/dictionaries/pythia-70m-deduped/{submod_type}_out_layer"
            f"{layer_idx}/10_32768/ae.pt"
        )
        ae = dictionary.AutoEncoder(512, 32768).to(device)
    ae.load_state_dict(torch.load(ae_path, map_location=device, weights_only=True))
    ae.eval()
    return ae


def saes_available() -> bool:
    """Return True when at least one Pythia SAE checkpoint is present locally."""
    probe = (
        PROJECT_ROOT
        / "feature-circuits-gp/dictionaries/pythia-70m-deduped/resid_out_layer0/10_32768/ae.pt"
    )
    return probe.exists()


def load_dictionaries(model_name: str, model: LanguageModel) -> dict:
    """Load all Pythia submodule SAE pairs used by the paper's causal analysis."""
    submodule_names = [
        "embed",
        *(f"{module}_{i}" for i in range(6) for module in ("attn", "mlp", "resid")),
    ]
    dictionaries = {}
    for name in tqdm(submodule_names, desc="Loading SAEs"):
        dictionaries[name] = (submodule_name_to_submodule(model_name, name, model), load_autoencoder(model_name, name))
    return dictionaries


def get_performance(
    model: LanguageModel,
    prompts: Iterable[str],
    indices: tuple[list[int], list[int]],
    dictionaries: dict,
    features: FeatureMap,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run forward passes with optional feature edits and return GP / non-GP probs."""
    model_out = model.embed_out
    use_inputs: dict = {}
    for submodule_name, (submodule, _ae) in dictionaries.items():
        use_inputs[submodule] = False

    gp_probs: list[float] = []
    non_gp_probs: list[float] = []
    device = _runtime_device()
    for prompt in prompts:
        with model.trace(prompt), torch.no_grad():
            for submodule_name, (submodule, ae) in dictionaries.items():
                if len(features.get(submodule_name, [])) == 0:
                    continue
                x = submodule.output if not use_inputs[submodule] else submodule.input
                if use_inputs[submodule]:
                    x = x[0][0]
                elif isinstance(x, tuple):
                    x = x[0]
                encoded = ae.encode(x)
                reconstruction = ae.decode(encoded)
                residual = x - reconstruction
                edited = torch.clone(encoded)
                positions, feature_ids, values = zip(*features[submodule_name])
                edited[:, list(positions), list(feature_ids)] = torch.tensor(
                    values, device=device, dtype=model.dtype
                )
                patched = ae.decode(edited)
                if use_inputs[submodule]:
                    submodule.input[0][0][:] = patched + residual
                elif isinstance(submodule.output, tuple):
                    submodule.output[0][:] = patched + residual
                else:
                    submodule.output = patched + residual
            logits = model_out.output.save()
        saved = logits.value if hasattr(logits, "value") else logits
        probs = torch.nn.functional.softmax(saved.squeeze(0), dim=-1)
        gp_idx, non_gp_idx = indices
        gp_probs.append(probs[-1, gp_idx].sum().item())
        non_gp_probs.append(probs[-1, non_gp_idx].sum().item())
    return torch.tensor(gp_probs), torch.tensor(non_gp_probs)


def build_feature_edits(
    feature_df: pd.DataFrame,
    condition: str,
    *,
    subject_amp: float = 2.0,
    object_amp: float = 2.0,
    clause_amp: float = 2.0,
    use_random: bool = False,
    seed: int = 0,
) -> FeatureMap:
    """Build intervention feature edits from annotated feature tables and slider amps."""
    rng = torch.Generator().manual_seed(seed)
    dict_size = 32768
    if condition == "NPZ":
        upweight = {"subject detector"}
        downweight = {"object detector"}
        clause_categories = {"end-clause detector", "clause detector", "end of clause detector"}
    else:
        upweight = {"object detector", "end of sentence detector"}
        downweight = {"subject detector", "CP verb detector"}
        clause_categories = set()

    edits: DefaultDict[str, List[FeatureEdit]] = defaultdict(list)
    annotated: DefaultDict[str, list[int]] = defaultdict(list)
    for _, row in feature_df.iterrows():
        submodule_name, feature_idx = row["Feature"].split("/")
        annotated[submodule_name].append(int(feature_idx))

    random_map: dict[tuple[str, int], int] = {}
    if use_random:
        for subname, idxs in annotated.items():
            candidates = torch.randperm(dict_size, generator=rng).tolist()
            cursor = 0
            for idx in idxs:
                while candidates[cursor] in annotated[subname]:
                    cursor += 1
                random_map[(subname, idx)] = candidates[cursor]
                cursor += 1

    for _, row in feature_df.iterrows():
        submodule_name, feature_idx = row["Feature"].split("/")
        feature_idx = int(feature_idx)
        position = int(row["Position"])
        category = row["Category"]
        chosen_idx = random_map.get((submodule_name, feature_idx), feature_idx)

        if category in clause_categories:
            edits[submodule_name].extend(
                [
                    (-3, chosen_idx, clause_amp),
                    (-2, chosen_idx, 0.0),
                    (-1, chosen_idx, 0.0),
                ]
            )
        elif category in downweight:
            edits[submodule_name].append((position, chosen_idx, 0.0))
        elif category in upweight:
            amp = subject_amp if "subject" in category else object_amp
            edits[submodule_name].append((position, chosen_idx, amp))
    return edits


def run_intervention_suite(
    df: pd.DataFrame,
    condition: str,
    *,
    subject_amp: float = 2.0,
    object_amp: float = 2.0,
    clause_amp: float = 2.0,
    use_random: bool = False,
    model: LanguageModel | None = None,
    tokenizer: AutoTokenizer | None = None,
) -> dict[str, float]:
    """Return mean GP/non-GP probabilities under one intervention setting."""
    if not saes_available():
        raise FileNotFoundError("Pythia SAE checkpoints are not available locally.")

    model_name = MODEL_NAME
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model is None:
        device = _runtime_device()
        model = LanguageModel(
            model_name,
            torch_dtype=torch.float32,
            device_map=str(device),
            dispatch=True,
        )

    dictionaries = load_dictionaries(model_name, model)
    feature_df = pd.read_csv(
        FEATURE_RESULTS / ("npz_features.csv" if condition == "NPZ" else "nps_features.csv")
    )
    tokens = continuation_tokens_for_condition(condition)
    gp_ids = [tokenizer(tok, add_special_tokens=False)["input_ids"][0] for tok in tokens.gp_tokens]
    non_gp_ids = [
        tokenizer(tok, add_special_tokens=False)["input_ids"][0] for tok in tokens.non_gp_tokens
    ]
    prompts = df[df["condition"] == condition]["sentence_ambiguous"].tolist()
    edits = build_feature_edits(
        feature_df,
        condition,
        subject_amp=subject_amp,
        object_amp=object_amp,
        clause_amp=clause_amp,
        use_random=use_random,
    )
    gp, non_gp = get_performance(model, prompts, (gp_ids, non_gp_ids), dictionaries, edits)
    return {
        "mean_p_gp": float(gp.mean()),
        "mean_p_non_gp": float(non_gp.mean()),
        "mean_diff": float((gp - non_gp).mean()),
    }


def paper_style_interventions(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce baseline, syntactic, and random interventions for NPZ and NPS."""
    rows: list[dict[str, float | str]] = []
    model_name = MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = _runtime_device()
    model = LanguageModel(model_name, torch_dtype=torch.float32, device_map=str(device), dispatch=True)
    dictionaries = load_dictionaries(model_name, model)

    for condition in ("NPZ", "NPS"):
        feature_df = pd.read_csv(
            FEATURE_RESULTS / ("npz_features.csv" if condition == "NPZ" else "nps_features.csv")
        )
        tokens = continuation_tokens_for_condition(condition)
        gp_ids = [tokenizer(tok, add_special_tokens=False)["input_ids"][0] for tok in tokens.gp_tokens]
        non_gp_ids = [
            tokenizer(tok, add_special_tokens=False)["input_ids"][0] for tok in tokens.non_gp_tokens
        ]
        prompts = df[df["condition"] == condition]["sentence_ambiguous"].tolist()
        for label, random_flag in (
            ("baseline", None),
            ("syntactic", False),
            ("random", True),
        ):
            if label == "baseline":
                edits: FeatureMap = defaultdict(list)
            else:
                edits = build_feature_edits(feature_df, condition, use_random=random_flag)
            gp, non_gp = get_performance(model, prompts, (gp_ids, non_gp_ids), dictionaries, edits)
            rows.append(
                {
                    "condition": condition,
                    "intervention": label,
                    "mean_p_gp": float(gp.mean()),
                    "mean_p_non_gp": float(non_gp.mean()),
                    "mean_diff": float((gp - non_gp).mean()),
                }
            )
    return pd.DataFrame(rows)
