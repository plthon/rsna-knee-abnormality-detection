# %% [markdown]
# # Frozen local-LLM report labels
#
# Private, GPU-backed, internet-disabled extraction with the frozen
# Qwen2.5-3B-Instruct contract. Raw reports and row-level labels remain inside
# Kaggle; only aggregate-safe evidence may be retrieved.

# %%
import base64
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from importlib.resources import files
from pathlib import Path
from typing import Any, cast

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
STAGE = "__RSNA_KNEE_REPORT_LLM_STAGE__"
COMPETITION_ROOT = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
RULE_LABEL_ROOTS = (
    Path("/kaggle/input/kernels/plthon/rsna-knee-report-labels"),
    Path("/kaggle/input/rsna-knee-report-labels"),
)
WORK_ROOT = Path("/kaggle/working")
PACKAGE_ARCHIVE = WORK_ROOT / "rsna_knee_source.zip"
PACKAGE_ARCHIVE.write_bytes(base64.b64decode("__RSNA_KNEE_PACKAGE_ARCHIVE__"))
sys.path.insert(0, str(PACKAGE_ARCHIVE))
notebook_started = time.monotonic()

# %%
import torch
from transformers import (  # type: ignore[import-not-found]
    AutoModelForCausalLM,
    AutoTokenizer,
)

from rsna_knee.competition_contract import load_verified_submission_contract
from rsna_knee.public_label_audit import resolve_dataset_root
from rsna_knee.report_llm import (
    GenerationRequest,
    GenerationResponse,
    compute_stability,
    load_llm_contract,
    read_non_gold_study_uids,
    read_rule_label_csv,
    run_llm_extraction,
    select_hashed_reports,
    truncate_token_ids,
    verify_model_asset,
    write_llm_artifacts,
)
from rsna_knee.reports import read_report_csv

competition_contract = load_verified_submission_contract()
llm_contract_bytes = (
    files("rsna_knee").joinpath("report-llm-extractor-v1.json").read_bytes()
)
llm_contract = load_llm_contract(
    json.loads(llm_contract_bytes),
    expected_target_order=competition_contract.target_names,
)
model_asset = cast(dict[str, object], llm_contract.model_asset_identity)
model_root = Path(cast(str, model_asset["mount_path"]))
verified_model_files = verify_model_asset(model_root, llm_contract)

expected_runtime = cast(dict[str, object], llm_contract.runtime)
transformers_version = importlib.metadata.version("transformers")
observed_runtime = {
    "python": ".".join(str(value) for value in sys.version_info[:3]),
    "torch": torch.__version__,
    "transformers": transformers_version,
    "cuda": torch.version.cuda,
}
for package, expected in expected_runtime.items():
    if package == "gpu":
        continue
    if observed_runtime.get(package) != expected:
        raise RuntimeError(
            f"Frozen runtime mismatch for {package}: "
            f"{observed_runtime.get(package)!r} != {expected!r}"
        )
if not torch.cuda.is_available():
    raise RuntimeError("Frozen report LLM extraction requires CUDA")
gpu_name = torch.cuda.get_device_name(0)
if cast(str, expected_runtime["gpu"]) not in gpu_name:
    raise RuntimeError(f"Frozen GPU mismatch: {gpu_name}")

# %%
report_rows = read_report_csv(
    COMPETITION_ROOT / "train.csv",
    requested_columns=("StudyInstanceUID", "Report"),
    target_names=competition_contract.target_names,
)
non_gold_study_uids = read_non_gold_study_uids(
    COMPETITION_ROOT / "train.csv", competition_contract.target_names
)
if STAGE == "smoke":
    selected_rows = select_hashed_reports(
        report_rows,
        non_gold_study_uids,
        salt="report-llm-smoke-v1",
        count=32,
    )
elif STAGE == "extract":
    selected_rows = report_rows
else:
    raise ValueError(f"Unsupported report LLM stage: {STAGE}")
expected_study_count = 32 if STAGE == "smoke" else 4_407
if len(selected_rows) != expected_study_count:
    raise RuntimeError(
        f"Frozen {STAGE} study count mismatch: "
        f"{len(selected_rows)} != {expected_study_count}"
    )

rule_label_path = (
    resolve_dataset_root(RULE_LABEL_ROOTS, "report-labels-v1.csv")
    / "report-labels-v1.csv"
)
rule_labels = read_rule_label_csv(
    rule_label_path,
    expected_sha256="6651f60b00e5bec292e8c98e7fa83c75e8aa5af5931414a9a68c8ce10a52d527",
    target_order=competition_contract.target_names,
)
selected_uids = {row.study_uid for row in selected_rows}
selected_rule_labels = tuple(
    label for label in rule_labels if label.study_uid in selected_uids
)

# %%
torch.cuda.reset_peak_memory_stats()
tokenizer = AutoTokenizer.from_pretrained(model_root, local_files_only=True)
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(
    model_root,
    local_files_only=True,
    dtype=torch.float16,
).to("cuda")
model.eval()


def _message_text(request: GenerationRequest) -> tuple[str, bool]:
    empty_messages = (
        request.messages[0],
        {
            "role": "user",
            "content": 'Report:\n"""\n\n"""\n\nReturn the required JSON object.',
        },
    )
    empty_tokens = tokenizer.apply_chat_template(
        list(empty_messages), tokenize=True, add_generation_prompt=True
    )
    max_input_tokens = cast(int, llm_contract.input_token_policy["max_input_tokens"])
    report_budget = max_input_tokens - len(empty_tokens)
    report_tokens = tokenizer.encode(
        request.normalized_report, add_special_tokens=False
    )
    fitted_tokens, input_truncated = truncate_token_ids(
        report_tokens, max_tokens=report_budget
    )
    fitted_report = tokenizer.decode(fitted_tokens, skip_special_tokens=False)
    messages = (
        request.messages[0],
        {
            "role": "user",
            "content": (
                f'Report:\n"""\n{fitted_report}\n"""\n\n'
                "Return the required JSON object."
            ),
        },
    )
    text = tokenizer.apply_chat_template(
        list(messages), tokenize=False, add_generation_prompt=True
    )
    if len(tokenizer.encode(text, add_special_tokens=False)) > max_input_tokens:
        raise RuntimeError("Frozen input token budget was exceeded")
    return cast(str, text), input_truncated


def generate(
    batch: tuple[GenerationRequest, ...],
) -> tuple[GenerationResponse, ...]:
    prepared = tuple(_message_text(request) for request in batch)
    encoded: Any = tokenizer(
        [text for text, _ in prepared],
        return_tensors="pt",
        padding=True,
        truncation=False,
    ).to("cuda")
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=cast(int, llm_contract.decoding["max_new_tokens"]),
            pad_token_id=tokenizer.pad_token_id,
        )
    input_width = encoded["input_ids"].shape[1]
    return tuple(
        GenerationResponse(
            tokenizer.decode(generated[index][input_width:], skip_special_tokens=True),
            input_truncated,
        )
        for index, (_, input_truncated) in enumerate(prepared)
    )


# %%
checkpoint_path = WORK_ROOT / "report-llm-checkpoint.json"
checkpoint_resumed_study_count = 0
if STAGE == "smoke":
    run_llm_extraction(
        selected_rows[:16],
        llm_contract,
        generate,
        checkpoint_path=checkpoint_path,
    )
    checkpoint_resumed_study_count = 16
results = run_llm_extraction(
    selected_rows,
    llm_contract,
    generate,
    checkpoint_path=checkpoint_path,
)
stability: dict[str, object] = {}
if STAGE == "extract":
    stability_rows = select_hashed_reports(
        report_rows,
        non_gold_study_uids,
        salt="report-llm-stability-v1",
        count=64,
    )
    first_stability = run_llm_extraction(
        stability_rows,
        llm_contract,
        generate,
        checkpoint_path=WORK_ROOT / "report-llm-stability-first.json",
    )
    second_stability = run_llm_extraction(
        stability_rows,
        llm_contract,
        generate,
        checkpoint_path=WORK_ROOT / "report-llm-stability-second.json",
    )
    stability = compute_stability(
        first_stability, second_stability, competition_contract.target_names
    )

elapsed = time.monotonic() - notebook_started
execution = {
    **observed_runtime,
    "gpu": gpu_name,
    "internet_disabled": True,
    "runtime_seconds": elapsed,
    "throughput_studies_per_second": len(selected_rows) / elapsed,
    "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
    "batch_size": llm_contract.decoding["default_batch_size"],
    "verified_model_file_count": len(verified_model_files),
    "checkpoint_resume_exercised": checkpoint_resumed_study_count > 0,
    "checkpoint_resumed_study_count": checkpoint_resumed_study_count,
}
artifact_paths = write_llm_artifacts(
    selected_rows,
    results,
    llm_contract,
    WORK_ROOT,
    expected_study_count=expected_study_count,
    competition_contract_sha256=competition_contract.sha256,
    llm_contract_sha256=hashlib.sha256(llm_contract_bytes).hexdigest(),
    source_provenance={
        "git_commit": "__RSNA_KNEE_GIT_COMMIT__",
        "git_dirty": json.loads("__RSNA_KNEE_GIT_DIRTY__"),
        "notebook_source_sha256": "__RSNA_KNEE_JUPYTEXT_SOURCE_SHA256__",
        "rendered_notebook_sha256": "__RSNA_KNEE_RENDERED_NOTEBOOK_SHA256__",
    },
    execution=execution,
    rule_labels=selected_rule_labels,
    stability=stability,
)
print(f"Completed {STAGE}: {len(selected_rows)} studies")
for artifact_path in artifact_paths.values():
    print(artifact_path)
