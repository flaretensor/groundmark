import gc
import hashlib
import json
import math
import os
import subprocess
import threading
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
from urllib.request import urlopen
from uuid import uuid4
from zipfile import ZipFile

from groundmark.eval.datasets import DRUID
from groundmark.eval.splits import SPLITS
from groundmark.nli import NLI_MODELS, NLIModel
from groundmark.sources import private_json

ARTIFACTS = Path(".artifacts/eval/druid-dev")
EXPERIMENT = "groundmark-checker-druid-dev-v2"
FOCUS = (
    ("lytang/MiniCheck-RoBERTa-Large", "74c8919647e61ed0f71bc177d94f10930f090068", "roberta-large"),
    (
        "lytang/MiniCheck-DeBERTa-v3-Large",
        "2f2d01a54fa022a7ffadb76260e1ea8bc88c82bb",
        "deberta-v3-large",
    ),
    ("lytang/MiniCheck-Flan-T5-Large", "96eafd01cee2d16cf81aaa2fb226b14f422a37b3", "flan-t5-large"),
)
FACTCG = ("yaxili96/FactCG-DeBERTa-v3-Large", "0430e3509dbd28d2dff7a117c0eae25359ff3e80", "factcg")
CANDIDATES = (*((model, rev, "nli") for model, rev in NLI_MODELS.items()), *FOCUS, FACTCG)
PUNKT_BASE = (
    "https://raw.githubusercontent.com/nltk/nltk_data/"
    "550b6625bcef1f2abff2ff770a5a0d272c9c6b2a/packages/tokenizers"
)
WEIGHT_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer*",
    "spm.model",
    "spiece.model",
    "merges.txt",
    "vocab.json",
    "*tokens*.json",
)


def local_punkt() -> None:
    import nltk

    path = Path(".cache/nltk")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)
    target = path / "tokenizers/punkt_tab/english"
    files = ("abbrev_types.txt", "collocations.tab", "ortho_context.tab", "sent_starters.txt")
    if not all((target / file).is_file() for file in files):
        target.mkdir(parents=True, exist_ok=True)
        with (
            urlopen(f"{PUNKT_BASE}/punkt_tab.zip", timeout=60) as response,
            ZipFile(BytesIO(response.read())) as archive,
        ):
            for filename in files:
                output = target / filename
                if not output.exists():
                    output.write_bytes(archive.read(f"punkt_tab/english/{filename}"))
    # MiniCheck checks for legacy Punkt data at import, then uses punkt_tab.
    legacy = path / "tokenizers/punkt"
    if not (legacy / "README").is_file():
        legacy.mkdir(parents=True, exist_ok=True)
        with (
            urlopen(f"{PUNKT_BASE}/punkt.zip", timeout=60) as response,
            ZipFile(BytesIO(response.read())) as archive,
        ):
            (legacy / "README").write_bytes(archive.read("punkt/README"))
    nltk.data.path[:] = [str(path.resolve())]


def snapshot(model: str, revision: str, kind: str) -> Path:
    from huggingface_hub import snapshot_download

    weight = "model.safetensors" if kind in ("nli", "factcg") else "pytorch_model.bin"
    return Path(
        snapshot_download(
            model,
            revision=revision,
            token=False,
            max_workers=3,
            allow_patterns=[weight, *WEIGHT_FILES],
        )
    )


@dataclass
class Measurements:
    lengths: list[int]
    actual_truncations: list[bool]
    predictions: list[int]
    scores: list[float]
    contradiction_scores: list[float | None]
    batches_ms: list[float]
    peak_allocated_bytes: int
    nvidia_peak_mib: int | None


def _minicheck(model_dir: Path, kind: str, dtype: str):
    import torch
    from minicheck.inference import Inferencer
    from minicheck.minicheck import MiniCheck
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    # Pin and load locally in the requested dtype. Official MiniCheck @
    # b58b9fa69acbd1015ec970fa65dd752413a053d2, minicheck/{minicheck,inference}.py,
    # constructs from mutable "main" and chooses its device placement itself.
    inferencer = Inferencer.__new__(Inferencer)
    inferencer.model_name = kind
    inferencer.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    loader = (
        AutoModelForSeq2SeqLM if kind == "flan-t5-large" else AutoModelForSequenceClassification
    )
    inferencer.model = (
        loader.from_pretrained(model_dir, local_files_only=True, dtype=getattr(torch, dtype))
        .to("cuda")
        .eval()
    )
    if kind == "flan-t5-large" and dtype == "bfloat16":

        def float_logits(_model, _inputs, output):
            # The official scorer ends by converting its softmax output to NumPy.
            # NumPy cannot represent PyTorch bfloat16 tensors.
            output.logits = output.logits.float()
            return output

        inferencer.model.register_forward_hook(float_logits)
    inferencer.max_model_len = 512 if kind == "roberta-large" else 2048
    inferencer.batch_size = 4
    checker = MiniCheck.__new__(MiniCheck)
    checker.model = inferencer
    original_inference = inferencer.inference

    def score(doc: str, claim: str):
        lengths = []

        def measured_inference(chunks, repeated):
            prefix = "predict: " if kind == "flan-t5-large" else ""
            formatted = [
                prefix + inferencer.tokenizer.eos_token.join((a, b))
                for a, b in zip(chunks, repeated, strict=True)
            ]
            lengths.extend(
                inferencer.tokenizer(
                    formatted, truncation=False, return_length=True, verbose=False
                )["length"]
            )
            return original_inference(chunks, repeated)

        inferencer.inference = measured_inference
        try:
            output = checker._score_inferencer([doc], [claim], None)
        finally:
            inferencer.inference = original_inference
        return output[1][0], max(lengths), max(lengths) > inferencer.max_model_len

    return score


def _factcg(model_dir: Path, dtype: str):
    import torch
    from nltk import sent_tokenize, word_tokenize
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model = (
        AutoModelForSequenceClassification.from_pretrained(
            model_dir, local_files_only=True, dtype=getattr(torch, dtype)
        )
        .to("cuda")
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    def score(doc: str, claim: str):
        # FactCG @ 41f185413312ed9b3904ec03963fe899693498e0
        # factcg/inference.py and factcg/utils.py (preprocessing and positive class).
        chunks, current, words = [], [], 0
        for sentence in sent_tokenize(doc):
            n_words = len(word_tokenize(sentence))
            if words + n_words > 550:
                chunks.append("\n".join(current))
                current, words = [], 0
            current.append(sentence)
            words += n_words
        if words:
            chunks.append("\n".join(current))
        text = [
            f"{chunk}\n\nChoose your answer: based on the paragraph above can we conclude that "
            f'"{claim}"?\n\nOPTIONS:\n- Yes\n- No\nI think the answer is '
            for chunk in chunks
        ]
        lengths = tokenizer(text, return_length=True, truncation=False, verbose=False)["length"]
        inputs = tokenizer(
            text, padding=True, truncation="only_first", max_length=2048, return_tensors="pt"
        ).to("cuda")
        with torch.inference_mode():
            probability = model(**inputs).logits.float().softmax(dim=-1)[:, 1].max().item()
        return probability, max(lengths), max(lengths) > 2048

    return score


def _sample_gpu(stop: threading.Event, vram: list[int]):
    while not stop.is_set():
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True,
                timeout=3,
            )
            vram.append(int(output.splitlines()[0]))
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            return
        stop.wait(0.25)


def _measure(rows, checkpoint, kind, dtype):
    import torch

    stop, gpu_samples = threading.Event(), []
    thread = threading.Thread(target=_sample_gpu, args=(stop, gpu_samples), daemon=True)
    torch.cuda.reset_peak_memory_stats()
    thread.start()
    lengths, truncated, predictions, scores, batches = [], [], [], [], []
    contradiction_scores = []
    try:
        if kind == "nli":
            scorer = NLIModel(checkpoint, device="cuda", dtype=dtype)
            batch_size = 4
            for start in range(0, len(rows), batch_size):
                batch = rows[start : start + batch_size]
                now = perf_counter()
                probabilities, counts = scorer.score(
                    [row["evidence"] for row in batch], [row["claim"] for row in batch]
                )
                torch.cuda.synchronize()
                batches.append((perf_counter() - now) * 1000)
                scores.extend(row[0] for row in probabilities)
                contradiction_scores.extend(row[1] for row in probabilities)
                predictions.extend(max(range(3), key=row.__getitem__) for row in probabilities)
                lengths.extend(counts)
                truncated.extend(n > scorer.max_length for n in counts)
        else:
            binary = (
                _factcg(checkpoint, dtype)
                if kind == "factcg"
                else _minicheck(checkpoint, kind, dtype)
            )
            for row in rows:
                now = perf_counter()
                probability, count, was_truncated = binary(row["evidence"], row["claim"])
                torch.cuda.synchronize()
                batches.append((perf_counter() - now) * 1000)
                lengths.append(count)
                truncated.append(was_truncated)
                scores.append(probability)
                contradiction_scores.append(None)
                predictions.append(0 if probability > 0.5 else 2)
        if not all(
            score is None or math.isfinite(score) for score in (*scores, *contradiction_scores)
        ):
            raise ValueError(f"Nonfinite probabilities in {dtype}")
        allocated = torch.cuda.max_memory_allocated()
    finally:
        stop.set()
        thread.join()
    return Measurements(
        lengths,
        truncated,
        predictions,
        scores,
        contradiction_scores,
        batches,
        allocated,
        max(gpu_samples, default=None),
    )


def _file_hashes(directory: Path) -> dict:
    result = {}
    for path in sorted(directory.iterdir()):
        if path.is_file():
            with path.open("rb") as source:
                result[path.name] = hashlib.file_digest(source, "sha256").hexdigest()
    return result


def _log_mlflow(result: dict, path: Path, uri: str) -> None:
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=uri)
    name = EXPERIMENT
    experiment = client.get_experiment_by_name(name)
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(name)
    run_id = client.create_run(experiment_id, run_name=result["model"]).info.run_id
    try:
        for key in ("model", "model_revision", "commit", "split_sha256", "dtype", "batch_size"):
            client.log_param(run_id, key, result[key])
        for subset in ("open_web", "fact_checker"):
            for metric, summary in result[subset]["metrics"].items():
                if summary["value"] is not None:
                    client.log_metric(run_id, f"{subset}.{metric}", summary["value"])
        for filename in ("metrics.json", "predictions.parquet"):
            client.log_artifact(run_id, str(path / filename))
        client.set_terminated(run_id)
    except Exception:
        client.set_terminated(run_id, status="FAILED")
        raise


def druid_dev() -> tuple[list[dict], dict, str]:
    raw_manifest = (SPLITS / "druid.json").read_bytes()
    manifest = json.loads(raw_manifest)
    if manifest["dataset"] != DRUID.manifest():
        raise ValueError("DRUID manifest does not match pinned data")
    ids = {
        row_id: group for group, row_ids in manifest["dev"]["groups"].items() for row_id in row_ids
    }
    rows = [{**row, "group": ids[row["id"]]} for row in DRUID.read() if row["id"] in ids]
    if len(rows) != len(ids) or not rows:
        raise ValueError("DRUID DEV manifest contains missing IDs")
    return rows, manifest, hashlib.sha256(raw_manifest).hexdigest()


def write_predictions(path: Path, predictions: list[dict]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    with NamedTemporaryFile(dir=path.parent) as file:
        pq.write_table(pa.Table.from_pylist(predictions), file)
        file.flush()
        os.link(file.name, path)


def run_candidates(mlflow_uri: str = "", only: list[str] | None = None) -> list[dict]:
    import numpy as np
    import torch

    from groundmark.eval.checker_metrics import checker_metrics

    if not torch.cuda.is_available():
        raise RuntimeError("DRUID checker evaluation requires CUDA")

    rows, manifest, split_sha = druid_dev()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    local_punkt()
    failures = []
    candidates = [candidate for candidate in CANDIDATES if not only or candidate[0] in only]
    if not candidates or (only and set(only) != {candidate[0] for candidate in candidates}):
        raise ValueError("Unexpected candidate model")
    for model, revision, kind in candidates:
        dtype = (
            "bfloat16"
            if kind == "flan-t5-large" and torch.cuda.is_bf16_supported()
            else "float32"
            if kind == "flan-t5-large"
            else "float16"
        )
        print(f"checker: {model} @ {revision} ({dtype})", flush=True)
        try:
            checkpoint = snapshot(model, revision, kind)
            measurements = _measure(rows, checkpoint, kind, dtype)
            predictions = [
                {
                    "row_id": row["id"],
                    "group": row["group"],
                    "native_gold": row["evidence_stance"],
                    "is_gold": row["is_gold"] == "True",
                    "prediction": prediction,
                    "support_score": score,
                    "contradiction_score": contradiction,
                    "max_input_tokens": length,
                    "truncated": was_truncated,
                }
                for row, prediction, score, contradiction, length, was_truncated in zip(
                    rows,
                    measurements.predictions,
                    measurements.scores,
                    measurements.contradiction_scores,
                    measurements.lengths,
                    measurements.actual_truncations,
                    strict=True,
                )
            ]
            result = {
                "model": model,
                "model_revision": revision,
                "model_sha256s": _file_hashes(checkpoint),
                "dataset": manifest["dataset"],
                "split_sha256": split_sha,
                "commit": commit,
                "split": "dev",
                "dtype": dtype,
                "batch_size": 4 if kind == "nli" else 1,
                "batches": len(measurements.batches_ms),
                "token_length": dict(
                    zip(
                        ("min", "p50", "p95", "max"),
                        np.quantile(measurements.lengths, (0, 0.5, 0.95, 1)).tolist(),
                        strict=True,
                    )
                ),
                "pairs_over_512": sum(length > 512 for length in measurements.lengths),
                "truncated_pairs": sum(measurements.actual_truncations),
                "pairs_per_s": len(rows) * 1000 / sum(measurements.batches_ms),
                "batch_latency_ms": dict(
                    zip(
                        ("p50", "p95"),
                        np.quantile(measurements.batches_ms, (0.5, 0.95)).tolist(),
                        strict=True,
                    )
                ),
                "peak_allocated_bytes": measurements.peak_allocated_bytes,
                "nvidia_smi_peak_mib": measurements.nvidia_peak_mib,
                "open_web": checker_metrics(
                    [p for p in predictions if not p["is_gold"]], kind == "nli"
                ),
                "fact_checker": checker_metrics(
                    [p for p in predictions if p["is_gold"]], kind == "nli"
                ),
            }
            path = ARTIFACTS / uuid4().hex
            path.mkdir(parents=True, mode=0o700)
            write_predictions(path / "predictions.parquet", predictions)
            private_json(path / "metrics.json", result)
            print(f"checker: artifacts {path}", flush=True)
            if mlflow_uri:
                try:
                    _log_mlflow(result, path, mlflow_uri)
                except Exception as error:
                    failures.append({"model": model, "phase": "mlflow", "reason": str(error)})
        except (OSError, RuntimeError, ValueError, TypeError, LookupError, ImportError) as error:
            failures.append({"model": model, "phase": "checker", "reason": str(error)})
            print(f"checker: failed {model}: {error}", flush=True)
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    return failures
