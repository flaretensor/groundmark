import hashlib
import json
import random
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from groundmark.eval.datasets import (
    CACHE,
    DRUID,
    EVOUNA,
    SEAL_0,
    SEAL_HARD,
    SIMPLEQA,
    evouna_counts,
)

SEED = 20261124
SPLITS = Path("eval/splits")


def question_id(question: str) -> str:
    return hashlib.sha256(question.encode()).hexdigest()


def druid_groups(rows: list[dict]) -> dict[str, list[str]]:
    parents = {row["id"]: row["id"] for row in rows}
    if len(parents) != len(rows):
        raise ValueError("Duplicate DRUID row IDs")

    def root(key: str) -> str:
        while key != parents[key]:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    prior = {}
    for row in rows:
        text = " ".join(unicodedata.normalize("NFKC", row["claim"]).casefold().split())
        for key in (("claim_id", row["claim_id"]), ("claim", text)):
            first = prior.setdefault(key, row["id"])
            left, right = sorted((root(first), root(row["id"])))
            parents[right] = left
    groups = defaultdict(list)
    for row_id in sorted(parents):
        groups[root(row_id)].append(row_id)
    return dict(groups)


def druid_split(rows: list[dict]) -> dict:
    groups = druid_groups(rows)
    shuffled = sorted(groups)
    random.Random(SEED).shuffle(shuffled)
    cutoff = round(len(groups) * 0.3)
    gold = {row["id"]: row["is_gold"] for row in rows}

    def manifest(keys: list[str]) -> dict:
        ids = {key: groups[key] for key in keys}
        counts = Counter(gold[row_id] for row_ids in ids.values() for row_id in row_ids)
        return {"groups": ids, "rows_by_is_gold": {key: counts[key] for key in ("False", "True")}}

    return {
        "dataset": DRUID.manifest(),
        "seed": SEED,
        "method": "30/70 shuffled groups joined by claim_id or NFKC/casefold/whitespace claim",
        "native_claim_ids": len({row["claim_id"] for row in rows}),
        "paper_claims": 1329,
        "group_count": len(groups),
        "dev": manifest(shuffled[:cutoff]),
        "test": manifest(shuffled[cutoff:]),
    }


def simpleqa_split(rows: list[dict]) -> dict:
    ids = sorted(int(row["original_index"]) for row in rows)
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate SimpleQA Verified original_index")
    random.Random(SEED).shuffle(ids)
    return {
        "dataset": SIMPLEQA.manifest(),
        "seed": SEED,
        "method": "shuffled original_index",
        "dev": ids[:30],
        "test": ids[30:180],
        "reserve": ids[180:],
    }


def sealqa_split(hard: list[dict], seal_0: list[dict]) -> dict:
    hard_ids = {question_id(row["question"]) for row in hard}
    zero_ids = {question_id(row["question"]) for row in seal_0}
    if len(hard_ids) != len(hard) or len(zero_ids) != len(seal_0) or not zero_ids <= hard_ids:
        raise ValueError("Invalid Seal-0/Seal-Hard question membership")
    rng = random.Random(SEED)
    strata = [sorted(zero_ids), sorted(hard_ids - zero_ids)]
    for ids in strata:
        rng.shuffle(ids)

    def take(n: int) -> list[str]:
        n_zero = round(n * len(strata[0]) / sum(map(len, strata)))
        picked = strata[0][:n_zero] + strata[1][: n - n_zero]
        del strata[0][:n_zero]
        del strata[1][: n - n_zero]
        rng.shuffle(picked)
        return picked

    return {
        "dataset": SEAL_HARD.manifest(),
        "stratifier": SEAL_0.manifest(),
        "seed": SEED,
        "method": "sha256(question), proportional shuffled strata by Seal-0 membership",
        "dev": take(20),
        "test": take(60),
        "reserve": take(len(hard) - 80),
        "seal_0": sorted(zero_ids),
    }


def manifests(cache: Path = CACHE) -> dict[str, dict]:
    simpleqa = simpleqa_split(SIMPLEQA.read(cache))
    sealqa = sealqa_split(SEAL_HARD.read(cache), SEAL_0.read(cache))
    return {
        "druid": druid_split(DRUID.read(cache)),
        "simpleqa": simpleqa,
        "sealqa": sealqa,
        "smoke": {
            "seed": SEED,
            "method": "first five IDs from each DEV split",
            "datasets": {
                "simpleqa": {"dataset": simpleqa["dataset"], "ids": simpleqa["dev"][:5]},
                "sealqa": {"dataset": sealqa["dataset"], "ids": sealqa["dev"][:5]},
            },
        },
        "evouna": {
            "files": [data.manifest() | evouna_counts(data.read(cache)) for data in EVOUNA],
        },
    }


def serialize(manifest: dict) -> bytes:
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()


def freeze(cache: Path = CACHE, destination: Path = SPLITS) -> None:
    generated = manifests(cache)
    destination.mkdir(parents=True, exist_ok=True)
    for name, manifest in generated.items():
        path = destination / f"{name}.json"
        payload = serialize(manifest)
        if path.is_file():
            if path.read_bytes() != payload:
                raise ValueError(f"Frozen split differs: {path}")
        else:
            with path.open("xb") as output:
                output.write(payload)
