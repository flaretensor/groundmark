import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

import pyarrow.parquet as pq

CACHE = Path(".cache/datasets")
EVOUNA_JUDGES = ("fid", "gpt35", "chatgpt", "gpt4", "newbing")


@dataclass(frozen=True)
class DatasetFile:
    dataset: str
    repo: str
    revision: str
    filename: str
    sha256: str
    rows: int

    @property
    def url(self) -> str:
        if self.dataset == "evouna":
            return f"https://raw.githubusercontent.com/{self.repo}/{self.revision}/EVOUNA/{self.filename}"
        return (
            f"https://huggingface.co/datasets/{self.repo}/resolve/{self.revision}/{self.filename}"
        )

    def manifest(self) -> dict:
        return {
            "repo": self.repo,
            "revision": self.revision,
            "filename": self.filename,
            "sha256": self.sha256,
            "rows": self.rows,
        }

    def path(self, cache: Path = CACHE) -> Path:
        return cache / self.dataset / self.revision / self.filename

    def verify(self, path: Path) -> None:
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if digest != self.sha256:
            raise ValueError(f"{path}: SHA256 {digest} != {self.sha256}")

    def fetch(self, cache: Path = CACHE) -> None:
        path = self.path(cache)
        if path.is_file():
            self.verify(path)
            return
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with NamedTemporaryFile(dir=path.parent, delete_on_close=False) as tmp:
            with urlopen(self.url, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    tmp.write(chunk)
            tmp.flush()
            self.verify(Path(tmp.name))
            os.link(tmp.name, path)  # Publish complete data without overwriting a cache entry.

    def read(self, cache: Path = CACHE) -> list[dict]:
        path = self.path(cache)
        self.verify(path)
        if path.suffix == ".parquet":
            rows = pq.read_table(path).to_pylist()
        elif path.suffix == ".json":
            rows = json.loads(path.read_bytes())
        else:
            with path.open(encoding="utf-8", newline="") as stream:
                rows = list(
                    csv.DictReader(stream, delimiter="\t" if path.suffix == ".tsv" else ",")
                )
        if not isinstance(rows, list) or len(rows) != self.rows:
            raise ValueError(f"{path}: expected {self.rows} rows")
        return rows


DRUID = DatasetFile(
    "druid",
    "copenlu/druid",
    "d3f49c237d7d99870f97df51ac52e4f5e16f5cb6",
    "druid.tsv",
    "80f49f5ff0a2128cb272e055117e4d7f938dd40f4a0c8b96fd2936cf7224ddc6",
    5490,
)
SEAL_HARD = DatasetFile(
    "sealqa",
    "vtllms/sealqa",
    "0e75db526ad93dd039714897b7631afc03510e30",
    "seal-hard.parquet",
    "b13fbacabac4b055dcabfae161df4dc661e31e915507324790779d107ab53e7e",
    254,
)
SEAL_0 = DatasetFile(
    "sealqa",
    SEAL_HARD.repo,
    SEAL_HARD.revision,
    "seal-0.parquet",
    "89c729bd0a13c56847bb30cbdce1e091f7b832bbe9929d3744357cedcf4e33be",
    111,
)
SIMPLEQA = DatasetFile(
    "simpleqa",
    "google/simpleqa-verified",
    "0dc97e0d28d8233463e005cdc4475cc2a13ba2dc",
    "simpleqa_verified.csv",
    "b5db21155444763543fe31b67e7cf28ce2bb225742a5b889421c5f182e2f92f5",
    1000,
)
EVOUNA = (
    DatasetFile(
        "evouna",
        "wangcunxiang/QA-Eval",
        "90007d39d4ef0720984f80d56a586c66d6241e70",
        "NQ.json",
        "617c7fd95fe4177f39e2304c943cc0abbdbf539bc8a4fb61aeaef255bed503a1",
        3610,
    ),
    DatasetFile(
        "evouna",
        "wangcunxiang/QA-Eval",
        "90007d39d4ef0720984f80d56a586c66d6241e70",
        "TQ.json",
        "e01709ec6b979d3976f2f8f676dab2e2aea50e0a07f3c66d70dae358e5360d23",
        2000,
    ),
)
DATASETS = (DRUID, SEAL_HARD, SEAL_0, SIMPLEQA, *EVOUNA)


def evouna_counts(rows: list[dict]) -> dict:
    counts = {
        key: {"correct": 0, "incorrect": 0, "missing": 0, "eligible": 0} for key in EVOUNA_JUDGES
    }
    for row in rows:
        if type(row["question"]) is not str or type(row["improper"]) is not bool:
            raise ValueError("Unexpected EVOUNA question or improper value")
        if type(row["golden_answer"]) not in {str, type(None)}:
            raise ValueError("Unexpected EVOUNA reference answer")
        for key, stats in counts.items():
            answer, judgment = row[f"answer_{key}"], row[f"judge_{key}"]
            if type(answer) not in {str, type(None)}:
                raise ValueError(f"Unexpected EVOUNA answer field: {key}")
            if type(judgment) is bool:
                stats["correct" if judgment else "incorrect"] += 1
                if not row["improper"] and answer is not None and row["golden_answer"] is not None:
                    stats["eligible"] += 1
            elif judgment is None or judgment == "nan":
                stats["missing"] += 1
            else:
                raise ValueError(f"Unexpected EVOUNA judge field: {key}")
    return {"improper": sum(row["improper"] for row in rows), "judges": counts}
