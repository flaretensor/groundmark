import hashlib
from dataclasses import dataclass
from importlib.resources import files

from groundmark.eval.judge_template import GRADER_TEMPLATE


@dataclass(frozen=True)
class Grader:
    name: str
    text: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


ORIGINAL = Grader("simpleqa_original", GRADER_TEMPLATE)
DATASET_GRADERS = {
    "simpleqa": ("simpleqa_verified",),
    "simpleqa-verified": ("simpleqa_verified",),
    "sealqa": ("sealqa",),
    "seal_hard": ("sealqa",),
    "seal_0": ("sealqa",),
    "evouna": ("simpleqa_verified", "sealqa"),
}


def load_grader(name: str) -> Grader:
    if name == ORIGINAL.name:
        return ORIGINAL
    if name not in {"simpleqa_verified", "sealqa"}:
        raise ValueError(f"Unknown grader: {name}")
    resource = files("groundmark.eval").joinpath("graders", f"{name}.txt")
    return Grader(name, resource.read_bytes().decode("utf-8"))


def dataset_graders(dataset: str) -> tuple[Grader, ...]:
    return tuple(load_grader(name) for name in DATASET_GRADERS[dataset])
