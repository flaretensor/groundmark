import hashlib
import json
from pathlib import Path
from string import Formatter

from groundmark.eval.graders import ORIGINAL, dataset_graders, load_grader
from groundmark.eval.judge import SYSTEM


def test_dataset_grade_templates_match_manifest_and_keep_original():
    manifest = json.loads(Path("eval/graders/manifest.json").read_bytes())
    assert manifest["byte_verified_against_authors_released_code"] is False
    for name, entry in manifest["templates"].items():
        grader = load_grader(name)
        assert grader.sha256 == entry["sha256"]
        fields = [field for _, field, _, _ in Formatter().parse(grader.text) if field]
        assert fields == ["question", "target", "predicted_answer"]
        assert grader.text.splitlines()[-1] == (
            'Just return the letters "A", "B", or "C", with no text around it.'
        )
    assert ORIGINAL.sha256 == "063c04ea798f91c4706c050b66a721afc77a458b12d8f0d72286d4b0eee8a3ee"
    assert hashlib.sha256(SYSTEM.encode()).hexdigest() == (
        "f429510e41ec9276adde203160acc0443d1de40aeaeec24e8da9ba1b3f3f5a80"
    )


def test_dataset_grader_mapping():
    def names(dataset):
        return [grader.name for grader in dataset_graders(dataset)]

    assert names("simpleqa") == names("simpleqa-verified") == ["simpleqa_verified"]
    assert names("sealqa") == names("seal_hard") == names("seal_0") == ["sealqa"]
    assert names("evouna") == ["simpleqa_verified", "sealqa"]
