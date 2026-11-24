import pytest

from groundmark.eval.datasets import DATASETS
from groundmark.eval.splits import SPLITS, druid_groups, manifests, serialize


def test_druid_components_join_transitively():
    rows = [
        {"id": "a", "claim_id": "1", "claim": "one"},
        {"id": "b", "claim_id": "1", "claim": "Two"},
        {"id": "c", "claim_id": "2", "claim": "  ＴＷＯ  "},
        {"id": "d", "claim_id": "3", "claim": "three"},
    ]
    assert druid_groups(rows) == {"a": ["a", "b", "c"], "d": ["d"]}


def test_frozen_splits_match_pinned_data_byte_for_byte():
    if not all(data.path().is_file() for data in DATASETS):
        pytest.skip("Pinned dataset bytes are unavailable")
    for name, manifest in manifests().items():
        assert serialize(manifest) == (SPLITS / f"{name}.json").read_bytes()
