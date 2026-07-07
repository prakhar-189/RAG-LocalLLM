"""Validates the evaluation golden dataset stays well-formed."""
import json
from pathlib import Path

DATASET = Path(__file__).resolve().parents[1] / "eval" / "golden_dataset.json"


def _load():
    with open(DATASET, encoding="utf-8") as f:
        return json.load(f)


def test_dataset_is_valid_json_with_expected_keys():
    data = _load()
    assert "document" in data
    assert isinstance(data.get("qa_pairs"), list)
    assert len(data["qa_pairs"]) >= 1


def test_every_qa_pair_has_required_fields():
    for item in _load()["qa_pairs"]:
        assert item.get("question"), "question must be non-empty"
        assert item.get("ground_truth"), "ground_truth must be non-empty"
        assert isinstance(item["in_document"], bool)


def test_ids_are_unique():
    ids = [item["id"] for item in _load()["qa_pairs"]]
    assert len(ids) == len(set(ids))


def test_has_out_of_scope_refusal_case():
    # At least one question must be answerable only by refusing, to test grounding.
    assert any(item["in_document"] is False for item in _load()["qa_pairs"])
