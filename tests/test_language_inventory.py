"""Language inventory seed validation (master spec §7.1).

The inventory is DATA, not a capability claim: the coding engine remains
PLANNED. This test guarantees every entry carries the required verifiable
fields and that the coverage gap vs the 2,500-language goal is stated
explicitly in the file itself.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "data" / "language_inventory.json"

REQUIRED_FIELDS = ["name", "aliases", "paradigms", "typing", "execution",
                   "first_appeared", "notes"]


def load_inventory() -> dict:
    return json.loads(INVENTORY.read_text(encoding="utf-8"))


def test_inventory_exists_and_loads():
    data = load_inventory()
    assert "languages" in data and "_meta" in data


def test_every_language_has_required_fields():
    data = load_inventory()
    assert len(data["languages"]) >= 20, "seed inventory too small to be useful"
    for lang in data["languages"]:
        for field in REQUIRED_FIELDS:
            assert field in lang, f"{lang.get('name', '?')} missing field {field}"
        assert isinstance(lang["aliases"], list)
        assert isinstance(lang["paradigms"], list) and lang["paradigms"]
        assert isinstance(lang["first_appeared"], int)
        assert 1950 <= lang["first_appeared"] <= 2026


def test_names_are_unique():
    data = load_inventory()
    names = [l["name"].lower() for l in data["languages"]]
    assert len(names) == len(set(names)), "duplicate language names"


def test_meta_states_aspirational_target_honestly():
    meta = load_inventory()["_meta"]
    assert "2500" in str(meta.get("target", ""))
    assert "NOT claimed" in meta.get("coverage_note", "") + meta.get("status", "") \
        or "not claimed" in meta.get("coverage_note", "").lower()
    assert meta.get("status", "").startswith("PROTOTYPE")
