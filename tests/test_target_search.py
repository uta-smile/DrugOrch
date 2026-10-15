"""Target query classification and the readiness verdict.

These are the two pieces of the picker that are pure logic, so they are the two worth
pinning. Everything else in that path is a network call covered by driving the UI.

Classification matters more than it looks: a user arriving from a paper has a PDB ID or a
ChEMBL ID in hand, not a gene symbol, and before this both of those returned nothing from a
name search -- a dead end that looked like "this protein is not in the database".
"""

from __future__ import annotations

import pytest

from drugorch.clients.targets import SUGGESTED, Readiness, classify


@pytest.mark.parametrize(
    ("query", "kind"),
    [
        ("O60674", "accession"),
        ("P00533", "accession"),
        ("Q9BYF1", "accession"),
        ("o60674", "accession"),  # case-insensitive: people paste what they copied
        ("7LL4", "pdb"),
        ("6q4h", "pdb"),
        ("1ATP", "pdb"),
        ("CHEMBL2971", "chembl"),
        ("chembl301", "chembl"),
        ("JAK2", "text"),
        ("epidermal growth factor receptor", "text"),
        ("", "text"),
    ],
)
def test_query_kinds(query, kind):
    assert classify(query) == kind


def test_gene_symbols_are_not_mistaken_for_identifiers():
    """A four-character gene name must not be read as a PDB ID.

    PDB IDs start with a digit, which is the only thing separating '7LL4' from 'ABL1'.
    Getting this backwards would send a gene search to the structure bank and return nothing.
    """
    for gene in ("ABL1", "CDK2", "EGFR", "BRAF", "KRAS", "MTOR"):
        assert classify(gene) == "text"


def test_whitespace_is_tolerated():
    assert classify("  O60674  ") == "accession"
    assert classify(" 7LL4 ") == "pdb"


# --- the verdict -----------------------------------------------------------
def _ready(**kw) -> Readiness:
    base = {"uniprot": "O60674", "gene": "JAK2", "name": "JAK2", "organism": "Homo sapiens"}
    return Readiness(**{**base, **kw})


def test_a_fully_ready_target_says_so_and_names_the_structure():
    r = _ready(
        n_actives=17653,
        structures=[{"pdb_id": "7LL4", "resolution": 1.31, "ligand_code": "Y5D"}],
    )
    v = r.verdict()
    assert v["level"] == "ok"
    assert "7LL4" in v["text"]
    assert "17,653" in v["text"]
    assert r.can_dock and r.can_generate


def test_no_structure_is_a_warning_not_a_pass():
    """Without a dockable structure the run has no objective, which the user must see."""
    r = _ready(n_actives=500, structures=[])
    v = r.verdict()
    assert v["level"] == "warn"
    assert not r.can_dock
    assert "docking will be unavailable" in v["text"]


def test_too_few_actives_blocks_generation():
    """GraphGA crosses over real molecules; nine of them is not a seed set."""
    r = _ready(n_actives=9, structures=[{"pdb_id": "7LL4", "resolution": 1.3, "ligand_code": "Y5D"}])
    assert r.can_dock
    assert not r.can_generate
    assert r.verdict()["level"] == "warn"


def test_an_unresolved_target_is_an_error():
    v = Readiness(uniprot="").verdict()
    assert v["level"] == "error"


def test_verdict_survives_a_structure_with_no_resolution():
    """Cryo-EM and NMR entries have no resolution; formatting must not divide by it."""
    r = _ready(n_actives=200, structures=[{"pdb_id": "8ABC", "resolution": None, "ligand_code": "XYZ"}])
    assert "8ABC" in r.verdict()["text"]


def test_an_apo_structure_does_not_count_as_dockable():
    """The docking box is centred on the native ligand, so nothing bound means no box.

    ABL1 returns a run of very high-resolution apo structures at the top of the RCSB
    search. Counting those as dockable promised a run that receptor prep would refuse.
    """
    r = _ready(
        n_actives=6060,
        structures=[
            {"pdb_id": "5OAZ", "resolution": 1.03, "ligand_code": ""},
            {"pdb_id": "5HU9", "resolution": 1.53, "ligand_code": "66K"},
        ],
    )
    assert r.dockable == [{"pdb_id": "5HU9", "resolution": 1.53, "ligand_code": "66K"}]
    assert r.can_dock
    # The verdict must name the structure that can actually be docked into, not the
    # higher-resolution one that cannot.
    assert "5HU9" in r.verdict()["text"]
    assert "5OAZ" not in r.verdict()["text"]


def test_only_apo_structures_means_docking_is_unavailable():
    r = _ready(n_actives=500, structures=[{"pdb_id": "5OAZ", "resolution": 1.03,
                                           "ligand_code": ""}])
    assert not r.can_dock
    assert r.verdict()["level"] == "warn"
    assert "docking will be unavailable" in r.verdict()["text"]


# --- the starter list ------------------------------------------------------
def test_suggested_targets_are_well_formed():
    """These are shown before anything is typed, so a bad accession is a dead first click."""
    assert len(SUGGESTED) >= 6
    for s in SUGGESTED:
        assert classify(s["uniprot"]) == "accession", s
        assert s["gene"] and s["name"] and s["why"]
    assert len({s["uniprot"] for s in SUGGESTED}) == len(SUGGESTED)
