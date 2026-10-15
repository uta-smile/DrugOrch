"""The metric glossary, checked against the code that computes the metrics.

A glossary is a second description of something that already exists, so the failure mode is
drift: a metric renamed in the harness, a descriptor added to the property panel, or a
leaderboard column that no entry explains. None of those break anything, and all of them
leave a reader with a number nobody defines. These tests tie the entries to their sources.
"""

from __future__ import annotations

import pytest

from drugorch.glossary import ALIASES, BY_KEY, GROUPS, as_dict, lookup
from drugorch.harness.report import RANKED_BY
from drugorch.subsystems.properties import DESCRIPTOR_DOC, FLAG_DOC, SCORE_DOC


class TestCoverage:
    def test_every_leaderboard_ranking_metric_is_defined(self):
        """These are the column the leaderboard sorts on. An undefined one is unreadable."""
        missing = [k for k in RANKED_BY if k not in BY_KEY]

        assert not missing, f"leaderboard metrics with no glossary entry: {missing}"

    def test_every_property_panel_score_and_flag_is_defined(self):
        missing = [k for k in (*SCORE_DOC, *FLAG_DOC) if k not in BY_KEY]

        assert not missing, f"property panel values with no glossary entry: {missing}"

    def test_the_headline_descriptors_are_defined(self):
        """Not every descriptor needs an entry, but the ones a goal is written against do."""
        for key in ("mw", "logp", "tpsa", "hbd", "hba", "rotatable_bonds", "heavy_atoms"):
            assert key in BY_KEY, key
            assert key in DESCRIPTOR_DOC, f"{key} is not a real descriptor"

    def test_the_derived_metrics_the_article_teaches_are_defined(self):
        for key in ("ligand_efficiency", "lipophilic_ligand_efficiency", "pchembl", "tanimoto"):
            assert key in BY_KEY, key


class TestEntries:
    @pytest.mark.parametrize("key", sorted(BY_KEY))
    def test_each_entry_is_complete(self, key):
        m = BY_KEY[key]

        assert m.term, key
        assert m.short.endswith("."), f"{key}: short form should be a sentence"
        assert m.reading, f"{key}: an entry with no reading guidance is a label, not a definition"
        assert m.group in GROUPS, f"{key}: unknown group {m.group!r}"

    @pytest.mark.parametrize("key", sorted(BY_KEY))
    def test_cross_references_resolve(self, key):
        for ref in BY_KEY[key].see_also:
            assert ref in BY_KEY, f"{key} points at missing entry {ref!r}"

    @pytest.mark.parametrize("key", sorted(BY_KEY))
    def test_no_entry_points_at_itself(self, key):
        assert key not in BY_KEY[key].see_also

    def test_the_metrics_that_mislead_carry_a_caution(self):
        """The three this interface makes it easiest to misread."""
        for key in ("best_score", "score", "ligand_efficiency"):
            assert BY_KEY[key].caution, f"{key} should warn about its common misuse"

    def test_direction_is_declared_for_ranked_metrics(self):
        for key in RANKED_BY:
            assert BY_KEY[key].direction in ("higher", "lower"), key


class TestLookup:
    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("discovery_auc", "discovery_auc"),
            ("LE", "ligand_efficiency"),
            ("le", "ligand_efficiency"),
            ("QED", "qed"),
            ("best", "best_score"),
            ("pains", "pains_alert"),
            ("Docking score", "score"),
            ("molecular_weight", "mw"),
            ("clogp", "logp"),
            ("sa", "sa_score"),
        ],
    )
    def test_it_resolves_the_spellings_the_ui_renders(self, query, expected):
        """Columns are headed `best`, records key on `sa_score`, prose says `LE`."""
        m = lookup(query)

        assert m is not None, query
        assert m.key == expected

    def test_an_unknown_name_returns_none_rather_than_guessing(self):
        assert lookup("not_a_metric") is None
        assert lookup("") is None

    def test_every_alias_points_somewhere_real(self):
        broken = {a: t for a, t in ALIASES.items() if t not in BY_KEY}

        assert not broken, broken

    def test_no_alias_shadows_a_real_key(self):
        """An alias with the same name as a metric would silently win or lose at random."""
        clashes = [a for a in ALIASES if a in BY_KEY]

        assert not clashes, clashes


class TestPayload:
    def test_it_serialises(self):
        import json

        json.dumps(as_dict())

    def test_the_short_text_comes_from_the_computing_module(self):
        """Not a copy. Editing RANKED_BY should change the glossary."""
        assert BY_KEY["rediscovery"].short.rstrip(".") in RANKED_BY["rediscovery"]

    def test_a_missing_ranked_entry_is_named_rather_than_blank(self):
        """`_ranked` must not raise at import time, and must not degrade to a bare ".".

        An earlier version returned "." for a missing key, so the metric's published
        definition became a single full stop in both the popover and the article, with
        nothing anywhere indicating a problem.
        """
        from drugorch.glossary import _ranked

        out = _ranked("no_such_metric")

        assert out != "."
        assert "no_such_metric" in out
