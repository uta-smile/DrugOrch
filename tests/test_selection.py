"""The selection-under-noise curve, which is an argument the Goals page makes in public.

A figure that asserts "searching harder mostly finds the oracle's error" has to be checked,
not just plotted -- and checked against the *direction* of each effect, because the mistake
that matters here is a sign error. Picking `argmax` instead of `argmin` on a kcal/mol column
produces a table that is perfectly well-formed, internally consistent, and backwards; it
happened once in this project already, on exactly this kind of comparison.

So these tests pin the claims rather than the plumbing: that a perfect oracle has no regret,
that a noisy one's regret *grows* with the pool, and that the reported number improves far
faster than the molecule behind it does.
"""

from __future__ import annotations

import pytest

from drugorch.harness.selection import (
    DEFAULT_SIZES,
    SelectionRow,
    selection_curve,
)


def test_a_perfect_oracle_always_picks_the_best_molecule_in_the_pool():
    """The control. With noise=0 there is no winner's curse and no case for a budget."""
    curve = selection_curve((10, 1_000), noise=0.0, trials=50, seed=1)

    for row in curve.rows:
        assert row.regret == pytest.approx(0.0, abs=1e-12), f"n={row.n}"
        assert row.best_measured == pytest.approx(row.true_of_pick, abs=1e-12)
        assert row.true_of_pick == pytest.approx(row.best_true_available, abs=1e-12)


def test_a_noisy_oracle_does_not_pick_the_best_molecule_in_the_pool():
    curve = selection_curve((1_000,), noise=2.0, trials=200, seed=1)

    assert curve.rows[0].regret > 0.5


def test_regret_grows_with_the_pool():
    """The counterintuitive claim, and the whole reason the page exists.

    More candidates means more chances for a mediocre molecule to get lucky and outrank the
    genuinely best one, so a bigger search is *worse* at identifying the best thing it holds.
    """
    curve = selection_curve((10, 100, 1_000, 10_000), noise=2.0, trials=400, seed=7)
    regrets = [r.regret for r in curve.rows]

    assert regrets == sorted(regrets), f"regret should be non-decreasing in n, got {regrets}"
    assert regrets[-1] > regrets[0] + 0.5, "the effect should be large, not marginal"


def test_the_reported_number_improves_far_faster_than_the_molecule_does():
    """Best-measured runs away; true-of-pick crawls. That gap is the wasted budget."""
    curve = selection_curve((10, 100_000), noise=2.0, spread=1.0, trials=400, seed=3)
    small, large = curve.rows

    reported_gain = small.best_measured - large.best_measured
    real_gain = small.true_of_pick - large.true_of_pick

    assert reported_gain > 0 and real_gain > 0, "both should improve; more negative is better"
    assert reported_gain > 4 * real_gain, (
        f"reported gain {reported_gain:.2f} should dwarf the real gain {real_gain:.2f}"
    )


def test_the_real_gain_is_bounded_by_the_chemistry_in_the_pool():
    """Searching cannot find a molecule the pool does not contain."""
    curve = selection_curve(DEFAULT_SIZES, noise=2.0, trials=200, seed=11)

    for row in curve.rows:
        assert row.true_of_pick >= row.best_true_available - 1e-9, (
            f"n={row.n}: the pick cannot beat the pool's own ceiling"
        )


def test_a_worse_oracle_produces_more_regret():
    at_one = selection_curve((1_000,), noise=1.0, trials=300, seed=5).rows[0]
    at_three = selection_curve((1_000,), noise=3.0, trials=300, seed=5).rows[0]

    assert at_three.regret > at_one.regret


def test_the_reported_number_overstates_what_was_found():
    """`inflation` is negative: best_measured is more negative than the molecule is worth."""
    row = selection_curve((1_000,), noise=2.0, trials=200, seed=2).rows[0]

    assert row.best_measured < row.true_of_pick
    assert row.inflation > 0


def test_it_is_deterministic_in_the_seed():
    """The figure on the page has to be reproducible, not merely similar on a rerun."""
    a = selection_curve((100, 1_000), noise=2.0, trials=50, seed=42)
    b = selection_curve((100, 1_000), noise=2.0, trials=50, seed=42)
    c = selection_curve((100, 1_000), noise=2.0, trials=50, seed=43)

    assert [r.as_dict() for r in a.rows] == [r.as_dict() for r in b.rows]
    assert [r.as_dict() for r in a.rows] != [r.as_dict() for r in c.rows]


def test_the_answer_does_not_depend_on_how_it_is_chunked():
    """Memory management must not move the published numbers.

    An earlier draft averaged over blocks of trials sized by a memory constant, and this
    test caught that the table changed when the constant did -- same statistics, different
    figure, because a (64, n) draw and sixty-four (1, n) draws consume the generator
    differently. Drawing one trial at a time removed the knob. This guards the property that
    mattered: identical input, identical table, whatever the pool size.
    """
    small_pool = selection_curve((7,), noise=2.0, trials=64, seed=9)
    big_pool = selection_curve((50_000,), noise=2.0, trials=4, seed=9)

    assert selection_curve((7,), noise=2.0, trials=64, seed=9).as_dict() == small_pool.as_dict()
    assert selection_curve((50_000,), noise=2.0, trials=4, seed=9).as_dict() == big_pool.as_dict()


def test_assumptions_travel_with_the_numbers():
    payload = selection_curve((10,), noise=1.5, spread=0.5, trials=10, seed=4).as_dict()

    assert payload["assumptions"] == {
        "noise": 1.5,
        "spread": 0.5,
        "center": -8.0,
        "trials": 10,
        "seed": 4,
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sizes": ()},
        {"sizes": (0,)},
        {"sizes": (-5,)},
        {"trials": 0},
        {"noise": -1.0},
        {"spread": -1.0},
    ],
)
def test_nonsense_parameters_are_refused(kwargs):
    sizes = kwargs.pop("sizes", (10,))
    kwargs.setdefault("trials", 5)
    with pytest.raises(ValueError):
        selection_curve(sizes, **kwargs)


def test_row_arithmetic_is_what_it_claims():
    row = SelectionRow(
        n=10, best_measured=-12.0, true_of_pick=-9.0, best_true_available=-10.0
    )

    assert row.regret == pytest.approx(1.0)
    assert row.inflation == pytest.approx(3.0)


class TestEndpoint:
    """The figure on the Goals page is served, not computed in the browser.

    Worth its own tests because the page's claim is only as good as the numbers behind it,
    and because the two query parameters are the one place a reader can push on the
    assumption the whole argument rests on.
    """

    @staticmethod
    def _client():
        from fastapi.testclient import TestClient

        from drugorch.api.app import create_app

        return TestClient(create_app())

    def test_it_serves_the_curve_with_its_assumptions(self):
        body = self._client().get("/api/selection-curve").json()

        assert [r["n"] for r in body["rows"]] == list(DEFAULT_SIZES)
        assert body["assumptions"]["noise"] == 2.0
        assert all(r["regret"] > 0 for r in body["rows"])

    def test_the_perfect_oracle_control_is_reachable(self):
        body = self._client().get("/api/selection-curve", params={"noise": 0}).json()

        assert all(r["regret"] == 0.0 for r in body["rows"])

    def test_repeated_requests_agree(self):
        """Memoised, and in any case seeded -- a reader who reloads sees the same figure."""
        client = self._client()
        first = client.get("/api/selection-curve", params={"noise": 1.0}).json()
        second = client.get("/api/selection-curve", params={"noise": 1.0}).json()

        assert first == second

    @pytest.mark.parametrize("params", [{"noise": -1}, {"noise": 99}, {"spread": 0}])
    def test_parameters_outside_the_modelled_range_are_refused(self, params):
        assert self._client().get("/api/selection-curve", params=params).status_code == 422
