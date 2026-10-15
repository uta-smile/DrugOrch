"""Definitions for every metric the interface displays.

The numbers in this application come from four different places: RDKit descriptors, a
docking scoring function, a public activity database, and the benchmark harness. A reader
looking at a column headed ``score_auc`` or ``LE`` has no way to tell which, what range is
normal, or which direction is good. Without that, a leaderboard is a table of numbers that
cannot be argued with.

One entry per metric, in one module, for three reasons.

**The short text is not duplicated.** Where a definition already exists next to the code
that computes it, it is imported rather than restated: the descriptor and flag text from
:mod:`drugorch.subsystems.properties`, the ranking text from
:mod:`drugorch.harness.report`. Restating them here would create a second copy free to drift
from the computation.

**``reading`` is the part that is usually missing.** Every entry says how to read the number
and what commonly misleads. That is not decoration: a docking score rising with heavy-atom
count, or a best-of-batch improving with batch size, are the two errors this interface makes
easiest to commit.

**It is served, not hardcoded in the frontend.** The web page, the goals article and any
later export read the same entries, so a metric cannot be described one way in a tooltip and
another in the documentation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

from .harness.report import RANKED_BY
from .subsystems.properties import DESCRIPTOR_DOC, FLAG_DOC, SCORE_DOC


def _ranked(key: str) -> str:
    """The second clause of a `RANKED_BY` entry, which is what the metric actually counts.

    Those strings are written as "higher is better. <what it counts>" because the leaderboard
    subtitle needs the direction. Here the direction is already a field, so only the second
    clause is wanted. Split defensively: indexing the result of `split` would turn an edit to
    an unrelated module into an ImportError at startup, and the direction prefix is a
    convention rather than a contract.
    """
    text = RANKED_BY.get(key, "")
    if not text:
        # Returning "." here was silent and produced a metric whose published definition was
        # a bare full stop. A missing key is a programming error, and the entry says so.
        return f"No description is registered for {key}."
    _, _, rest = text.partition(". ")
    return (rest or text).rstrip(" .") + "."


Direction = Literal["higher", "lower", "context"]


@dataclass(frozen=True, slots=True)
class Metric:
    """One quantity the interface shows, and how to read it."""

    key: str
    term: str
    short: str
    """One sentence. What the number is."""

    group: str
    direction: Direction = "context"
    range: str = ""
    threshold: str = ""
    """A decision line quoted with the metric, where one is conventional."""

    computed: str = ""
    """How the value is produced, including the implementation where that matters."""

    reading: str = ""
    """How to interpret it in this application."""

    caution: str = ""
    """The error this metric most often invites. Empty when there is no common one."""

    see_also: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return asdict(self)


GROUPS = {
    "benchmark": "Leaderboard",
    "external": "Public benchmark",
    "docking": "Docking and binding",
    "efficiency": "Efficiency",
    "activity": "Measured activity",
    "property": "Physicochemical properties",
    "flag": "Structural flags",
    "similarity": "Similarity and novelty",
    "run": "Run accounting",
}


def _descriptor(key: str, *, reading: str, caution: str = "", direction: Direction = "context",
                see_also: tuple[str, ...] = ()) -> Metric:
    """A property-panel descriptor, with its title and threshold taken from properties.py."""
    title, unit, threshold = DESCRIPTOR_DOC[key]
    return Metric(
        key=key,
        term=title,
        short=f"{title}{f' ({unit})' if unit else ''}.",
        group="property",
        direction=direction,
        # `range` and `threshold` are different claims. Folding the second into the first
        # published "Range. Lipinski: > 500 is a violation", which is not a range.
        range=unit,
        threshold=threshold,
        computed="RDKit, from the canonical SMILES.",
        reading=reading,
        caution=caution,
        see_also=see_also,
    )


_METRICS: tuple[Metric, ...] = (
    # --- the public benchmark ---------------------------------------------------------
    #
    # These seven are MolDesignBench's, not this project's. Their definitions are quoted
    # from its evaluator rather than restated, because the board shows them beside numbers
    # published by its authors and a definition that drifted would make the comparison
    # meaningless while still looking like one.
    Metric(
        key="success",
        term="Success",
        short="Fraction of instances solved outright.",
        group="external",
        direction="higher",
        range="0 to 1",
        computed=(
            "A feasible instance counts when the answer is a valid molecule satisfying "
            "every constraint in the key. An infeasible instance counts when the answer is "
            "None. Computed by MolDesignBench's own evaluator."
        ),
        reading=(
            "The headline. It is all-or-nothing per instance, so an answer missing one "
            "constraint of seven scores the same as an answer missing all seven."
        ),
        caution=(
            "Two success rates are comparable only over the same instances. A row measured "
            "on a subset carries its sample size for that reason."
        ),
        see_also=("csr", "mcd", "inf_acc"),
    ),
    Metric(
        key="csr",
        term="Condition satisfaction rate",
        short="Average fraction of constraints satisfied per feasible instance.",
        group="external",
        direction="higher",
        range="0 to 1",
        computed=(
            "Satisfied constraints over total constraints, averaged across feasible "
            "instances, counting an instance with no valid molecule as zero."
        ),
        reading=(
            "The partial-credit view of Success. A large gap between the two means answers "
            "are landing close to the specification and missing one constraint."
        ),
        see_also=("success", "mcd"),
    ),
    Metric(
        key="mcd",
        term="Mean constraint distance",
        short="Average normalised distance outside the required windows, as a percentage.",
        group="external",
        direction="lower",
        range="0 upward",
        computed=(
            "Each missed constraint is divided by that property's p5 to p95 range over "
            "10.8 million molecules, averaged over the constraints of an instance and "
            "reported over valid molecules only."
        ),
        reading=(
            "How badly a near miss misses. Normalisation is what lets a molecular weight "
            "error and a QED error be averaged at all."
        ),
        caution=(
            "Reported over valid molecules, so an arm that answers rarely but precisely can "
            "post a low distance. Read it beside Validity and Success."
        ),
        see_also=("success", "csr", "validity"),
    ),
    Metric(
        key="inf_acc",
        term="Infeasible accuracy",
        short="Fraction of contradictory instances correctly answered None.",
        group="external",
        direction="higher",
        range="0 to 1",
        computed="Instances flagged infeasible in the key that were answered None.",
        reading=(
            "One instance in ten is impossible by construction, with a minimum above a "
            "maximum. Recognising those is a separate skill from designing a molecule."
        ),
        caution=(
            "Easy to game by abstaining often, which costs Success on the nine in ten that "
            "are solvable. MCC is the column that catches that trade."
        ),
        see_also=("mcc", "success"),
    ),
    Metric(
        key="mcc",
        term="Feasibility MCC",
        short="Correlation between attempting an instance and that instance being solvable.",
        group="external",
        direction="higher",
        range="-1 to 1",
        computed=(
            "Matthews correlation over the confusion matrix of attempted against feasible, "
            "pooled across tasks before the coefficient is taken."
        ),
        reading=(
            "Whether abstention is discriminating or indiscriminate. Zero is the score for "
            "always attempting and for always abstaining alike."
        ),
        see_also=("inf_acc",),
    ),
    Metric(
        key="validity",
        term="Validity",
        short="Fraction of feasible instances answered with a parseable molecule.",
        group="external",
        direction="higher",
        range="0 to 1",
        computed="RDKit parses the answer string, in the benchmark's own environment.",
        reading=(
            "The floor under every other column. A molecule that does not parse cannot "
            "satisfy anything, and published rows range from 0.25 to 0.97 on this."
        ),
        see_also=("success",),
    ),
    Metric(
        key="tool_calls",
        term="Tool calls",
        short="Average subsystem invocations per instance.",
        group="external",
        direction="context",
        range="0 to 20",
        computed=(
            "Counted the way the benchmark counts them, one per subsystem invocation, "
            "against the paper's allowance of twenty."
        ),
        reading=(
            "Cost, not quality. It is on the board so a score can be read against the "
            "compute that bought it."
        ),
        see_also=("success",),
    ),
    # --- leaderboard -----------------------------------------------------------------
    Metric(
        key="discovery_auc",
        term="Discovery AUC",
        short=_ranked("discovery_auc"),
        group="benchmark",
        direction="higher",
        range="0 to 1",
        computed=(
            "Area under the curve of held-out scaffolds found against budget spent, "
            "normalised so that finding every scaffold immediately scores 1."
        ),
        reading=(
            "The primary leaderboard metric. It rewards finding the right chemistry and "
            "rewards finding it early, so an arm that spends its whole budget before "
            "succeeding scores below one that succeeds halfway through."
        ),
        caution=(
            "Only comparable between arms given the same budget, targets and seed. It is a "
            "rate, so a longer run is not penalised for being longer but is penalised for "
            "being slow."
        ),
        see_also=("rediscovery", "score_auc"),
    ),
    Metric(
        key="rediscovery",
        term="Rediscovery",
        short=_ranked("rediscovery"),
        group="benchmark",
        direction="higher",
        range="0 to 1",
        computed=(
            "Fraction of the held-out scaffold set reached at any point in the run, "
            "ignoring when it was reached."
        ),
        reading=(
            "Discovery AUC without the timing component. Comparing the two separates an arm "
            "that found the chemistry slowly from one that did not find it at all."
        ),
        see_also=("discovery_auc",),
    ),
    Metric(
        key="best_score",
        term="Best score",
        short=_ranked("best_score"),
        group="benchmark",
        direction="higher",
        range="unbounded",
        computed="The highest docking score reached by any compound in the run.",
        reading=(
            "A weak endpoint on its own. It is a single order statistic from a noisy "
            "function, so it improves with the number of compounds scored whether or not "
            "the chemistry improved."
        ),
        caution=(
            "The expected maximum of a sample grows with sample size even when every "
            "compound is equally good. Compare medians, or compare at equal budget."
        ),
        see_also=("score", "discovery_auc"),
    ),
    Metric(
        key="score_auc",
        term="Score AUC",
        short=_ranked("score_auc"),
        group="benchmark",
        direction="higher",
        range="unbounded",
        computed="Area under the best-score-so-far curve against budget spent.",
        reading=(
            "Measures how quickly an arm reached its own best score, not how good that "
            "score was. An arm can lead on score AUC and still finish behind."
        ),
        see_also=("best_score",),
    ),
    Metric(
        key="mean_delta",
        term="Mean delta",
        short="Mean difference in the primary metric against the baseline arm.",
        group="benchmark",
        direction="higher",
        range="signed",
        computed=(
            "Paired per cell against the `fixed` arm on the same target and seed, then "
            "averaged. Pairing is what removes target difficulty from the comparison."
        ),
        reading=(
            "The number to read when asking whether a policy helps. An arm can have a worse "
            "absolute score and a positive delta if it drew harder targets."
        ),
        caution="Undefined for the baseline arm itself, which is shown as an empty cell.",
        see_also=("win_rate", "discovery_auc"),
    ),
    Metric(
        key="win_rate",
        term="Win rate",
        short="Fraction of paired cells where an arm beat the baseline.",
        group="benchmark",
        direction="higher",
        range="0% to 100%",
        computed="Count of cells with a positive delta, divided by the number of paired cells.",
        reading=(
            "A sign test companion to mean delta. A high win rate with a small mean delta "
            "means the arm wins often by little; the reverse means it is carried by a few "
            "cells."
        ),
        caution="With few cells this is a coarse statistic. Twelve cells cannot resolve small effects.",
        see_also=("mean_delta",),
    ),
    Metric(
        key="novelty",
        term="Novelty",
        short="Fraction of proposed compounds unlike anything the run was given.",
        group="benchmark",
        direction="context",
        range="0 to 1",
        computed=(
            "Proportion of generated compounds whose nearest neighbour among the retrieved "
            "actives falls below the similarity threshold."
        ),
        reading=(
            "Whether an arm explored or stayed close to what retrieval handed it. Neither "
            "direction is good in itself: high novelty with poor scores is drift, and low "
            "novelty with good scores is competent lead optimisation."
        ),
        caution=(
            "Not a measure of quality, and not freedom to operate. Read it next to "
            "discovery AUC rather than on its own."
        ),
        see_also=("tanimoto", "resemblance_to_actives", "discovery_auc"),
    ),
    Metric(
        key="diversity",
        term="Diversity",
        short="Spread of the compound set produced by a run.",
        group="benchmark",
        direction="context",
        range="0 to 1",
        computed="Mean pairwise distance across the run's scaffolds.",
        reading=(
            "A run that collapses onto one scaffold has stopped searching. Low diversity "
            "late in a run is expected; low diversity early is a stuck generator."
        ),
        see_also=("novelty",),
    ),
    Metric(
        key="cost_to_first_hit",
        term="Cost to first hit",
        short="Budget units spent before the first held-out scaffold was found.",
        group="benchmark",
        direction="lower",
        range="0 to the run budget, or undefined",
        computed="Budget consumed at the step that first reached a held-out scaffold.",
        reading=(
            "Measures how quickly an arm got started. Undefined when an arm never found "
            "one, which is a different result from a large value and is shown as empty."
        ),
        see_also=("discovery_auc", "spent"),
    ),
    # --- docking ---------------------------------------------------------------------
    Metric(
        key="score",
        term="Docking score",
        short="AutoDock Vina's estimate of binding affinity for a pose.",
        group="docking",
        direction="higher",
        range="internally negated, so higher is better here",
        computed=(
            "Vina reports binding affinity in kcal/mol where more negative is better. The "
            "value stored in a run is negated so that every score in the system is "
            "maximised. Displays that show kcal/mol negate it back."
        ),
        reading=(
            "A ranking function, not an assay. Use it to order candidates within one run, "
            "where the pocket, box and exhaustiveness are identical."
        ),
        caution=(
            "Published affinity error is around 2 kcal/mol, which is a factor of roughly 29 "
            "in predicted affinity. Differences below about 1 kcal/mol are not differences. "
            "Scores also rise with heavy-atom count, so size alone can win."
        ),
        see_also=("ligand_efficiency", "best_score"),
    ),
    # --- efficiency ------------------------------------------------------------------
    Metric(
        key="ligand_efficiency",
        term="Ligand efficiency (LE)",
        short="Binding free energy per heavy atom.",
        group="efficiency",
        direction="higher",
        range="about 0.3 kcal/mol per atom is a common minimum",
        computed=(
            "LE = -dG / heavy atoms. From a measured affinity, LE = 1.364 x pChEMBL / heavy "
            "atoms, since RT ln(10) is 1.364 kcal/mol at 298 K."
        ),
        reading=(
            "Removes the size dependence of raw binding energy, which rises with atom count "
            "because more atoms make more contacts. It is how a series is checked for "
            "improving by growing rather than by binding better."
        ),
        caution=(
            "LE is defined on experimental free energy. Computing it from a docking score "
            "reproduces the size dependence it exists to remove."
        ),
        see_also=("score", "lipophilic_ligand_efficiency", "heavy_atoms"),
    ),
    Metric(
        key="lipophilic_ligand_efficiency",
        term="Lipophilic ligand efficiency (LLE)",
        short="Measured activity discounted by lipophilicity.",
        group="efficiency",
        direction="higher",
        range="5 is a common target",
        computed="LLE = pActivity - cLogP.",
        reading=(
            "Detects potency bought by adding greasy bulk, which usually costs solubility "
            "and metabolic stability."
        ),
        caution=(
            "Requires a measured activity, so it can be computed for retrieved compounds "
            "and not for generated ones."
        ),
        see_also=("ligand_efficiency", "logp", "pchembl"),
    ),
    # --- measured activity -----------------------------------------------------------
    Metric(
        key="pchembl",
        term="pChEMBL",
        short="Measured potency on a negative log10 molar scale.",
        group="activity",
        direction="higher",
        range="5 is 10 uM, 6 is 1 uM, 7 is 100 nM, 9 is 1 nM",
        computed=(
            "ChEMBL's standardised value across IC50, Ki, Kd and EC50. Retrieval filters on "
            "it and orders by it."
        ),
        reading=(
            "The only measured potency in the system. Retrieved actives carry it; generated "
            "compounds never do."
        ),
        caution=(
            "A pChEMBL from a Ki and one from a cell EC50 are not the same measurement. The "
            "assay type travels with each record for this reason."
        ),
        see_also=("lipophilic_ligand_efficiency",),
    ),
    # --- properties ------------------------------------------------------------------
    _descriptor(
        "mw",
        reading="Used for the Lipinski check and as a coarse proxy for size.",
        caution="Heavy-atom count is the better size normaliser for efficiency metrics.",
        see_also=("heavy_atoms", "lipinski_violations"),
    ),
    _descriptor(
        "heavy_atoms",
        reading="The denominator of ligand efficiency. Hydrogens are excluded.",
        see_also=("ligand_efficiency",),
    ),
    _descriptor(
        "logp",
        direction="lower",
        reading=(
            "Crippen cLogP, calculated rather than measured. Rising lipophilicity usually "
            "improves binding and costs solubility, clearance and hERG margin."
        ),
        caution="A calculated value. Measured logD can differ by more than a log unit.",
        see_also=("lipophilic_ligand_efficiency", "lipinski_violations"),
    ),
    _descriptor(
        "tpsa",
        direction="lower",
        reading=(
            "Polar surface area, used as a permeability proxy. Oral absorption is usually "
            "discussed below 140 A^2 and CNS penetration below about 90."
        ),
        see_also=("veber_pass",),
    ),
    _descriptor("hbd", direction="lower", reading="Counts toward the Lipinski check.",
                see_also=("lipinski_violations",)),
    _descriptor("hba", direction="lower", reading="Counts toward the Lipinski check.",
                see_also=("lipinski_violations",)),
    _descriptor(
        "rotatable_bonds",
        direction="lower",
        reading=(
            "A flexibility measure. A compound with many rotatable bonds has many accessible "
            "conformations, so a single generated conformer represents it poorly."
        ),
        see_also=("veber_pass",),
    ),
    _descriptor("fsp3", reading="Fraction of sp3 carbons, used as a three-dimensionality proxy."),
    Metric(
        key="qed",
        term=SCORE_DOC["qed"][0],
        short="A single drug-likeness score combining eight physicochemical properties.",
        group="property",
        direction="higher",
        range=SCORE_DOC["qed"][1],
        computed="RDKit's implementation of the quantitative estimate of drug-likeness.",
        reading="A quick comparison across a batch. Approved oral drugs cluster high.",
        caution=(
            "A mean QED over a batch says little about whether any individual compound is "
            "viable. Read the distribution."
        ),
        see_also=("lipinski_violations",),
    ),
    Metric(
        key="sa_score",
        term=SCORE_DOC["sa_score"][0],
        short="An estimate of how difficult a compound is to synthesise.",
        group="property",
        direction="lower",
        range=SCORE_DOC["sa_score"][1],
        computed=(
            "The Ertl and Schuffenhauer score, from fragment frequencies in known chemistry "
            "with penalties for size, stereocentres and unusual rings."
        ),
        reading=(
            "Filter on it before ranking by score, so an unpreparable compound cannot take "
            "the top position. Calibrate against the retrieved actives in the same run."
        ),
        caution=(
            "A fragment-frequency heuristic, not a route. It does not consider reagent "
            "availability."
        ),
    ),
    # --- flags -----------------------------------------------------------------------
    Metric(
        key="lipinski_violations",
        term=FLAG_DOC["lipinski_violations"][0],
        short="Count of the four rule-of-five criteria a compound breaks.",
        group="flag",
        direction="lower",
        range="0 to 4",
        computed=FLAG_DOC["lipinski_violations"][1],
        reading="Triage for oral absorption, applied early and cheaply.",
        caution="Several approved drugs violate it. Use it to prioritise, not to exclude.",
        see_also=("mw", "logp", "hbd", "hba"),
    ),
    Metric(
        key="veber_pass",
        term=FLAG_DOC["veber_pass"][0],
        short="Whether a compound meets Veber's two criteria for oral bioavailability.",
        group="flag",
        direction="higher",
        range="pass or fail",
        computed=FLAG_DOC["veber_pass"][1],
        reading="Complements Lipinski with flexibility and polarity terms.",
        see_also=("tpsa", "rotatable_bonds"),
    ),
    Metric(
        key="pains_alert",
        term=FLAG_DOC["pains_alert"][0],
        short="Whether a compound contains a pan-assay interference substructure.",
        group="flag",
        direction="lower",
        range="flag per compound, with the alert named",
        computed="RDKit's PAINS filter catalogue.",
        reading=(
            "Indicates an assay result to verify rather than a compound to discard. "
            "Reported, never enforced."
        ),
        caution="Some approved drugs contain PAINS motifs. The flag is a prompt to check.",
    ),
    # --- similarity ------------------------------------------------------------------
    Metric(
        key="tanimoto",
        term="Tanimoto similarity",
        short="Structural similarity between two compounds, from binary fingerprints.",
        group="similarity",
        direction="context",
        range="0 to 1",
        computed=(
            "ECFP4 at 2048 bits, computed locally for every hit rather than taken from the "
            "bank, because ChEMBL and PubChem report different quantities and PubChem "
            "reports none."
        ),
        reading=(
            "Above about 0.7 a compound belongs to the same series as its neighbour. Below "
            "about 0.4 the structure-activity relationship should not be assumed to carry "
            "over."
        ),
        caution=(
            "A clean similarity search is not freedom to operate. Patent claims cover "
            "structures that were never synthesised and so appear in no bank."
        ),
        see_also=("resemblance_to_actives",),
    ),
    Metric(
        key="resemblance_to_actives",
        term="Resemblance to actives",
        short="Mean Tanimoto similarity of a batch to the run's known actives.",
        group="similarity",
        direction="context",
        range="0 to 1",
        computed="Mean of each compound's similarity to the retrieved active set.",
        reading=(
            "High values mean the generator is producing chemistry the target already has. "
            "Whether that is wanted depends on the goal: it is reassuring for lead "
            "optimisation and a failure for a novelty goal."
        ),
        see_also=("tanimoto",),
    ),
    # --- run accounting --------------------------------------------------------------
    Metric(
        key="spent",
        term="Budget spent",
        short="Budget units consumed by subsystem calls so far.",
        group="run",
        direction="context",
        range="0 to the run budget",
        computed="Summed per call from each subsystem's declared unit cost.",
        reading=(
            "The denominator for every rate metric. Two runs are comparable only at equal "
            "budget."
        ),
        see_also=("discovery_auc",),
    ),
    Metric(
        key="yield",
        term="Generation yield",
        short="Accepted children per attempt in a generation call.",
        group="run",
        direction="higher",
        range="0% to 100%",
        computed="Accepted compounds divided by attempts, counted per operator.",
        reading=(
            "A falling yield means the seed pool is close to exhausted. Crossover fails "
            "silently on rigid scaffolds, and the per-operator counts show when mutation is "
            "quietly supplying the whole quota."
        ),
    ),
)

BY_KEY: dict[str, Metric] = {m.key: m for m in _METRICS}

# Column headings and record keys that name the same quantity under another spelling. The UI
# looks a term up by whatever string it happens to be rendering, so the aliases live with the
# definitions rather than in the frontend.
ALIASES = {
    "best": "best_score",
    "vina": "score",
    "docking_score": "score",
    "le": "ligand_efficiency",
    "lle": "lipophilic_ligand_efficiency",
    "pchembl_value": "pchembl",
    "sa": "sa_score",
    "qed_score": "qed",
    "pains": "pains_alert",
    "lipinski": "lipinski_violations",
    "veber": "veber_pass",
    "similarity": "tanimoto",
    "budget": "spent",
    "molecular_weight": "mw",
    "clogp": "logp",
}


def lookup(name: str) -> Metric | None:
    """Find a metric by key, alias or display term. Case and punctuation insensitive."""
    if not name:
        return None
    k = name.strip().lower().replace(" ", "_").replace("-", "_")
    if k in BY_KEY:
        return BY_KEY[k]
    if k in ALIASES:
        return BY_KEY[ALIASES[k]]
    for m in _METRICS:
        if m.term.lower() == name.strip().lower():
            return m
    return None


def as_dict() -> dict:
    """The payload ``GET /api/glossary`` serves."""
    return {
        "groups": GROUPS,
        "metrics": [m.to_dict() for m in _METRICS],
        "aliases": ALIASES,
    }


__all__ = ["ALIASES", "BY_KEY", "GROUPS", "Metric", "as_dict", "lookup"]
