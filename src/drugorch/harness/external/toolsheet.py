"""What each side brings to the benchmark, tool by tool.

A score comparison between two agents is only interpretable next to what each one was
allowed to do. MolDesignBench gives its models seventeen tools; DrugOrch answers the same
instances with four subsystems. Those are different experiments, and the difference belongs
on screen rather than in a paragraph of a paper nobody opens beside the table.

Their side is **read out of their source**, never retyped. Each of their tools is a class
carrying a ``name``, a ``description`` and a parameter schema, and those strings are what
their agents actually see in the function-calling payload. Extraction is by :mod:`ast`
rather than by import: importing ``moldesign`` pulls in LLM clients, model downloads and a
tool server, none of which a documentation page should require.

Our side is written here, in the same shape, with one addition: which of their tools each
subsystem stands in for. A reader comparing the two boards needs to know that our pool
search and their ``query_moses_by_fg_counts_prop_ranges`` hit the same 1.9 million
molecules, and that nothing of ours stands in for their neural generator.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from .panel import BENCH_REPO

TOOLS_DIR = Path("moldesign/tools")

# Their roster, grouped the way a reader needs it rather than the way the registry lists it.
# The group names are ours; the membership follows what each tool does.
GROUPS: dict[str, str] = {
    "analyze_overall_molecule_properties": "measurement",
    "get_functional_groups": "measurement",
    "count_molecule_atoms": "measurement",
    "calculate_molecular_weight": "measurement",
    "calculate_tanimoto_similarity": "measurement",
    "query_moses_by_fg_counts_prop_ranges": "retrieval",
    "search_pubchem": "retrieval",
    "search_wikipedia": "retrieval",
    "generate_molecules_by_property_ranges": "generation",
    "optimize_molecule_by_prop_ranges": "generation",
    "generate_molecule_with_fragments": "generation",
    "edit_molecules_by_functional_groups": "generation",
    "canonicalize_smiles": "representation",
    "convert_iupac_to_smiles": "representation",
    "convert_chemical_name_to_smiles": "representation",
    "convert_selfies_to_smiles": "representation",
    "run_python_code": "general",
}

GROUP_TITLES = {
    "measurement": "Measurement",
    "retrieval": "Retrieval",
    "generation": "Generation and editing",
    "representation": "Representation and naming",
    "general": "General compute",
}


@dataclass
class ToolDoc:
    """One tool, as the agent using it sees it."""

    name: str
    description: str = ""
    parameters: dict[str, str] = field(default_factory=dict)
    group: str = "general"
    source: str = ""
    side: str = "benchmark"
    counterpart: str = ""
    """For our tools: the benchmark tool it stands in for, or an empty string."""

    detail: str = ""
    """For our tools: how it works here, which the description alone does not say."""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "group": self.group,
            "group_title": GROUP_TITLES.get(self.group, self.group),
            "source": self.source,
            "side": self.side,
            "counterpart": self.counterpart,
            "detail": self.detail,
        }


def _string(node: ast.AST) -> str:
    """A class attribute's value, when it is a plain string."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):  # an f-string: not resolvable without running it
        return ""
    return ""


def _parameters(node: ast.AST) -> dict[str, str]:
    """The ``properties`` dict of a tool class, flattened to name -> description."""
    out: dict[str, str] = {}
    if not isinstance(node, ast.Dict):
        return out
    # strict=True: an ast.Dict always pairs keys with values, and a mismatch would mean
    # the parse is wrong rather than that a pair is missing.
    for key, value in zip(node.keys, node.values, strict=True):
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
            continue
        desc = ""
        if isinstance(value, ast.Dict):
            for k2, v2 in zip(value.keys, value.values, strict=True):
                if isinstance(k2, ast.Constant) and k2.value == "description":
                    desc = _string(v2)
        out[key.value] = desc
    return out


def their_tools(repo: Path | str = BENCH_REPO) -> list[ToolDoc]:
    """Every tool the benchmark registers, parsed from its own source.

    Returns an empty list when the sources are not vendored, so a page can say the roster is
    unavailable rather than invent one.
    """
    root = Path(repo) / TOOLS_DIR
    if not root.is_dir():
        return []

    found: list[ToolDoc] = []
    for path in sorted(root.glob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            name = description = ""
            params: dict[str, str] = {}
            for stmt in node.body:
                # Both spellings appear in their sources: `name = '...'` and
                # `name: str = '...'`. Reading only the first silently drops the Python
                # shell, which is the tool with the widest reach of any on the roster.
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                    target, value = stmt.targets[0], stmt.value
                elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                    target, value = stmt.target, stmt.value
                else:
                    continue
                if not isinstance(target, ast.Name):
                    continue
                if target.id == "name":
                    name = _string(value)
                elif target.id == "description":
                    description = _string(value)
                elif target.id == "properties":
                    params = _parameters(value)
            if name and description:
                found.append(
                    ToolDoc(
                        name=name,
                        description=description,
                        parameters=params,
                        group=GROUPS.get(name, "general"),
                        source=f"{TOOLS_DIR.as_posix()}/{path.name}",
                        side="benchmark",
                    )
                )
    found.sort(key=lambda t: (list(GROUP_TITLES).index(t.group), t.name))
    return found


# --- our side ----------------------------------------------------------------------
#
# Written out rather than generated from the subsystem registry, because the registry
# describes what a subsystem does in a target-driven run and this page is about what the arm
# can do on this benchmark. `detail` carries what a one-line description cannot: the limits,
# because a tool's limits are most of what decides whether a score is reachable with it.

OUR_TOOLS: tuple[ToolDoc, ...] = (
    ToolDoc(
        name="measure",
        side="drugorch",
        group="measurement",
        counterpart="analyze_overall_molecule_properties, get_functional_groups",
        description=(
            "Measure a batch of molecules: twenty physicochemical and predicted properties "
            "(MW, logP, logD, logS, HBD, HBA, TPSA, rotatable bonds, ring counts, QED, "
            "BBBP, mutagenicity, molar refractivity, heavy atoms, formal charge, Fsp3, "
            "heteroatoms, topological diameter, HIA) and the counts of fifty functional "
            "groups."
        ),
        detail=(
            "This is the benchmark's own analyzer and its own functional-group counter, run "
            "in the environment its answer key was built in. It is not a reimplementation. "
            "Batched: a whole generation of candidates is measured in one call, because the "
            "model load behind it costs about five seconds and a measurement about half a "
            "second. An unparseable molecule comes back empty rather than raising."
        ),
        parameters={"smiles": "The molecules to measure."},
        source="harness/external/panel.py",
    ),
    ToolDoc(
        name="search_pool",
        side="drugorch",
        group="retrieval",
        counterpart="query_moses_by_fg_counts_prop_ranges",
        description=(
            "Retrieve molecules from 1.9 million MOSES compounds, pre-tagged with the same "
            "property predictions the benchmark scores against, that satisfy the current "
            "specification's property windows and functional-group requirements."
        ),
        detail=(
            "The same pool their retrieval tool uses, queried directly from the partitioned "
            "parquet, which takes about a tenth of a second over the full set. Two "
            "differences from theirs: up to 24 molecules are returned rather than 5, and a "
            "query matching nothing is relaxed one window at a time, widest first, with the "
            "response naming what was dropped. An agent told only that nothing matched "
            "tends to conclude the instance is impossible, and nine in ten are not."
        ),
        parameters={
            "spec": "The current constraint specification.",
            "limit": "How many molecules to return, at most 24.",
        },
        source="harness/external/pool.py",
    ),
    ToolDoc(
        name="generate",
        side="drugorch",
        group="generation",
        counterpart="no equivalent: theirs are MolMIM-based",
        description=(
            "Evolve the molecules found so far toward the specification with a graph genetic "
            "algorithm, scoring every child by how many constraints it satisfies."
        ),
        detail=(
            "DrugOrch's own GraphGA operators, the ones a target-driven run uses: parents "
            "are cut at acyclic single bonds and recombined, then perturbed by single-atom "
            "and single-bond edits. Fitness counts satisfied constraints first and uses "
            "normalised distance only to break ties, because the benchmark scores every "
            "constraint at once rather than an average miss. Each round measures its whole "
            "population in one call and stops as soon as a child satisfies everything. It "
            "repairs a constraint or two; it does not invent a chemotype."
        ),
        parameters={
            "seeds": "Molecules to evolve from.",
            "rounds": "Generations to run, 1 to 8.",
        },
        source="harness/external/ga.py",
    ),
    ToolDoc(
        name="propose",
        side="drugorch",
        group="generation",
        counterpart="generate_molecule_with_fragments (loosely)",
        description=(
            "Measure molecules written directly by the model, adding them to the candidate "
            "pool."
        ),
        detail=(
            "The route for chemistry the pool does not hold and the generator will not "
            "reach from its seeds. The molecules are the model's own; what this adds is "
            "that they are measured rather than assumed, so a proposal that misses the "
            "windows is visible on the next decision instead of being answered."
        ),
        parameters={"smiles": "Up to twelve molecules to measure."},
        source="harness/external/agent.py",
    ),
)


def gaps(theirs: list[ToolDoc]) -> list[dict]:
    """Capabilities on their side with nothing standing in for them on ours.

    Listed because the tool count is the first thing a reader will compare, and a bare count
    says nothing about which capabilities are missing or whether they bear on the score.
    """
    covered = {
        "analyze_overall_molecule_properties",
        "get_functional_groups",
        "query_moses_by_fg_counts_prop_ranges",
    }
    out = []
    for t in theirs:
        if t.name in covered:
            continue
        out.append({"name": t.name, "group": t.group, "description": t.description})
    return out


def sheet(repo: Path | str = BENCH_REPO) -> dict:
    """The whole comparison, for the page that renders it."""
    theirs = their_tools(repo)
    return {
        "benchmark": [t.to_dict() for t in theirs],
        "drugorch": [t.to_dict() for t in OUR_TOOLS],
        "groups": GROUP_TITLES,
        "n_benchmark": len(theirs),
        "n_drugorch": len(OUR_TOOLS),
        "uncovered": gaps(theirs),
        "note": (
            "Benchmark tool names, descriptions and parameters are read from "
            "moldesign/tools/*.py in the vendored sources. They are the strings the "
            "benchmark's own agents are shown."
        ),
    }


__all__ = ["GROUPS", "GROUP_TITLES", "OUR_TOOLS", "ToolDoc", "gaps", "sheet", "their_tools"]
