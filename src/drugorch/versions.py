"""Recipe version strings, baked into every cache key.

A cache that outlives a change in how its values are computed is worse than no cache: it
blends two recipes into one column of numbers and nothing downstream can tell. So every
cached artefact keys on the version of the code that produced it, and bumping a constant
here invalidates exactly the entries that constant governs -- old rows stay on disk,
unreferenced, rather than being silently served.

Bump a constant when the *meaning* of the output changes, not when the implementation is
refactored. ``tests/test_versions.py`` asserts a bump actually misses.
"""

from __future__ import annotations

# Molecule standardization: parse, largest fragment, neutralize, canonical SMILES.
# Bump when the standardizer pipeline changes -- it changes every cache key in the project,
# because the canonical SMILES *is* the key.
STD_V = "v1"

# The RDKit property panel: which descriptors, and how each is computed.
PANEL_V = "v1"

# Receptor preparation: chain selection, ligand extraction, box derivation, pdbqt writing.
RECPREP_V = "v1"

# Docking: Vina invocation parameters and how the score is parsed.
DOCK_V = "v1"

# Generation: GraphGA crossover/mutation operators and their rates.
GEN_V = "v1"

# The policy prompt and its surrounding scaffolding (system text, state rendering).
PROMPT_V = "v1"

# The decision JSON schema the LLM is constrained to.
SCHEMA_V = "v1"

# HTTP client cache: request keying and response normalization.
HTTP_V = "v1"

# Molecule-first bank search: which banks, the similarity metric, and the safety gate.
# Bump when a safety rule changes what it admits -- the molecules that entered a run are
# part of its result, so a run recorded under a different gate is not comparable.
SIM_V = "v1"

ADMET_V = "v2"
"""ADMET-AI model generation. In the cache key so a model update invalidates rather than
blending two models' predictions into one column.

v2 is admet_ai 1.2.0 as vendored by MolDesignBench, replacing PyPI's 2.0.1. Not an upgrade --
a deliberate step back to the model the external benchmark's answer key was built with. The
two disagree by about the width of that benchmark's tolerance windows (r=0.93 on logS with a
median error of 0.50 against a half-width of 0.75; worse on the classification endpoints),
which is small enough to look like noise and large enough to decide a result."""

FP_V = "v1"
"""Morgan fingerprint recipe: radius and bit count. In the cache key so changing either
invalidates rather than silently blending two different fingerprints into one comparison."""


def all_versions() -> dict[str, str]:
    """Every version constant, for the run provenance blob."""
    return {
        "std": STD_V,
        "panel": PANEL_V,
        "recprep": RECPREP_V,
        "dock": DOCK_V,
        "gen": GEN_V,
        "prompt": PROMPT_V,
        "schema": SCHEMA_V,
        "http": HTTP_V,
        "sim": SIM_V,
        "fp": FP_V,
        "admet": ADMET_V,
    }
