"""The pose store: naming, bucketing, and the index a run leaves behind.

The naming tests are the point of this file. Poses were previously named by Python's
``hash()``, which is salted per process, so the same molecule landed under a different
filename on every run and no pose could be traced back to the molecule it belonged to.
A test that only checked "a file appears" would have passed against that bug, so these
check the property that actually matters: the same molecule always gets the same name,
in this process and in any other.
"""

from __future__ import annotations

import json
import subprocess
import sys

from drugorch.prep import poses

KEY = "docking|v1+v1|7LL4:A|ex8"
OTHER_KEY = "docking|v1+v1|7LL4:A|ex16"


def test_pose_name_is_stable_within_a_process():
    assert poses.pose_name("CCO") == poses.pose_name("CCO")


def test_pose_name_is_stable_across_processes():
    """The regression this module exists for.

    ``hash()`` is salted per interpreter unless PYTHONHASHSEED is pinned, so this is the
    assertion the old naming scheme could not satisfy.
    """
    code = "from drugorch.prep import poses; print(poses.pose_name('CCO'))"
    runs = {
        subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        ).stdout.strip()
        for _ in range(2)
    }
    assert len(runs) == 1
    assert runs.pop() == poses.pose_name("CCO")


def test_different_molecules_get_different_names():
    assert poses.pose_name("CCO") != poses.pose_name("CCC")


def test_bucket_separates_docking_configurations():
    """Two exhaustiveness settings produce different numbers, so they must not share files."""
    assert poses.bucket(KEY) != poses.bucket(OTHER_KEY)


def test_bucket_is_a_legal_directory_name():
    """The cache key contains '|' and ':', neither legal on Windows."""
    name = poses.bucket(KEY)
    assert name.isalnum()
    assert not (set(name) & set('|:/\\<>"?*'))


def test_put_then_pose_path_round_trips(tmp_path):
    written = poses.put(tmp_path, KEY, "CCO", "MODEL 1\nREMARK VINA RESULT: -7.5\n")
    assert written.is_file()
    assert written == poses.pose_path(tmp_path, KEY, "CCO")
    assert "VINA RESULT" in written.read_text(encoding="utf-8")


def test_index_round_trips(tmp_path):
    entries = [
        {"smiles": "CCO", "score": 7.5, "file": "abc.pdbqt", "from_cache": False},
        {"smiles": "CCC", "score": 6.1, "file": None, "from_cache": True},
    ]
    poses.write_index(tmp_path, KEY, entries, receptor={"pdb_id": "7LL4"})

    got = poses.read_index(tmp_path)
    assert got["bucket"] == poses.bucket(KEY)
    assert got["receptor"] == {"pdb_id": "7LL4"}
    assert got["poses"] == entries


def test_read_index_on_missing_directory_returns_empty_shape(tmp_path):
    """Callers should not have to distinguish 'no poses' from 'no run'."""
    got = poses.read_index(tmp_path / "nope")
    assert got == {"bucket": None, "receptor": None, "poses": []}


def test_read_index_survives_a_corrupt_file(tmp_path):
    (tmp_path / "index.json").write_text("{not json", encoding="utf-8")
    assert poses.read_index(tmp_path)["poses"] == []


def test_index_is_json_serialisable_as_written(tmp_path):
    poses.write_index(tmp_path, KEY, [], receptor=None)
    json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
