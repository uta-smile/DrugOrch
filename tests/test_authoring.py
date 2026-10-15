"""The exported tool template, checked by using it.

The claim this module makes to a user is that the template is a working subsystem, not a
sketch: save it, restart, and it is routable. A test that only asserted the text was
non-empty would pass against a template with a typo in the factory name, a `Capability`
field that no longer exists, or a `run` signature the orchestrator cannot call. So these
write the exported text to a directory, load it through the real plugin loader, and call it.

The contract export is checked the same way, against the dataclasses rather than against a
copy of what they said when this was written.
"""

from __future__ import annotations

import threading
from dataclasses import fields

import pytest

from drugorch.subsystems.authoring import TEMPLATE, authoring_info, contract
from drugorch.subsystems.base import Capability, ParamSpec, SubsystemResult
from drugorch.subsystems.plugins import FACTORY_NAME, discover_paths, instantiate


@pytest.fixture
def plugin_dir(tmp_path):
    """The exported template, written where the loader looks for it."""
    d = tmp_path / "tools"
    d.mkdir()
    (d / "example_tool.py").write_text(TEMPLATE, encoding="utf-8")
    return d


class _Ctx:
    settings = None
    cache = None


class TestTemplate:
    def test_it_is_valid_python(self):
        compile(TEMPLATE, "example_tool.py", "exec")

    def test_the_loader_finds_it(self, plugin_dir):
        found, failures = discover_paths([plugin_dir])

        assert not failures, failures
        assert [d.name for d in found] == ["example_tool"]

    def test_it_loads_and_declares_a_usable_capability(self, plugin_dir):
        found, _ = discover_paths([plugin_dir])
        instances, _, failures = instantiate(found, _Ctx())

        assert not failures, failures
        cap = instances[0].capability
        assert isinstance(cap, Capability), "capability is an attribute, not a method"

        assert cap.key and cap.summary
        # The declaration has to survive the two transforms the run loop applies to it.
        assert cap.params_schema()["type"] == "object"
        assert cap.manifest_line().startswith(f"- {cap.key}:")

    def test_its_parameters_coerce(self, plugin_dir):
        found, _ = discover_paths([plugin_dir])
        instances, _, _ = instantiate(found, _Ctx())
        cap = instances[0].capability

        out = cap.coerce_params({"limit": 10_000, "mode": "nonsense"})

        assert out["limit"] == 100, "a value above the declared maximum should clamp"
        assert out["mode"] == "fast", "an undeclared enum member should fall back to default"

    def test_it_runs_and_returns_records(self, plugin_dir, state):
        found, _ = discover_paths([plugin_dir])
        instances, _, _ = instantiate(found, _Ctx())

        state.add_records([{"smiles": s} for s in ["CCO", "CCN", "CCC", "c1ccccc1"]], "test")
        result = instances[0].run({"limit": 3}, state, cancel=threading.Event(), progress=None)

        assert isinstance(result, SubsystemResult)
        assert result.ok, result.error
        assert 0 < len(result.records) <= 3
        assert all("smiles" in r and "scores" in r for r in result.records)
        assert result.explain, "the template should demonstrate the explain blob"

    def test_it_reports_failure_rather_than_raising(self, plugin_dir):
        """The one invariant the run loop depends on."""
        found, _ = discover_paths([plugin_dir])
        instances, _, _ = instantiate(found, _Ctx())

        result = instances[0].run(
            {"limit": 3}, object(), cancel=threading.Event(), progress=None
        )  # a state with no .molecules

        assert result.ok is False
        assert result.error

    def test_it_honours_cancellation(self, plugin_dir, state):
        found, _ = discover_paths([plugin_dir])
        instances, _, _ = instantiate(found, _Ctx())
        cancelled = threading.Event()
        cancelled.set()

        state.add_records([{"smiles": "CCO"}], "test")
        result = instances[0].run(state=state, params={"limit": 50},
                                  cancel=cancelled, progress=None)

        assert result.ok
        assert result.records == []

    def test_it_defines_the_factory_the_loader_looks_for(self):
        assert f"def {FACTORY_NAME}(" in TEMPLATE

    def test_it_says_there_is_no_sandbox(self):
        assert "sandbox" in TEMPLATE.lower()


class TestContract:
    def test_capability_fields_match_the_dataclass(self):
        exported = {f["name"] for f in contract()["capability_fields"]}

        assert exported == {f.name for f in fields(Capability)}

    def test_param_spec_fields_match_the_dataclass(self):
        exported = {f["name"] for f in contract()["param_spec_fields"]}

        assert exported == {f.name for f in fields(ParamSpec)}

    def test_the_parameter_types_are_resolved_not_left_as_source_text(self):
        """`from __future__ import annotations` turns every annotation into a string.

        Reading `field.type` directly returned an empty list here and did it silently, so
        the exported contract advertised no parameter types at all.
        """
        types = contract()["param_types"]

        assert "int" in types and "enum" in types and "smiles_list" in types

    def test_every_type_renders_as_something_an_author_could_type(self):
        rendered = [f["type"] for f in contract()["capability_fields"]]
        rendered += [f["type"] for f in contract()["param_spec_fields"]]

        for t in rendered:
            assert "Ellipsis" not in t, t
            assert "UnionType" not in t, t
            assert "NoneType" not in t, t

    def test_the_template_only_uses_declared_parameter_types(self):
        declared = set(contract()["param_types"])
        used = {
            line.split('type="')[1].split('"')[0]
            for line in TEMPLATE.splitlines()
            if 'type="' in line
        }

        assert used <= declared, f"template uses undeclared param types: {used - declared}"


class TestAuthoringInfo:
    class _Settings:
        plugin_paths = ()
        plugins_allow = ()

    def test_it_reports_where_files_go(self, tmp_path):
        class S:
            plugin_paths = (tmp_path,)
            plugins_allow = ()

        info = authoring_info(S())

        assert info["plugin_paths"][0]["path"] == str(tmp_path)
        assert info["plugin_paths"][0]["exists"] is True

    def test_a_missing_directory_is_reported_rather_than_omitted(self, tmp_path):
        class S:
            plugin_paths = (tmp_path / "absent",)
            plugins_allow = ()

        info = authoring_info(S())

        assert info["plugin_paths"][0]["exists"] is False

    def test_it_carries_the_template_and_the_contract(self):
        info = authoring_info(self._Settings())

        assert info["template"] == TEMPLATE
        assert info["contract"]["factory_name"] == FACTORY_NAME

    def test_load_failures_travel_when_a_registry_is_given(self):
        """A tool with a syntax error is otherwise simply absent, with no explanation."""

        class Failure:
            name = "broken"  # the filter checks this against what loaded

            @staticmethod
            def to_dict():
                return {"name": "broken", "source": "path:x.py", "error": "SyntaxError: bad"}

        class Reg:
            load_failures = [Failure()]
            origins = {}

        info = authoring_info(self._Settings(), Reg())

        assert info["load_failures"] == [
            {"name": "broken", "source": "path:x.py", "error": "SyntaxError: bad"}
        ]


class TestEndToEnd:
    """The claim made to the user: save the file, and the policy can call it.

    Everything above tests a piece. This tests the sentence the Tools panel prints, because
    each piece can pass while the whole fails: a factory that loads but produces something
    the registry rejects, or a capability whose key collides with a built-in, would both
    slip past the unit tests.
    """

    def test_the_template_becomes_a_routable_subsystem(self, plugin_dir):
        from drugorch.config import get_settings
        from drugorch.subsystems.registry import build_registry

        settings = get_settings().model_copy(update={"plugin_paths": (plugin_dir,)})
        reg = build_registry(settings)

        assert not reg.load_failures, [f.to_dict() for f in reg.load_failures]
        assert "example" in reg, "the saved template should appear beside the built-ins"
        assert reg.origins["example"].source.startswith("path:")

    def test_it_reaches_the_prompt_the_policy_reads(self, plugin_dir):
        """A tool absent from the manifest cannot be chosen, however well it loads."""
        from drugorch.config import get_settings
        from drugorch.subsystems.registry import build_registry

        settings = get_settings().model_copy(update={"plugin_paths": (plugin_dir,)})
        manifest = build_registry(settings).manifest()

        assert "example" in manifest
        assert "Example tool" in manifest
        assert "param limit" in manifest, "declared parameters should reach the prompt"

    def test_a_broken_tool_is_reported_rather_than_silently_absent(self, tmp_path):
        """The failure mode this export exists to make visible."""
        from drugorch.config import get_settings
        from drugorch.subsystems.registry import build_registry

        d = tmp_path / "broken"
        d.mkdir()
        (d / "broken_tool.py").write_text("def build(ctx):\n    syntax error here\n", "utf-8")

        settings = get_settings().model_copy(update={"plugin_paths": (d,)})
        reg = build_registry(settings)

        assert reg.load_failures, "a tool that cannot import must be recorded"
        assert "broken" in reg.load_failures[0].name
        assert "Error" in reg.load_failures[0].error


class TestExportedMethods:
    """The `methods` key, which a reader of the Contract tab follows literally.

    It was hand-written while every other key was generated, and it was wrong in all three
    entries: `capability()` as a method rather than an attribute, an `available()` hook that
    does not exist, and `run`'s parameters in the wrong order. A subsystem written to it
    raises inside `plugins.instantiate`, which escapes `build_registry` and makes
    `/api/manifest` return 500, so the whole interface fails to load. These tests tie it to
    the protocol.
    """

    def test_capability_is_published_as_an_attribute(self):
        methods = contract()["methods"]

        assert "capability" in methods, "capability is a data attribute"
        assert "capability()" not in methods, "publishing it as a method bricks the registry"
        assert "attribute" in methods["capability"]

    def test_every_protocol_member_is_published(self):
        from drugorch.subsystems.base import Subsystem

        published = contract()["methods"]
        for name in Subsystem.__protocol_attrs__:
            assert any(k == name or k.startswith(name + "(") for k in published), name

    def test_nothing_is_published_that_the_protocol_does_not_have(self):
        from drugorch.subsystems.base import Subsystem

        for key in contract()["methods"]:
            name = key.split("(")[0]
            assert name in Subsystem.__protocol_attrs__, f"{name} is not part of the contract"

    def test_run_is_published_with_its_real_signature(self):
        """params before state, and cancel/progress keyword-only."""
        key = next(k for k in contract()["methods"] if k.startswith("run("))

        assert key.index("params") < key.index("state"), "the order was published reversed"
        assert "*" in key, "cancel and progress are keyword-only"

    def test_result_keys_are_derived_from_the_dataclass(self):
        """A hand-written copy had already lost `artifacts`."""
        from drugorch.subsystems.base import SubsystemResult

        assert contract()["result_keys"] == [f.name for f in fields(SubsystemResult)]

    def test_a_subsystem_written_to_the_contract_actually_loads(self, tmp_path):
        """The end of the chain: follow what is published, and the registry accepts it."""
        from drugorch.config import get_settings
        from drugorch.subsystems.registry import build_registry

        d = tmp_path / "p"
        d.mkdir()
        (d / "contract_tool.py").write_text(TEMPLATE, encoding="utf-8")

        settings = get_settings().model_copy(update={"plugin_paths": (d,)})
        reg = build_registry(settings)  # must not raise

        assert "example" in reg


BROKEN_SOURCE = "def build(ctx):" + chr(10) + "    nope nope" + chr(10)


class TestFailureReporting:
    def test_a_tool_that_loaded_is_not_also_reported_as_failed(self):
        """With an allow-list, the registry records a failure for each built-in it filters
        and then re-adds them, which put a red "5 tools failed to load" banner directly
        above those same five listed as loaded."""
        from drugorch.config import get_settings
        from drugorch.subsystems.registry import build_registry

        settings = get_settings().model_copy(update={"plugins_allow": ("nothing_matches",)})
        info = authoring_info(settings, build_registry(settings))

        loaded = {entry["key"] for entry in info["loaded"]}
        reported = {f["name"] for f in info["load_failures"]}

        assert not (loaded & reported), f"reported as both loaded and failed: {loaded & reported}"

    def test_a_genuinely_broken_tool_is_still_reported(self, tmp_path):
        """The filter must not swallow the case the panel exists for."""
        from drugorch.config import get_settings
        from drugorch.subsystems.registry import build_registry

        d = tmp_path / "p"
        d.mkdir()
        (d / "snapped.py").write_text(BROKEN_SOURCE, encoding="utf-8")

        settings = get_settings().model_copy(update={"plugin_paths": (d,)})
        info = authoring_info(settings, build_registry(settings))

        assert any("snapped" in f["name"] for f in info["load_failures"])
