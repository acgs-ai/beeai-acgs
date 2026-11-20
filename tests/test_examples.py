# SPDX-FileCopyrightText: 2026-present ACGS Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""The README only shows code that runs, and the quickstart output it shows is the real output."""

from __future__ import annotations

import json
import os
import re
import runpy
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.requirement.requirements.ask_permission import AskPermissionRequirement

from beeai_acgs import ACGSMiddleware, io_confirm_approver
from tests.fakes import Tools, model, review_rules

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
QUICKSTART = ROOT / "examples" / "quickstart.py"


def _blocks(language: str) -> list[str]:
    return [textwrap.dedent(block) for block in re.findall(rf"```{language}\n(.*?)```", README, re.DOTALL)]


def test_quickstart_output_matches_readme() -> None:
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-W", "error", str(QUICKSTART)],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
        timeout=300,
    )
    [expected] = [block for block in _blocks("text") if block.startswith("review:")]
    assert completed.stdout == expected
    assert completed.stderr == ""


def test_readme_embeds_the_quickstart_verbatim() -> None:
    assert QUICKSTART.read_text(encoding="utf-8") in _blocks("python")


def test_deny_and_audit_runs_offline_without_api_keys(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for key in list(os.environ):
        if key.endswith("API_KEY"):
            monkeypatch.delenv(key)
    # Register the example's import-time environment change so monkeypatch restores it.
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    # pytest-socket disables network access for this in-process example execution.
    runpy.run_path(str(ROOT / "examples" / "deny_and_audit.py"), run_name="__main__")
    output = capsys.readouterr().out
    execution, audit = output.split("Audit trail (application decisions):\n")
    assert execution == "Executed record_note: The meeting starts at noon.\n"
    decisions = [json.loads(line) for line in audit.splitlines()]
    assert [decision["verdict"] for decision in decisions] == ["allow", "deny"]
    assert [decision["tool_name"] for decision in decisions] == ["record_note", "record_note"]
    assert decisions[0]["rule_ids"] == []
    assert decisions[1]["rule_ids"] == ["ACGS-006"]
    assert all(decision["error"] is None for decision in decisions)
    assert "Blocked by" in decisions[1]["reason"]
    assert "demo-secret" not in output


def test_readme_states_the_supported_versions() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for package in ("beeai-framework", "acgs-lite"):
        [requirement] = re.findall(rf'"({package}>=[^"]+)"', pyproject)
        assert f"`{requirement}`" in README
        [floor] = re.findall(rf'"{package}==([^"]+)"', pyproject)
        assert requirement.startswith(f"{package}>={floor},")
    assert README.count("beeai-framework 0.1.85") == 2
    assert "Requires Python 3.11 to 3.13" in README
    assert 'requires-python = ">=3.11,<3.14"' in pyproject


def test_every_readme_python_block_runs(capsys: pytest.CaptureFixture[str]) -> None:
    blocks = _blocks("python")
    assert len(blocks) == 6
    namespaces: list[dict[str, Any]] = []
    for block in blocks:
        namespace: dict[str, Any] = {
            "__name__": "readme",
            "llm": model("Done."),
            "constitution": review_rules(),
            "tools": Tools().all,
        }
        exec(compile(block, "<README>", "exec", dont_inherit=True), namespace)  # noqa: S102
        namespaces.append(namespace)
    assert "normalized: update config | dry run" in capsys.readouterr().out
    usage = namespaces[0]
    assert isinstance(usage["agent"], RequirementAgent)
    assert [type(m) for m in usage["agent"].middlewares] == [ACGSMiddleware]
    review = namespaces[3]["middleware"]
    assert (review.approver, review.review_timeout) == (io_confirm_approver, 300)
    requirement = namespaces[4]["agent"]
    assert requirement.middlewares == [namespaces[4]["middleware"]]
    assert any(isinstance(r, AskPermissionRequirement) for r in requirement._requirements)
    assert namespaces[-1]["middleware"].agent_id == "payments-app"
