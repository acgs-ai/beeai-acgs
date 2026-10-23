# SPDX-FileCopyrightText: 2026-present ACGS Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Blocked tool calls do not run, allowed ones do, wherever the tool is called inside the governed run."""

from __future__ import annotations

import json
from typing import Any

import pytest
from acgs_lite import Constitution
from beeai_framework.agents.react import ReActAgent
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.requirement.utils._tool import FinalAnswerTool
from beeai_framework.agents.tool_calling import ToolCallingAgent
from beeai_framework.backend import AssistantMessage
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools import StringToolOutput, tool
from beeai_framework.tools.handoff import HandoffTool

from beeai_acgs import ACGSMiddleware, Verdict
from beeai_acgs.middleware import _jsonable, _tool_input
from tests.fakes import (
    BLOCKED,
    HELLO,
    LEAK,
    REFUND,
    ScriptedChatModel,
    Tools,
    agent,
    call,
    model,
    results,
    spy_evaluate,
)


def _guard(**kwargs: Any) -> ACGSMiddleware:
    return ACGSMiddleware(**kwargs)


async def test_blocked_call_does_not_run_and_the_model_reads_why() -> None:
    tools = Tools()
    output = await agent(model(call("send_email", LEAK, "c1"), "Done."), tools.all, middlewares=[_guard()]).run("go")
    assert tools.ran == []
    assert results(output)["c1"].startswith(BLOCKED)
    assert output.last_message.text == "Done."


async def test_allowed_call_runs_unchanged() -> None:
    tools = Tools()
    output = await agent(model(call("send_email", HELLO, "c1"), "Done."), tools.all, middlewares=[_guard()]).run("go")
    assert tools.ran == [("send_email", HELLO)]
    assert results(output)["c1"] == "sent to friend@example.com"


async def test_parallel_calls_are_decided_one_by_one() -> None:
    tools = Tools()
    turn = [call("send_email", LEAK, "c1"), call("send_email", HELLO, "c2"), call("get_weather", {"city": "Oslo"}, "w")]
    output = await agent(model(turn, "Done."), tools.all, middlewares=[_guard()]).run("go")
    assert tools.names == ["send_email", "get_weather"]
    assert results(output)["c1"].startswith(BLOCKED)


async def test_run_level_middleware() -> None:
    tools = Tools()
    governed = agent(model(call("send_email", LEAK, "c1"), "Done."), tools.all)
    output = await governed.run("go").middleware(_guard())
    assert tools.ran == []
    assert results(output)["c1"].startswith(BLOCKED)


async def test_same_middleware_at_agent_and_run_level_decides_once(monkeypatch: pytest.MonkeyPatch) -> None:
    tools = Tools()
    guard = _guard()
    seen: list[str] = []
    spy_evaluate(guard, seen, monkeypatch)
    governed = agent(model(call("send_email", HELLO, "c1"), "Done."), tools.all, middlewares=[guard])
    await governed.run("go").middleware(guard)
    assert seen == ["send_email"]


async def test_middleware_stays_bound_to_one_run_only() -> None:
    tools = Tools()
    guard = _guard()
    governed = agent(model(call("send_email", LEAK, "c1"), "Done."), tools.all)
    await governed.run("go").middleware(guard)
    again = agent(model(call("send_email", LEAK, "c1"), "Done."), tools.all)
    await again.run("go")
    assert tools.ran == [("send_email", LEAK)]


async def test_sub_agent_tool_calls_are_governed() -> None:
    tools = Tools()
    expert = RequirementAgent(
        llm=model(call("send_email", LEAK, "s1"), "Expert done."), tools=tools.all, name="Mailer", description="Mails."
    )
    guard = _guard()
    governed = agent(
        model(call("mailer", {"task": "Email my friend"}, "h1"), "Done."), [HandoffTool(expert)], middlewares=[guard]
    )
    output = await governed.run("go")
    assert tools.ran == []
    assert results(output)["h1"] == "Expert done."


async def test_sub_agent_with_its_own_middleware_is_decided_once(monkeypatch: pytest.MonkeyPatch) -> None:
    tools = Tools()
    guard = _guard()
    seen: list[str] = []
    spy_evaluate(guard, seen, monkeypatch)
    expert = RequirementAgent(
        llm=model(call("send_email", HELLO, "s1"), "Expert done."),
        tools=tools.all,
        name="Mailer",
        description="Mails.",
        middlewares=[guard],
    )
    governed = agent(
        model(call("mailer", {"task": "Say hi"}, "h1"), "Done."), [HandoffTool(expert)], middlewares=[guard]
    )
    await governed.run("go")
    assert seen == ["mailer", "send_email"]
    assert tools.ran == [("send_email", HELLO)]


async def test_handoff_task_itself_is_checked() -> None:
    tools = Tools()
    expert = RequirementAgent(llm=model("never"), tools=tools.all, name="Mailer", description="Mails.")
    governed = agent(
        model(call("mailer", {"task": "The password is hunter2"}, "h1"), "Done."),
        [HandoffTool(expert)],
        middlewares=[_guard()],
    )
    output = await governed.run("go")
    assert results(output)["h1"].startswith("Tool call 'mailer' was not executed. Blocked by ACGS rule ACGS-006")


async def test_tool_that_calls_a_tool_is_governed_inside() -> None:
    tools = Tools()

    @tool()
    async def notify(to: str) -> str:
        """Notify someone."""
        result = await tools.send_email.run({"to": to, "body": "The password is hunter2"})
        return str(result.get_text_content())

    output = await agent(
        model(call("notify", {"to": "a@example.com"}, "n1"), "Done."), [notify], middlewares=[_guard()]
    ).run("go")
    assert tools.ran == []
    assert results(output)["n1"].startswith(BLOCKED)


async def test_direct_tool_run_with_middleware() -> None:
    tools = Tools()
    denied = await tools.send_email.run(LEAK).middleware(_guard())
    allowed = await tools.send_email.run(HELLO).middleware(_guard())
    assert isinstance(denied, StringToolOutput)
    assert denied.get_text_content().startswith(BLOCKED)
    assert allowed.get_text_content() == "sent to friend@example.com"
    assert tools.ran == [("send_email", HELLO)]


async def test_direct_tool_run_without_middleware_is_not_governed() -> None:
    tools = Tools()
    await tools.send_email.run(LEAK)
    assert tools.ran == [("send_email", LEAK)]


async def test_validated_arguments_are_evaluated_including_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    @tool()
    def search(query: str, scope: str = "the password is hunter2") -> str:
        """Search."""
        return "found"

    output = await search.run({"query": "docs"}).middleware(_guard())
    assert output.get_text_content().startswith("Tool call 'search' was not executed. Blocked by ACGS rule ACGS-006")


async def test_invalid_input_is_evaluated_as_sent_and_the_tool_rejects_it() -> None:
    tools = Tools()
    output = await agent(
        model(call("send_email", {"to": "a@example.com", "body": 7, "note": "hunter2 password"}, "c1"), "Done."),
        tools.all,
        middlewares=[_guard()],
    ).run("go")
    assert tools.ran == []
    assert results(output)["c1"].startswith(BLOCKED)


def test_tool_input_helpers() -> None:
    tools = Tools()
    assert _tool_input(tools.issue_refund, {"order_id": "A", "amount": "2"}) == {"order_id": "A", "amount": 2.0}
    assert _tool_input(tools.issue_refund, {"order_id": "A"}) == {"order_id": "A"}
    assert _tool_input(tools.issue_refund, "refund A") == {"input": "refund A"}
    assert _jsonable(tools.issue_refund.input_schema(order_id="A", amount=1)) == {"order_id": "A", "amount": 1.0}


async def test_final_answer_tool_is_not_governed() -> None:
    tools = Tools()
    output = await agent(model("The password is hunter2"), tools.all, middlewares=[_guard()]).run("go")
    assert output.last_message.text == "The password is hunter2"
    assert isinstance(FinalAnswerTool, type)


def _react(arguments: dict[str, Any]) -> ScriptedChatModel:
    thought = f"Thought: send it\nFunction Name: send_email\nFunction Input: {json.dumps(arguments)}\n"
    return ScriptedChatModel([[AssistantMessage(thought)], [AssistantMessage("Thought: done\nFinal Answer: Done.\n")]])


@pytest.mark.parametrize(("arguments", "runs"), [(LEAK, False), (HELLO, True)])
async def test_react_agent(arguments: dict[str, Any], runs: bool) -> None:
    tools = Tools()
    react = ReActAgent(llm=_react(arguments), tools=tools.all, memory=UnconstrainedMemory())
    output = await react.run("go").middleware(_guard())
    assert tools.ran == ([("send_email", arguments)] if runs else [])
    assert output.last_message.text.strip() == "Done."


async def test_tool_calling_agent_final_answer_is_governed() -> None:
    tools = Tools()
    calling = ToolCallingAgent(
        llm=model(
            call("final_answer", {"response": "The password is hunter2"}, "f1"),
            call("final_answer", {"response": "Sorry."}, "f2"),
        ),
        tools=tools.all,
        memory=UnconstrainedMemory(),
    )
    output = await calling.run("go").middleware(_guard())
    assert output.last_message.text == "Sorry."


def test_evaluate_and_repr() -> None:
    guard = ACGSMiddleware(agent_id="app")
    assert guard.evaluate("send_email", LEAK).verdict is Verdict.DENY
    assert guard.evaluate("issue_refund", REFUND).verdict is Verdict.ALLOW
    assert isinstance(guard.constitution, Constitution)
    assert repr(guard) == "ACGSMiddleware(agent_id='app', on_review='ask', approver=None, review_timeout=None)"


def test_formatter_is_used() -> None:
    guard = ACGSMiddleware(tool_call_formatter=lambda name, args: "the password is hunter2")
    assert guard.evaluate("get_weather", {"city": "Oslo"}).verdict is Verdict.DENY


@pytest.mark.parametrize(
    ("kwargs", "message"), [({"on_review": "maybe"}, "on_review"), ({"review_timeout": 0}, "review_timeout")]
)
def test_invalid_options(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ACGSMiddleware(**kwargs)


async def test_evaluation_error_denies(monkeypatch: pytest.MonkeyPatch) -> None:
    tools = Tools()
    guard = _guard()

    def broken(name: str, args: Any) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr(guard, "evaluate", broken)
    output = await agent(model(call("send_email", HELLO, "c1"), "Done."), tools.all, middlewares=[guard]).run("go")
    assert tools.ran == []
    assert results(output)["c1"] == (
        "Tool call 'send_email' was not executed. ACGS governance could not evaluate this tool call, so it was denied."
    )
    decision = await guard.decide(tools.send_email, HELLO)
    assert decision.error == "RuntimeError: boom"
