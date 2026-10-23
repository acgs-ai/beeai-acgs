# SPDX-FileCopyrightText: 2026-present ACGS Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Calls that need review run only when a reviewer returns True; no reviewer, an error or a timeout denies them."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from beeai_framework.utils.io import setup_io_context

from beeai_acgs import ACGSMiddleware, ReviewRequest, Verdict, io_confirm_approver
from beeai_acgs import middleware as module
from tests.fakes import (
    BLOCKED,
    HELLO,
    LEAK,
    REFUND,
    REFUND_REVIEW,
    Tools,
    agent,
    call,
    model,
    recording,
    results,
    review_rules,
)

NOT_RUN = "Tool call 'issue_refund' was not executed. "


async def _run(guard: ACGSMiddleware, tools: Tools, **kwargs: Any) -> dict[str, str]:
    output = await agent(
        model(call("issue_refund", REFUND, "r1"), "Done."), tools.all, middlewares=[guard], **kwargs
    ).run("go")
    return results(output)


async def test_review_without_a_reviewer_is_denied() -> None:
    tools = Tools()
    result = await _run(ACGSMiddleware(constitution=review_rules()), tools)
    assert tools.ran == []
    assert result["r1"] == f"{NOT_RUN}{REFUND_REVIEW} No reviewer is configured, so it was denied."


async def test_review_turned_off_is_denied_without_asking() -> None:
    tools = Tools()
    asked: list[ReviewRequest] = []
    guard = ACGSMiddleware(constitution=review_rules(), on_review="deny", approver=recording(asked, True))
    result = await _run(guard, tools)
    assert (tools.ran, asked) == ([], [])
    assert result["r1"] == f"{NOT_RUN}{REFUND_REVIEW} Review is turned off (on_review='deny'), so it was denied."


async def test_approved_call_runs() -> None:
    tools = Tools()
    asked: list[ReviewRequest] = []
    guard = ACGSMiddleware(constitution=review_rules(), approver=recording(asked, True))
    result = await _run(guard, tools)
    assert tools.ran == [("issue_refund", REFUND)]
    assert result["r1"] == "refunded 20.0 for A-17"
    [request] = asked
    assert request.tool is tools.issue_refund
    assert request.arguments == REFUND
    assert request.decision.verdict is Verdict.REVIEW
    assert request.decision.rule_ids == ("REFUND-REVIEW",)
    assert request.run_id
    assert request.prompt == (
        f"ACGS: the agent wants to call the tool `issue_refund`. {REFUND_REVIEW} Approve this call?"
    )


async def test_async_approver() -> None:
    tools = Tools()

    async def approve(request: ReviewRequest) -> bool:
        await asyncio.sleep(0)
        return True

    await _run(ACGSMiddleware(constitution=review_rules(), approver=approve, review_timeout=5), tools)
    assert tools.names == ["issue_refund"]


@pytest.mark.parametrize("answer", [False, None, "yes", 1])
async def test_only_true_approves(answer: Any) -> None:
    tools = Tools()
    result = await _run(ACGSMiddleware(constitution=review_rules(), approver=lambda r: answer), tools)
    assert tools.ran == []
    assert result["r1"] == f"{NOT_RUN}{REFUND_REVIEW} The reviewer did not approve this call, so it was denied."


async def test_failing_reviewer_denies() -> None:
    tools = Tools()

    def broken(request: ReviewRequest) -> bool:
        raise RuntimeError("down")

    result = await _run(ACGSMiddleware(constitution=review_rules(), approver=broken), tools)
    assert tools.ran == []
    assert result["r1"] == f"{NOT_RUN}{REFUND_REVIEW} No approval was received (the reviewer failed), so it was denied."


async def test_slow_reviewer_times_out_and_denies() -> None:
    tools = Tools()

    async def slow(request: ReviewRequest) -> bool:
        await asyncio.sleep(10)
        return True

    result = await _run(ACGSMiddleware(constitution=review_rules(), approver=slow, review_timeout=0.05), tools)
    assert tools.ran == []
    assert (
        result["r1"] == f"{NOT_RUN}{REFUND_REVIEW} No approval was received (no answer within 0.05s), so it was denied."
    )


async def test_blocking_rule_wins_over_review_without_asking() -> None:
    tools = Tools()
    asked: list[ReviewRequest] = []
    guard = ACGSMiddleware(constitution=review_rules(), approver=recording(asked, True))
    leak = {"order_id": "issue refund, the password is hunter2", "amount": 20.0}
    refund_leak = await agent(model(call("issue_refund", leak, "r1"), "Done."), tools.all, middlewares=[guard]).run(
        "go"
    )
    assert asked == []
    assert results(refund_leak)["r1"].startswith(NOT_RUN + "Blocked by ACGS rule ACGS-006")


async def test_io_confirm_approver_uses_the_beeai_io_handlers() -> None:
    prompts: list[tuple[str, dict[str, Any]]] = []

    async def confirm(prompt: str, **kwargs: Any) -> bool:
        prompts.append((prompt, kwargs))
        return True

    async def read(prompt: str) -> str:
        return ""

    reset = setup_io_context(read=read, confirm=confirm)
    try:
        tools = Tools()
        await _run(ACGSMiddleware(constitution=review_rules(), approver=io_confirm_approver), tools)
    finally:
        reset()
    assert tools.names == ["issue_refund"]
    [(prompt, kwargs)] = prompts
    assert prompt.startswith("ACGS: the agent wants to call the tool `issue_refund`.")
    assert kwargs == {"data": REFUND}


async def test_decide_uses_the_given_approver() -> None:
    tools = Tools()
    guard = ACGSMiddleware(constitution=review_rules())
    assert (await guard.decide(tools.issue_refund, REFUND)).verdict is Verdict.DENY
    approved = await guard.decide(tools.issue_refund, REFUND, approver=lambda r: True)
    assert approved.verdict is Verdict.ALLOW
    assert approved.reason == f"{REFUND_REVIEW} A reviewer approved this call."
    assert approved.rule_ids == ("REFUND-REVIEW",)


async def test_review_error_inside_decide_denies(monkeypatch: pytest.MonkeyPatch) -> None:
    tools = Tools()
    guard = ACGSMiddleware(constitution=review_rules(), approver=lambda r: True)

    async def broken(*args: Any) -> Any:
        raise RuntimeError("review crashed")

    monkeypatch.setattr(guard, "_review", broken)
    decision = await guard.decide(tools.issue_refund, REFUND)
    assert decision.verdict is Verdict.DENY
    assert decision.error == "RuntimeError: review crashed"


# --- AskPermissionRequirement -------------------------------------------------------------------------------


async def test_permission_requirement_asks_only_about_review_calls() -> None:
    tools = Tools()
    asked: list[ReviewRequest] = []
    guard = ACGSMiddleware(constitution=review_rules())
    requirement = guard.permission_requirement(approver=recording(asked, True))
    turn = [call("issue_refund", REFUND, "r1"), call("send_email", HELLO, "c1"), call("send_email", LEAK, "c2")]
    output = await agent(model(turn, "Done."), tools.all, requirements=[requirement], middlewares=[guard]).run("go")
    assert [request.tool.name for request in asked] == ["issue_refund"]
    assert sorted(tools.names) == ["issue_refund", "send_email"]  # parallel calls finish in any order
    assert results(output)["c2"].startswith(BLOCKED)


async def test_permission_requirement_rejection_is_not_asked_again() -> None:
    tools = Tools()
    requirement_asked: list[ReviewRequest] = []
    middleware_asked: list[ReviewRequest] = []
    guard = ACGSMiddleware(constitution=review_rules(), approver=recording(middleware_asked, True))
    requirement = guard.permission_requirement(approver=recording(requirement_asked, False))
    result = await _run(guard, tools, requirements=[requirement])
    assert tools.ran == []
    assert (len(requirement_asked), middleware_asked) == (1, [])
    assert result["r1"] == f"{NOT_RUN}{REFUND_REVIEW} The reviewer did not approve this call, so it was denied."


async def test_permission_requirement_approval_is_not_asked_again() -> None:
    tools = Tools()
    asked: list[ReviewRequest] = []
    guard = ACGSMiddleware(constitution=review_rules(), approver=recording(asked, True))
    await _run(guard, tools, requirements=[guard.permission_requirement()])
    assert len(asked) == 1
    assert tools.names == ["issue_refund"]


async def test_permission_requirement_defaults_to_io_confirm() -> None:
    answers: list[str] = []

    async def confirm(prompt: str, **kwargs: Any) -> bool:
        answers.append(prompt)
        return False

    async def read(prompt: str) -> str:
        return ""

    reset = setup_io_context(read=read, confirm=confirm)
    try:
        tools = Tools()
        guard = ACGSMiddleware(constitution=review_rules())
        await _run(guard, tools, requirements=[guard.permission_requirement()])
    finally:
        reset()
    assert tools.ran == []
    assert len(answers) == 1


async def test_permission_requirement_include_and_exclude() -> None:
    tools = Tools()
    guard = ACGSMiddleware(constitution=review_rules())
    excluded = guard.permission_requirement(exclude=[tools.issue_refund], approver=lambda r: True)
    result = await _run(guard, tools, requirements=[excluded])
    assert tools.ran == []
    assert result["r1"].endswith("No reviewer is configured, so it was denied.")
    included = guard.permission_requirement(include=[tools.issue_refund], approver=lambda r: True)
    await _run(guard, tools, requirements=[included])
    assert tools.names == ["issue_refund"]


async def test_permission_requirement_without_the_middleware_still_governs_its_tools() -> None:
    tools = Tools()
    guard = ACGSMiddleware(constitution=review_rules())
    requirement = guard.permission_requirement(approver=lambda r: True)
    turn = [call("issue_refund", REFUND, "r1"), call("send_email", LEAK, "c1")]
    output = await agent(model(turn, "Done."), tools.all, requirements=[requirement]).run("go")
    assert tools.names == ["issue_refund"]
    assert results(output)["c1"] == "This tool is not allowed to be used."


async def test_remembered_review_must_match_the_call(monkeypatch: pytest.MonkeyPatch) -> None:
    tools = Tools()
    guard = ACGSMiddleware(constitution=review_rules())
    event_input: dict[str, Any] = {"input": dict(REFUND)}
    approved = await guard.decide(tools.issue_refund, REFUND, approver=lambda r: True)
    guard._remember_review(event_input, REFUND, approved)
    changed = await guard._decide_event(tools.issue_refund, {**REFUND, "amount": 999.0}, "run", event_input)
    assert changed.verdict is Verdict.DENY
    guard._remember_review(event_input, REFUND, approved)
    other = await guard._decide_event(tools.issue_refund, REFUND, "run", {"input": dict(REFUND)})
    assert other.verdict is Verdict.DENY
    guard._remember_review(event_input, REFUND, approved)
    same = await guard._decide_event(tools.issue_refund, REFUND, "run", event_input)
    assert same is approved


def test_remembered_reviews_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "_MAX_REMEMBERED_REVIEWS", 2)
    guard = ACGSMiddleware()
    inputs = [{"input": {"n": n}} for n in range(3)]
    decision = guard.evaluate("x", {})
    for event_input in inputs:
        guard._remember_review(event_input, {}, decision)
    assert [entry[0] for entry in guard._reviewed.values()] == inputs[1:]
