# SPDX-FileCopyrightText: 2026-present ACGS Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""ACGS governance for BeeAI Framework runs: a run middleware that checks each tool call before the tool runs."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import sys
import weakref
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from acgs_lite import Constitution
from beeai_framework.agents.requirement.requirements.ask_permission import AskPermissionRequirement
from beeai_framework.agents.requirement.utils._tool import FinalAnswerTool
from beeai_framework.context import RunContext, RunContextStartEvent, RunMiddlewareProtocol
from beeai_framework.emitter import EmitterOptions, EventMeta
from beeai_framework.emitter.utils import create_internal_event_matcher
from beeai_framework.errors import FrameworkError
from beeai_framework.tools import AnyTool, StringToolOutput, Tool
from beeai_framework.tools.events import ToolStartEvent
from beeai_framework.utils.io import io_confirm
from pydantic import BaseModel

from beeai_acgs._governance import GovernanceDecision, ToolCallFormatter, ToolCallGovernor, Verdict

logger = logging.getLogger(__name__)

LAST = -sys.maxsize
"""Listener priority of the middleware. BeeAI runs higher priorities first, so the checks run after other listeners."""

_EVALUATION_FAILED = "ACGS governance could not evaluate this tool call, so it was denied."
_REVIEW_DISABLED = "Review is turned off (on_review='deny'), so it was denied."
_NO_REVIEWER = "No reviewer is configured, so it was denied."
_NO_ANSWER = "No approval was received ({}), so it was denied."
_NOT_APPROVED = "The reviewer did not approve this call, so it was denied."
_APPROVED = "A reviewer approved this call."
_MAX_REMEMBERED_REVIEWS = 256


@dataclass(frozen=True)
class ReviewRequest:
    """A tool call that matched only review rules and needs a person's approval before it runs."""

    tool: AnyTool
    """The tool the agent wants to call."""
    arguments: dict[str, Any]
    """The validated tool input, as JSON-compatible values."""
    decision: GovernanceDecision
    """The ACGS decision; ``decision.reason`` and ``decision.rule_ids`` name the matched review rules."""
    run_id: str | None = None
    """The run ID of the tool call, if known."""
    prompt: str = field(init=False)
    """A one-line question for the reviewer that names the tool and the rules (not the argument values)."""

    def __post_init__(self) -> None:
        """Build the prompt."""
        object.__setattr__(
            self,
            "prompt",
            f"ACGS: the agent wants to call the tool `{self.tool.name}`. {self.decision.reason} Approve this call?",
        )


Approver = Callable[[ReviewRequest], "bool | Awaitable[bool]"]
"""A reviewer: returns ``True`` (exactly) to run the call. Anything else, an error or a timeout denies it."""


async def io_confirm_approver(request: ReviewRequest) -> bool:
    """Ask in the terminal (or in the handler set with ``beeai_framework.utils.io.setup_io_context``).

    This is the reviewer BeeAI's own ``AskPermissionRequirement`` uses by default. In the terminal, an answer that
    starts with ``yes`` approves the call.

    Args:
        request: The call to review.

    Returns:
        Whether the reviewer approved the call.
    """
    return await io_confirm(request.prompt, data=request.arguments)


class ACGSToolCallChangedError(FrameworkError):
    """Raised when a tool starts with input other than the input ACGS checked, or after ACGS denied it.

    Another listener changed the call after the check. The tool does not run, and BeeAI ends the run with this
    error.
    """

    def __init__(self, message: str) -> None:
        """Create the error.

        Args:
            message: What changed.
        """
        super().__init__(message, is_fatal=True, is_retryable=False)


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    return value


def _tool_input(tool: AnyTool, raw: Any) -> dict[str, Any]:
    """The input the tool will run with: the raw input validated by the tool's schema, defaults included."""
    try:
        return dict(tool.input_schema.model_validate(raw).model_dump(mode="json"))
    except Exception:
        # The tool rejects this input itself and does not run; evaluate what the model sent.
        value = _jsonable(raw)
        return value if isinstance(value, dict) else {"input": value}


def _fingerprint(arguments: Any) -> str:
    return json.dumps(arguments, sort_keys=True, default=repr)


def _with(decision: GovernanceDecision, verdict: Verdict, reason: str) -> GovernanceDecision:
    return GovernanceDecision(
        verdict=verdict,
        tool_name=decision.tool_name,
        reason=f"{decision.reason} {reason}",
        rule_ids=decision.rule_ids,
        warning_rule_ids=decision.warning_rule_ids,
        error=decision.error,
    )


class ACGSMiddleware(RunMiddlewareProtocol):
    """Check BeeAI tool calls against an ACGS constitution before the tool runs.

    Pass it where BeeAI accepts run middleware::

        agent = RequirementAgent(llm=llm, tools=[...], middlewares=[ACGSMiddleware()])
        # or for one run: await agent.run("...").middleware(ACGSMiddleware())

    It sees every tool call made inside that run, including calls of sub-agents and tools that call other tools.
    For each call it:

    - runs allowed calls unchanged;
    - does not run blocked calls: ``"Tool call '<name>' was not executed. <reason>"`` becomes the tool output,
      which the model reads;
    - puts calls that need review to ``approver``; they run only if it returns ``True``.

    A blocking rule always wins over review rules, and any error while evaluating or reviewing a call denies it.
    Right before the tool runs it checks again that the input is the input it checked; if another listener
    changed it, or started a denied call, the run ends with ``ACGSToolCallChangedError``.

    Example:
        >>> from beeai_acgs import ACGSMiddleware
        >>> middleware = ACGSMiddleware()
        >>> middleware.evaluate("send_email", {"to": "a@example.com", "body": "The password is hunter2"}).verdict
        <Verdict.DENY: 'deny'>
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        constitution: Constitution | None = None,
        agent_id: str = "beeai",
        on_review: Literal["ask", "deny"] = "ask",
        approver: Approver | None = None,
        review_timeout: float | None = None,
        tool_call_formatter: ToolCallFormatter | None = None,
    ) -> None:
        """Create the middleware.

        Args:
            constitution: The ACGS constitution to enforce. Defaults to ``acgs_lite.Constitution.default()``.
            agent_id: Agent identifier recorded by ACGS for each validation.
            on_review: ``"ask"`` puts calls that need review to ``approver``; ``"deny"`` denies them.
            approver: The reviewer, sync or async. Without one, calls that need review are denied.
                ``io_confirm_approver`` asks in the terminal.
            review_timeout: Seconds to wait for an async approver; ``None`` waits as long as it takes.
            tool_call_formatter: Optional function ``(tool_name, args) -> str`` that replaces ``format_tool_call``.

        Raises:
            ValueError: If ``on_review`` or ``review_timeout`` is invalid.
        """
        if on_review not in ("ask", "deny"):
            msg = f"on_review must be 'ask' or 'deny', got {on_review!r}"
            raise ValueError(msg)
        if review_timeout is not None and review_timeout <= 0:
            msg = "review_timeout must be greater than zero"
            raise ValueError(msg)
        self._governor = ToolCallGovernor(constitution, agent_id=agent_id, tool_call_formatter=tool_call_formatter)
        self.constitution: Constitution = self._governor.constitution
        self.agent_id = agent_id
        self.on_review = on_review
        self.approver = approver
        self.review_timeout = review_timeout
        self._bound: weakref.WeakSet[RunContext] = weakref.WeakSet()
        self._decided: weakref.WeakSet[RunContext] = weakref.WeakSet()
        self._reviewed: OrderedDict[int, tuple[dict[str, Any], str, GovernanceDecision]] = OrderedDict()

    def __repr__(self) -> str:
        """Show the options, not the constitution."""
        return (
            f"ACGSMiddleware(agent_id={self.agent_id!r}, on_review={self.on_review!r}, "
            f"approver={self.approver!r}, review_timeout={self.review_timeout!r})"
        )

    def evaluate(self, tool_name: str, arguments: Any) -> GovernanceDecision:
        """Evaluate a tool call without running anything or asking anyone. Never raises.

        Args:
            tool_name: Name of the tool.
            arguments: The tool arguments.

        Returns:
            The governance decision.
        """
        return self._governor.evaluate(tool_name, arguments)

    async def decide(
        self, tool: AnyTool, arguments: dict[str, Any], *, run_id: str | None = None, approver: Approver | None = None
    ) -> GovernanceDecision:
        """Decide one call the way the middleware does, including review. Never raises.

        Args:
            tool: The tool.
            arguments: The tool input.
            run_id: The run ID of the tool call, passed on to the reviewer.
            approver: Use this reviewer instead of ``self.approver``.

        Returns:
            The decision; ``ALLOW`` means the tool runs.
        """
        return await self._resolve(self._evaluate(tool, arguments), tool, arguments, run_id, approver)

    def _evaluate(self, tool: AnyTool, arguments: dict[str, Any]) -> GovernanceDecision:
        try:
            return self.evaluate(tool.name, arguments)
        except Exception as error:
            return self._failed(tool, error)

    @staticmethod
    def _failed(tool: AnyTool, error: Exception) -> GovernanceDecision:
        logger.warning("ACGS could not decide tool call '%s'; denying it (fail-closed).", tool.name, exc_info=error)
        return GovernanceDecision(
            verdict=Verdict.DENY,
            tool_name=tool.name,
            reason=_EVALUATION_FAILED,
            error=f"{type(error).__name__}: {error}",
        )

    async def _resolve(
        self,
        decision: GovernanceDecision,
        tool: AnyTool,
        arguments: dict[str, Any],
        run_id: str | None,
        approver: Approver | None,
    ) -> GovernanceDecision:
        """Apply review handling to an evaluated call. Never raises."""
        try:
            if decision.verdict is not Verdict.REVIEW:
                return decision
            if self.on_review == "deny":
                return _with(decision, Verdict.DENY, _REVIEW_DISABLED)
            reviewer = approver or self.approver
            if reviewer is None:
                return _with(decision, Verdict.DENY, _NO_REVIEWER)
            return await self._review(reviewer, ReviewRequest(tool, arguments, decision, run_id))
        except Exception as error:
            return self._failed(tool, error)

    async def _review(self, reviewer: Approver, request: ReviewRequest) -> GovernanceDecision:
        decision = request.decision
        try:
            answer = reviewer(request)
            if inspect.isawaitable(answer):
                answer = await asyncio.wait_for(answer, timeout=self.review_timeout)
        except TimeoutError:
            logger.warning("ACGS review of tool call '%s' timed out; denying it.", decision.tool_name)
            return _with(decision, Verdict.DENY, _NO_ANSWER.format(f"no answer within {self.review_timeout:g}s"))
        except Exception:
            logger.warning("ACGS review of tool call '%s' failed; denying it.", decision.tool_name, exc_info=True)
            return _with(decision, Verdict.DENY, _NO_ANSWER.format("the reviewer failed"))
        if answer is True:
            return _with(decision, Verdict.ALLOW, _APPROVED)
        return _with(decision, Verdict.DENY, _NOT_APPROVED)

    def bind(self, ctx: RunContext) -> None:
        """Attach the middleware to a run. BeeAI calls this; pass the middleware in ``middlewares=[...]``.

        Args:
            ctx: The run context.
        """
        if ctx in self._bound:
            return
        self._bound.add(ctx)

        async def on_start(data: Any, meta: EventMeta) -> None:
            creator = meta.creator
            if not isinstance(data, RunContextStartEvent) or not isinstance(creator, RunContext):
                return
            # BeeAI forwards the events of a run only to its direct parent, so follow every nested run
            # (sub-agents, tools that call tools) to see the tool calls made inside it.
            self.bind(creator)
            if isinstance(creator.instance, Tool):
                await self._check(creator, creator.instance, data)

        async def on_finish(_: Any, __: EventMeta) -> None:
            ctx.emitter.off(callback=on_start)
            ctx.emitter.off(callback=on_finish)

        ctx.emitter.on(
            create_internal_event_matcher("start"),
            on_start,
            EmitterOptions(is_blocking=True, match_nested=True, priority=LAST),
        )
        ctx.emitter.on(create_internal_event_matcher("finish"), on_finish, EmitterOptions(match_nested=False))

    async def _check(self, tool_ctx: RunContext, tool: AnyTool, data: RunContextStartEvent) -> None:
        if isinstance(tool, FinalAnswerTool) or tool_ctx in self._decided:
            return
        self._decided.add(tool_ctx)
        arguments = _tool_input(tool, data.input.get("input"))
        decision = await self._decide_event(tool, arguments, tool_ctx.run_id, data.input)
        allowed = decision.verdict is Verdict.ALLOW
        if not allowed:
            data.output = StringToolOutput(f"Tool call '{tool.name}' was not executed. {decision.reason}")
        checked = _fingerprint(arguments) if allowed else None

        def before_tool_runs(event: ToolStartEvent, _: EventMeta) -> None:
            if checked is None:
                msg = f"ACGS denied the call of tool '{tool.name}', but the tool started anyway."
                raise ACGSToolCallChangedError(msg)
            if _fingerprint(_jsonable(event.input)) != checked:
                msg = f"The input of tool '{tool.name}' changed after ACGS checked it."
                raise ACGSToolCallChangedError(msg)

        # The tool's own emitter, lowest priority: the last listener before the tool runs.
        tool_ctx.emitter.on("start", before_tool_runs, EmitterOptions(is_blocking=True, priority=LAST))

    async def _decide_event(
        self, tool: AnyTool, arguments: dict[str, Any], run_id: str, event_input: dict[str, Any]
    ) -> GovernanceDecision:
        remembered = self._reviewed.pop(id(event_input), None)
        if remembered is not None and remembered[0] is event_input and remembered[1] == _fingerprint(arguments):
            # permission_requirement already put this exact call to the reviewer; do not ask twice.
            return remembered[2]
        return await self.decide(tool, arguments, run_id=run_id)

    def permission_requirement(
        self,
        include: Any = None,
        *,
        exclude: Any = None,
        approver: Approver | None = None,
    ) -> AskPermissionRequirement:
        """An ``AskPermissionRequirement`` that asks only about calls ACGS sends to review.

        Use it with the middleware on a ``RequirementAgent``, so the agent's permission flow is the reviewer::

            middleware = ACGSMiddleware(constitution=constitution)
            agent = RequirementAgent(
                llm=llm,
                tools=[...],
                requirements=[middleware.permission_requirement()],
                middlewares=[middleware],
            )

        Allowed calls are not asked about, blocked calls are refused without asking, and a call the reviewer
        approved is not asked about again by the middleware.

        Args:
            include: Tools to cover, as for ``AskPermissionRequirement``; all tools if not set.
            exclude: Tools to leave out, as for ``AskPermissionRequirement``.
            approver: The reviewer. Defaults to the middleware's ``approver``, then ``io_confirm_approver``.

        Returns:
            The requirement.
        """

        async def handler(tool: AnyTool, event_input: dict[str, Any]) -> bool:
            arguments = _tool_input(tool, event_input.get("input"))
            evaluated = self._evaluate(tool, arguments)
            reviewer = approver or self.approver or io_confirm_approver
            decision = await self._resolve(evaluated, tool, arguments, None, reviewer)
            if evaluated.verdict is Verdict.REVIEW:
                self._remember_review(event_input, arguments, decision)
            return decision.verdict is Verdict.ALLOW

        return AskPermissionRequirement(include, exclude=exclude, handler=handler)

    def _remember_review(
        self, event_input: dict[str, Any], arguments: dict[str, Any], decision: GovernanceDecision
    ) -> None:
        self._reviewed[id(event_input)] = (event_input, _fingerprint(arguments), decision)
        while len(self._reviewed) > _MAX_REMEMBERED_REVIEWS:
            self._reviewed.popitem(last=False)
