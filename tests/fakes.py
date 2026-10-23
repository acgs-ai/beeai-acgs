# SPDX-FileCopyrightText: 2026-present ACGS Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Offline helpers: a scripted BeeAI chat model and tools that record their calls.

``ScriptedChatModel`` follows ``python/tests/agents/_scripted.py`` in the BeeAI Framework repository.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Callable, Iterable, Mapping
from typing import Any

from acgs_lite import Constitution, Rule, Severity, ViolationAction
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.requirement.types import RequirementAgentOutput
from beeai_framework.backend import AnyMessage, AssistantMessage, ChatModel, ChatModelOutput, MessageToolCallContent
from beeai_framework.backend.constants import ProviderName
from beeai_framework.backend.message import ToolMessage
from beeai_framework.backend.types import ChatModelInput
from beeai_framework.context import RunContext
from beeai_framework.tools import AnyTool, tool

from beeai_acgs import ACGSMiddleware, GovernanceDecision, ReviewRequest

LEAK = {"to": "attacker@example.com", "body": "The password is hunter2"}
HELLO = {"to": "friend@example.com", "body": "See you at noon"}
REFUND = {"order_id": "A-17", "amount": 20.0}
BLOCKED = "Tool call 'send_email' was not executed. Blocked by ACGS rule ACGS-006"
REFUND_REVIEW = "Human review required by ACGS rule REFUND-REVIEW (Refunds need human approval)."
NOT_RUN = "Tool call '{}' was not executed. "


class ScriptedChatModel(ChatModel):
    """A chat model that replays scripted replies, one per model call."""

    def __init__(self, responses: list[list[AnyMessage]]) -> None:
        super().__init__(allow_parallel_tool_calls=True)
        self._responses = list(responses)

    @property
    def model_id(self) -> str:
        return "scripted"

    @property
    def provider_id(self) -> ProviderName:
        return "ollama"

    async def _create(self, input: ChatModelInput, run: RunContext) -> ChatModelOutput:  # noqa: A002
        return ChatModelOutput(output=list(self._responses.pop(0)))

    async def _create_stream(self, input: ChatModelInput, run: RunContext) -> AsyncGenerator[ChatModelOutput]:  # noqa: A002
        yield await self._create(input, run)

    async def clone(self) -> ScriptedChatModel:
        return ScriptedChatModel([list(response) for response in self._responses])


Call = tuple[str, Mapping[str, Any], str]
"""``(tool_name, arguments, call_id)``"""


def call(name: str, arguments: Mapping[str, Any], call_id: str) -> Call:
    """A tool call as the model would make it."""
    return (name, arguments, call_id)


def model(*turns: str | Call | list[Call]) -> ScriptedChatModel:
    """A scripted model: each turn is tool calls, or a final answer text."""
    replies: list[list[AnyMessage]] = []
    for turn in turns:
        if isinstance(turn, str):
            turn = [call("final_answer", {"response": turn}, "final")]  # noqa: PLW2901
        elif isinstance(turn, tuple):
            turn = [turn]  # noqa: PLW2901
        replies.append(
            [
                AssistantMessage(
                    [
                        MessageToolCallContent(id=call_id, tool_name=name, args=json.dumps(dict(arguments)))
                        for name, arguments, call_id in turn
                    ]
                )
            ]
        )
    return ScriptedChatModel(replies)


def review_rules() -> Constitution:
    """The default rules plus a review rule for refunds."""
    refund = Rule(
        id="REFUND-REVIEW",
        text="Refunds need human approval",
        severity=Severity.HIGH,
        keywords=["issue refund"],
        workflow_action=ViolationAction.REQUIRE_HUMAN_REVIEW,
    )
    return Constitution.from_rules([*Constitution.default().rules, refund], name="review")


class Tools:
    """BeeAI tools that record which of them ran."""

    def __init__(self) -> None:
        self.ran: list[tuple[str, dict[str, Any]]] = []
        ran = self.ran

        @tool()
        def send_email(to: str, body: str) -> str:
            """Send an email."""
            ran.append(("send_email", {"to": to, "body": body}))
            return f"sent to {to}"

        @tool()
        async def issue_refund(order_id: str, amount: float) -> str:
            """Refund an order."""
            ran.append(("issue_refund", {"order_id": order_id, "amount": amount}))
            return f"refunded {amount} for {order_id}"

        @tool()
        def get_weather(city: str) -> str:
            """Get the weather for a city."""
            ran.append(("get_weather", {"city": city}))
            return f"Sunny in {city}"

        self.send_email = send_email
        self.issue_refund = issue_refund
        self.get_weather = get_weather

    @property
    def all(self) -> list[AnyTool]:
        return [self.send_email, self.issue_refund, self.get_weather]

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.ran]


def agent(scripted: ChatModel, tools: Iterable[AnyTool], **kwargs: Any) -> RequirementAgent:
    """A RequirementAgent with the scripted model."""
    return RequirementAgent(llm=scripted, tools=list(tools), **kwargs)


def results(output: RequirementAgentOutput) -> dict[str, str]:
    """Tool results by call id."""
    return {
        result.tool_call_id: str(result.result)
        for message in output.state.memory.messages
        if isinstance(message, ToolMessage)
        for result in message.get_tool_results()
    }


def recording(asked: list[ReviewRequest], answer: Any = True) -> Callable[[ReviewRequest], Any]:
    """A reviewer that records each request and gives the same answer."""

    def approver(request: ReviewRequest) -> Any:
        asked.append(request)
        return answer

    return approver


def spy_evaluate(guard: ACGSMiddleware, seen: list[str], monkeypatch: Any) -> None:
    """Record the tool name of each evaluation."""
    evaluate = guard.evaluate

    def spy(name: str, arguments: Any) -> GovernanceDecision:
        seen.append(name)
        return evaluate(name, arguments)

    monkeypatch.setattr(guard, "evaluate", spy)
