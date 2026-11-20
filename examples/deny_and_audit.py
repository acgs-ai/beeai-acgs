"""Run one allowed and one denied BeeAI tool call offline, then print a decision trail."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncGenerator
from dataclasses import asdict
from typing import Any

# Disable telemetry before framework imports; the scripted model uses no provider or API key.
os.environ["OTEL_SDK_DISABLED"] = "true"

from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.backend import AssistantMessage, ChatModel, ChatModelOutput, MessageToolCallContent
from beeai_framework.backend.constants import ProviderName
from beeai_framework.backend.types import ChatModelInput
from beeai_framework.context import RunContext
from beeai_framework.tools import tool

from beeai_acgs import ACGSMiddleware, GovernanceDecision


class AuditedMiddleware(ACGSMiddleware):
    """Collect allow/deny decisions through the public API without retaining arguments."""

    def __init__(self) -> None:
        super().__init__()
        self.decisions: list[GovernanceDecision] = []

    def evaluate(self, tool_name: str, arguments: Any) -> GovernanceDecision:
        decision = super().evaluate(tool_name, arguments)
        self.decisions.append(decision)
        return decision


class ScriptedChatModel(ChatModel):
    """Return prewritten tool calls without making network requests."""

    def __init__(self) -> None:
        super().__init__()
        self.turns = [
            ("note-1", "record_note", {"body": "The meeting starts at noon."}),
            ("note-2", "record_note", {"body": "The admin password is demo-secret."}),
            ("answer", "final_answer", {"response": "Done."}),
        ]

    @property
    def model_id(self) -> str:
        return "scripted"

    @property
    def provider_id(self) -> ProviderName:
        return "ollama"

    async def _create(self, input: ChatModelInput, run: RunContext) -> ChatModelOutput:  # noqa: A002
        call_id, name, arguments = self.turns.pop(0)
        message = AssistantMessage([MessageToolCallContent(id=call_id, tool_name=name, args=json.dumps(arguments))])
        return ChatModelOutput(output=[message])

    async def _create_stream(self, input: ChatModelInput, run: RunContext) -> AsyncGenerator[ChatModelOutput]:  # noqa: A002
        yield await self._create(input, run)


@tool()
def record_note(body: str) -> str:
    """Record a demo note by printing it locally."""
    print(f"Executed record_note: {body}")
    return "Note recorded."


async def main() -> None:
    middleware = AuditedMiddleware()
    agent = RequirementAgent(llm=ScriptedChatModel(), tools=[record_note], middlewares=[middleware])
    await agent.run("Record the two notes.")
    print("Audit trail (application decisions):")
    for decision in middleware.decisions:
        print(json.dumps(asdict(decision), sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
