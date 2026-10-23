# SPDX-FileCopyrightText: 2026-present ACGS Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tools from an MCP server are governed like any other BeeAI tool."""

from __future__ import annotations

import sys
from pathlib import Path

from beeai_framework.tools.mcp import MCPTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from beeai_acgs import ACGSMiddleware
from tests.fakes import BLOCKED, HELLO, LEAK, agent, call, model, results

SERVER = Path(__file__).with_name("mcp_server.py")


async def test_mcp_tool_calls_are_governed(tmp_path: Path) -> None:
    log = tmp_path / "calls.log"
    log.touch()
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER), str(log)])
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = await MCPTool.from_session(session)
        turn = [call("send_email", LEAK, "c1"), call("send_email", HELLO, "c2")]
        output = await agent(model(turn, "Done."), tools, middlewares=[ACGSMiddleware()]).run("go")
    assert log.read_text(encoding="utf-8") == "send_email friend@example.com\n"
    assert results(output)["c1"].startswith(BLOCKED)
    assert "sent to friend@example.com" in results(output)["c2"]
