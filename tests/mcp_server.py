# SPDX-FileCopyrightText: 2026-present ACGS Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""A stdio MCP server with one tool. It appends each call to the file named by ``argv[1]``."""

from __future__ import annotations

import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

_LOG = Path(sys.argv[1]) if len(sys.argv) > 1 else None

app = FastMCP("acgs-test")


@app.tool()
def send_email(to: str, body: str) -> str:
    """Send an email."""
    if _LOG is not None:
        with _LOG.open("a", encoding="utf-8") as log:
            log.write(f"send_email {to}\n")
    return f"sent to {to}"


if __name__ == "__main__":
    app.run()
