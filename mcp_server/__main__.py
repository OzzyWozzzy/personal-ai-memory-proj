"""``python -m mcp_server`` — run the stdio MCP server."""
from __future__ import annotations

import sys

from mcp_server import main

if __name__ == "__main__":
    sys.exit(main())
