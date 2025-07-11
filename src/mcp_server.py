import os
import logging
import json
import time

import httpx
import uvicorn 
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from mcp.types import TextContent
from starlette.applications import Starlette
from starlette.routing import Mount, Route
from starlette.requests import Request


# MCP Basic setup 
mcp = FastMCP(name="mcp-scheduler", dependencies=["httpx"])

# Tools

# Get current time
@mcp.tool(
    name="get_current_time",
    description="Get the current time",
)
async def get_current_time() -> str:
    """
    Get the current time
    """
    return time.strftime("%Y-%m-%d %H:%M:%S")


# Starlette/ SSE Server Setup
def create_starlette_app(mcp_server, *, debug: bool = False) -> Starlette:
    """Create a Starlette application that can server the provied mcp server with SSE."""
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> None:
        async with sse.connect_sse(
            request.scope,
            request.receive,
            request._send,  # noqa: SLF001
        ) as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )

    return Starlette(
        debug=debug,
        routes=[
            Route("/sse/", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )


def start_mcp_server():
    mcp_server = mcp._mcp_server  # noqa: WPS437

    # Bind SSE request handling to MCP server
    starlette_app = create_starlette_app(mcp_server, debug=False)

    # Run the Starlette app
    uvicorn.run(starlette_app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    start_mcp_server()




