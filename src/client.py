from typing import Optional
from contextlib import AsyncExitStack
# from pprint import pprint

from mcp import ClientSession
from mcp.client.sse import sse_client


class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_sse_server(self, server_url: str):
        """Connect to an MCP server running with SSE transport"""
        # Ensure the server_url ends with a trailing slash for the messages endpoint
        if not server_url.endswith("/"):
            server_url += "/"

        # Store the context managers so they stay alive
        self._streams_context = sse_client(url=server_url)
        streams = await self._streams_context.__aenter__()

        self._session_context = ClientSession(*streams)
        self.session: ClientSession = await self._session_context.__aenter__()

        # Initialize
        await self.session.initialize()

        # List available tools to verify connection
        print("Initialized SSE client...")
        print("Listing tools...")
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def cleanup(self):
        """Clean up SSE client connection and session"""
        if self.session:
            await self._session_context.__aexit__(None, None, None)
            await self._streams_context.__aexit__(None, None, None)
            self.session = None
            print("\nClosed SSE connection and cleaned up session.")

    async def available_tools(self) -> str:
        """Return list of available tools in OpenAI format"""
        mcp_response = await self.session.list_tools()
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
            for tool in mcp_response.tools
        ]
