import re
import json
import uuid
from typing import Optional, List
from contextlib import AsyncExitStack
# from pprint import pprint

from mcp import ClientSession
from mcp.client.sse import sse_client
from openai.types.chat.chat_completion_message_tool_call import (
    Function,
    ChatCompletionMessageToolCall,
)

class MCPClient:
    def __init__(self):
        # Initialise session and client objects
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

    def process_tool_calls(self, text) -> Optional[List[ChatCompletionMessageToolCall]]:
        """Extract tool code from model response and return an OpenAI compatible class"""
        pattern = r"```tool_code\n([\s\S]*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            json_obj = json.loads(json_str)
            return [
                ChatCompletionMessageToolCall(
                    id=str(uuid.uuid4()).split("-")[-1],
                    type="function",
                    function=Function(
                        name=json_obj["name"],
                        arguments=str(json_obj["parameters"]).replace("'", '"'),
                    ),
                )
            ]
        return None

    def format_tool_output(self, text) -> str:
        return f"```tool_output\n{text}\n```"
