import os, json, asyncio
import threading
from typing import AsyncGenerator
from fastapi import FastAPI, Form, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv
from contextlib import asynccontextmanager

from client import MCPClient
from mcp_server import start_mcp_server

load_dotenv()

# Global clients - initialize as None first
llm_client: AsyncOpenAI | None = None
mcp_client: MCPClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global llm_client, mcp_client

    # Run MCP server in a separate thread
    mcp_thread = threading.Thread(target=start_mcp_server, daemon=True)
    mcp_thread.start()
    await asyncio.sleep(5)  # Give the server a moment to start

    llm_client = AsyncOpenAI(
        api_key="ollama",
        base_url="http://localhost:11434/v1",
    )
    mcp_client = MCPClient()
    await mcp_client.connect_to_sse_server("http://localhost:8080/sse/")
    print("Clients initialized and MCP connected.")
    try:
        yield
    finally:
        if mcp_client:
            await mcp_client.cleanup()
            print("MCP Client cleaned up.")
        print("Clients cleanup routine finished.")


app = FastAPI(lifespan=lifespan)

# Get the absolute path to the templates directory
templates_dir = os.path.join(os.path.dirname(__file__), "templates")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def stream_chat(request_data: dict):
    if not llm_client or not mcp_client or not mcp_client.session:
        yield f"data: {json.dumps({'type': 'error', 'message': 'Clients not initialized'})}\n\n"
        return

    messages = request_data["messages"]
    
    # First LLM call to get the tool call
    try:
        tools = await mcp_client.available_tools()
        if not tools:
            tools = []
            print("Warning: No tools available from MCPClient.")
    except Exception as e:
        print(f"Error fetching tools: {e}")
        tools = []

    # First call to the model
    stream = await llm_client.chat.completions.create(
        model=request_data["model"],
        messages=messages,
        tools=tools,
        stream=True,
    )

    # Stream the thought process and collect the tool call
    assistant_response = {"role": "assistant", "content": None, "tool_calls": []}
    async for chunk in stream:
        yield f"data: {chunk.model_dump_json()}\n\n"
        delta = chunk.choices[0].delta
        if delta.content:
            if assistant_response["content"] is None:
                assistant_response["content"] = ""
            assistant_response["content"] += delta.content
        if delta.tool_calls:
            assistant_response["tool_calls"].extend(delta.tool_calls)

    messages.append(assistant_response)

    # If there are tool calls, execute them
    if assistant_response["tool_calls"]:
        for tool_call in assistant_response["tool_calls"]:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            try:
                # Execute the tool call using MCP client
                result = await mcp_client.session.call_tool(function_name, function_args)
                result_text = result.content[0].text

                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": result_text,
                    }
                )
                
                # Stream the tool output to the client
                tool_output_data = {
                    "type": "tool_output",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": result_text
                }
                yield f"data: {json.dumps(tool_output_data)}\n\n"

            except Exception as e:
                print(f"Error executing tool {function_name}: {e}")
                # Optionally stream this error back to the client
    
        # Second call to the model with the tool response
        final_stream = await llm_client.chat.completions.create(
            model=request_data["model"],
            messages=messages,
            stream=True,
        )
        async for chunk in final_stream:
            yield f"data: {chunk.model_dump_json()}\n\n"

    yield f"data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat(
    messages: str = Form(...),
    model: str = Form(...),
    stream: bool = Form(...)
):
    try:
        messages_list = json.loads(messages)
    except json.JSONDecodeError:
        return Response(status_code=400, content="Invalid 'messages' format. Expected a JSON string.")
        
    request_data = {
        "messages": messages_list,
        "model": model,
        "stream": stream
    }
    return StreamingResponse(stream_chat(request_data), media_type="text/event-stream")


@app.get("/")
async def read_index():
    # Route to the testing page directly
    return FileResponse(os.path.join(templates_dir, 'index.html'))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8888)