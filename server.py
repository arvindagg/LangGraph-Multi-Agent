# server.py

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from dotenv import load_dotenv
from typing import AsyncGenerator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
import json

# Load environment variables from .env file
load_dotenv()

# --- Import your LangGraph components ---
# Assuming these imports are correct and available in your environment
from llms.ollama_llms import llm_agent_with_tools, llm_supervisor
from agents.calculator_agent import calculator_agent
from agents.data_analysis_agent import data_analysis_agent
from agents.stock_news_agent import stock_news_agent
from agents.text_processing_agent import text_processing_agent
from agents.web_search_agent import web_search_agent
from supervisor.supervisor_node import supervisor_node
from state import AgentState # Assuming AgentState is defined correctly

import sys

print("--- Python Sys Path for Uvicorn Reloader Process ---")
for path_entry in sys.path:
    print(path_entry)
print("--- End Python Sys Path ---")

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# --- Initialize FastAPI App ---
app = FastAPI(
    title="LangGraph Multi-Agent Framework API",
    description="API for the LangGraph-based multi-agent conversational framework with streaming capabilities.",
    version="1.0.0",
)

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- LangGraph Workflow Setup ---
# Ensure your agents and supervisor are callable as defined in your imports
workflow = StateGraph(AgentState)

workflow.add_node("text_processing_agent", text_processing_agent)
workflow.add_node("data_analysis_agent", data_analysis_agent)
workflow.add_node("calculator_agent", calculator_agent)
workflow.add_node("stock_news_agent", stock_news_agent)
workflow.add_node("web_search_agent", web_search_agent)
workflow.add_node("supervisor", supervisor_node)

workflow.set_entry_point("supervisor")

workflow.add_conditional_edges(
    "supervisor",
    lambda state: state['next'],
    {
        "text_processing_agent": "text_processing_agent",
        "data_analysis_agent": "data_analysis_agent",
        "calculator_agent": "calculator_agent",
        "stock_news_agent": "stock_news_agent",
        "web_search_agent": "web_search_agent",
        "END": END,
    },
)

workflow.add_edge("text_processing_agent", "supervisor")
workflow.add_edge("data_analysis_agent", "supervisor")
workflow.add_edge("calculator_agent", "supervisor")
workflow.add_edge("stock_news_agent", "supervisor")
workflow.add_edge("web_search_agent", "supervisor")

app_runnable = workflow.compile(checkpointer=MemorySaver())

# --- FastAPI Endpoint ---

@app.post("/chat")
async def chat_endpoint(request: Request) -> StreamingResponse:
    data = await request.json()
    user_message_content = data.get("message")
    session_id = data.get("session_id", "default_session")

    if not user_message_content:
        return {"error": "Message content is required."}, 400

    async def event_generator() -> AsyncGenerator[str, None]:
        input_message = HumanMessage(content=user_message_content)
        current_agent_message_content = "" # To accumulate tokens if they appear

        try:
            # First, stream the initial events
            async for s in app_runnable.astream_events(
                {"messages": [input_message]},
                config={"configurable": {"thread_id": session_id}},
                version="v1"
            ):
                event_type = s["event"]

                # --- DEBUGGING PRINT STATEMENT (Keep for now) ---
                print(f"\n--- LangGraph Event Raw Data ---")
                print(f"Event Type: {event_type}")
                print(f"Event Name: {s.get('name')}")
                print(f"Event Data: {s.get('data')}") # Print directly, as it might contain non-JSON serializable objects
                print(f"--- End LangGraph Event Raw Data ---\n")
                # --- END DEBUGGING PRINT STATEMENT ---

                if event_type == "on_chain_start":
                    if s["name"] == "LangGraph":
                        yield f"data: {{\"type\": \"start_stream\", \"data\": {{}}}}\n\n"
                    elif s["name"] == "supervisor":
                         yield f"data: {{\"type\": \"status_update\", \"data\": {{\"status\": \"Supervisor is routing...\"}}}}\n\n"
                    elif s["name"] in ["web_search_agent", "calculator_agent", "stock_news_agent", "text_processing_agent", "data_analysis_agent"]:
                        yield f"data: {{\"type\": \"status_update\", \"data\": {{\"status\": \"Executing {s['name'].replace('_', ' ').title()}...\"}}}}\n\n"

                elif event_type == "on_tool_start":
                    tool_name = s["name"]
                    tool_args = s["data"].get("input", {})
                    escaped_tool_args = json.dumps(tool_args).replace('"', '\\"')
                    yield f"data: {{\"type\": \"status_update\", \"data\": {{\"status\": \"Calling tool: {tool_name} with args {escaped_tool_args}...\"}}}}\n\n"

                elif event_type == "on_tool_end":
                    tool_name = s["name"]
                    tool_output_type = "output"
                    raw_output = s["data"].get("output", "")
                    if isinstance(raw_output, list):
                        tool_output_type = f"list of {len(raw_output)} items"
                    elif isinstance(raw_output, dict):
                        tool_output_type = f"dictionary"
                    elif isinstance(raw_output, str) and len(raw_output) > 50:
                         tool_output_type = f"text output ({len(raw_output)} chars)"
                    yield f"data: {{\"type\": \"status_update\", \"data\": {{\"status\": \"Tool '{tool_name}' finished. Produced {tool_output_type}.\"}}}}\n\n"

                elif event_type == "on_llm_stream":
                    token = s["data"].get("chunk").content
                    if token:
                        current_agent_message_content += token # Accumulate tokens
                        escaped_token = token.replace('"', '\\"').replace('\n', '\\n')
                        yield f"data: {{\"type\": \"token\", \"data\": {{\"token\": \"{escaped_token}\"}}}}\n\n"

                # Removed the on_chain_end logic for final_message from here,
                # as it will be handled explicitly after the loop completes.
                elif event_type == "on_end":
                    # This event signifies the end of the astream_events iterator
                    # We will send the 'end_stream' after the final message is sent.
                    pass # Don't yield end_stream here yet

            # --- Manual Final Message Handling AFTER the streaming loop ---
            # Retrieve the final state to ensure we get the last message
            final_state = await app_runnable.aget_state(
                config={"configurable": {"thread_id": session_id}}
            )

            # --- NEW DEBUGGING FOR STATE ACCESS ---
            print(f"\n--- Final State Debugging ---")
            print(f"Type of final_state: {type(final_state)}")
            print(f"Content of final_state: {final_state}")
            print(f"--- End Final State Debugging ---\n")
            # --- END NEW DEBUGGING FOR STATE ACCESS ---

            # CORRECTED LINE: Access messages directly from the StateSnapshot object
            # LangGraph's StateSnapshot object behaves like a dictionary for its state keys.
            # If AgentState is a TypedDict, direct access like final_state['messages'] is correct.
            # If AgentState is a Pydantic model, it might be final_state.messages.
            # If it's a tuple, it might be final_state[0]['messages'].
            # Based on the error "tuple indices must be integers or slices, not str",
            # it suggests final_state itself might be a tuple. Let's try accessing the first element.
            if isinstance(final_state, tuple):
                # Assuming the actual state dictionary is the first element of the tuple
                final_messages = final_state[0]['messages']
            else:
                # Fallback for direct dictionary-like access
                final_messages = final_state['messages']


            final_user_facing_message_content = None
            # Iterate through messages in reverse to find the last AIMessage that is not a supervisor routing message
            for msg in reversed(final_messages):
                if isinstance(msg, AIMessage) and msg.content:
                    if not msg.content.startswith("Supervisor routing to:") and \
                       not msg.content.strip().lower() in ["end", "end point"]:
                        final_user_facing_message_content = msg.content
                        break

            if final_user_facing_message_content:
                escaped_message = final_user_facing_message_content.replace('"', '\\"').replace('\n', '\\n')
                yield f"data: {{\"type\": \"final_message\", \"data\": {{\"message\": \"{escaped_message}\"}}}}\n\n"
                print(f"Server manually yielded final message: {final_user_facing_message_content[:100]}...") # Server-side debug
            elif current_agent_message_content:
                # Fallback: if no clear AIMessage but some tokens were accumulated (e.g., direct LLM call)
                escaped_message = current_agent_message_content.replace('"', '\\"').replace('\n', '\\n')
                yield f"data: {{\"type\": \"final_message\", \"data\": {{\"message\": \"{escaped_message}\"}}}}\n\n"
                print(f"Server manually yielded accumulated tokens as final message: {current_agent_message_content[:100]}...")
            else:
                yield f"data: {{\"type\": \"final_message\", \"data\": {{\"message\": \"Processing completed with no specific final message.\"}}}}\n\n"
                print("Server manually yielded fallback final message: Processing completed with no specific final message.")

            # Finally, yield the end_stream event after the final message is sent.
            yield f"data: {{\"type\": \"end_stream\", \"data\": {{}}}}\n\n"
            print("Server yielding end_stream after final message.")


        except Exception as e:
            # Catch any unexpected errors during streaming or state retrieval
            print(f"Error during LangGraph streaming or final message retrieval: {e}")
            error_msg_escaped = str(e).replace('"', '\\"').replace('\n', '\\n')
            yield f"data: {{\"type\": \"error\", \"data\": {{\"message\": \"Server error during processing: {error_msg_escaped}\"}}}}\n\n"
            yield f"data: {{\"type\": \"end_stream\", \"data\": {{}}}}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# --- Run the FastAPI App (for development) ---
if __name__ == "__main__":
    print("Starting FastAPI server...")
    print(f"Access the API at http://127.0.0.1:8000/docs (for OpenAPI docs)")
    print(f"Chat endpoint: http://127.0.0.1:8000/chat")
    uvicorn.run(app, host="0.0.0.0", port=8000)
