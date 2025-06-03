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
from llms.ollama_llms import llm_agent_with_tools, llm_supervisor, llm_code_generator # Ensure llm_code_generator is imported
from agents.calculator_agent import calculator_agent
from agents.data_analysis_agent import data_analysis_agent
from agents.stock_news_agent import stock_news_agent
from agents.text_processing_agent import text_processing_agent
from agents.web_search_agent import web_search_agent
from agents.code_generation_agent import code_generation_agent
from agents.code_review_agent import code_review_agent
from supervisor.supervisor_node import supervisor_node
from state import AgentState

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
workflow = StateGraph(AgentState)

workflow.add_node("text_processing_agent", text_processing_agent)
workflow.add_node("data_analysis_agent", data_analysis_agent)
workflow.add_node("calculator_agent", calculator_agent)
workflow.add_node("stock_news_agent", stock_news_agent)
workflow.add_node("web_search_agent", web_search_agent)
workflow.add_node("code_generation_agent", code_generation_agent)
workflow.add_node("code_review_agent", code_review_agent)
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
        "code_generation_agent": "code_generation_agent",
        "END": END,
    },
)

workflow.add_edge("text_processing_agent", "supervisor")
workflow.add_edge("data_analysis_agent", "supervisor")
workflow.add_edge("calculator_agent", "supervisor")
workflow.add_edge("stock_news_agent", "supervisor")
workflow.add_edge("web_search_agent", "supervisor")

# Define the iterative loop for code generation and review
workflow.add_edge("code_generation_agent", "code_review_agent")

workflow.add_conditional_edges(
    "code_review_agent",
    lambda state: state['next'],
    {
        "code_generation_agent": "code_generation_agent",
        "supervisor": "supervisor",
    }
)

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

        # Initialize state for new session or load for existing
        initial_state = {
            "messages": [input_message],
            "code_review_count": 0,
            "generated_code": "",
            "code_review_feedback": ""
        }
        
        try:
            async for s in app_runnable.astream_events(
                initial_state,
                config={"configurable": {"thread_id": session_id}},
                version="v1"
            ):
                event_type = s["event"]
                
                print(f"\n--- LangGraph Event Raw Data ---")
                print(f"Event Type: {event_type}")
                print(f"Event Name: {s.get('name')}")
                print(f"Event Data: {s.get('data')}")
                print(f"--- End LangGraph Event Raw Data ---\n")

                if event_type == "on_chain_start":
                    if s["name"] == "LangGraph":
                        yield f"data: {{\"type\": \"start_stream\", \"data\": {{}}}}\n\n"
                    elif s["name"] == "supervisor":
                         yield f"data: {{\"type\": \"status_update\", \"data\": {{\"status\": \"Supervisor is routing...\", \"sender\": \"Supervisor\"}}}}\n\n"
                    elif s["name"] in ["web_search_agent", "calculator_agent", "stock_news_agent", "text_processing_agent", "data_analysis_agent", "code_generation_agent", "code_review_agent"]:
                        yield f"data: {{\"type\": \"status_update\", \"data\": {{\"status\": \"Executing {s['name'].replace('_', ' ').title()}...\", \"sender\": \"Agent\"}}}}\n\n"

                elif event_type == "on_tool_start":
                    tool_name = s["name"]
                    tool_args = s["data"].get("input", {})
                    escaped_tool_args = json.dumps(tool_args).replace('"', '\\"')
                    yield f"data: {{\"type\": \"status_update\", \"data\": {{\"status\": \"Calling tool: {tool_name} with args {escaped_tool_args}...\", \"sender\": \"Agent\"}}}}\n\n"

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
                    yield f"data: {{\"type\": \"status_update\", \"data\": {{\"status\": \"Tool '{tool_name}' finished. Produced {tool_output_type}.\", \"sender\": \"Agent\"}}}}\n\n"

                elif event_type == "on_llm_stream":
                    token = s["data"].get("chunk").content
                    if token:
                        current_agent_message_content += token
                        escaped_token = token.replace('"', '\\"').replace('\n', '\\n')
                        yield f"data: {{\"type\": \"token\", \"data\": {{\"token\": \"{escaped_token}\"}}}}\n\n"

                elif event_type == "on_end":
                    pass # We will handle final output explicitly after the stream

            # --- Manual Final Message Handling AFTER the streaming loop ---
            final_state_snapshot = await app_runnable.aget_state(
                config={"configurable": {"thread_id": session_id}}
            )

            final_state = final_state_snapshot.values
            
            # --- NEW DEBUGGING PRINT ---
            print("\n--- SERVER: Debugging Final State ---")
            print(f"Final State (raw): {final_state}")
            print(f"Messages in final state: {final_state.get('messages', [])}")
            print(f"Generated Code in final state: '{final_state.get('generated_code', '')[:500]}' (truncated)") # Print first 500 chars
            print(f"Code Review Count in final state: {final_state.get('code_review_count')}")
            print(f"Code Review Feedback in final state: '{final_state.get('code_review_feedback', '')[:200]}' (truncated)")
            print("--- END SERVER: Debugging Final State ---\n")
            # --- END NEW DEBUGGING PRINT ---

            final_user_facing_message_content = None
            final_generated_code_output = final_state.get('generated_code', '').strip()

            if final_generated_code_output:
                escaped_code = final_generated_code_output.replace('"', '\\"').replace('\n', '\\n')
                yield f"data: {{\"type\": \"code_output\", \"data\": {{\"code\": \"{escaped_code}\", \"sender\": \"Code Generator\"}}}}\n\n"
                print(f"Server manually yielded final code output: {final_generated_code_output[:100]}...")
            
            # Find the last user-facing AIMessage content, excluding internal routing messages
            if 'messages' in final_state:
                for msg in reversed(final_state['messages']):
                    if isinstance(msg, AIMessage) and msg.content and msg.content.strip():
                        if not msg.content.startswith("Supervisor routing to:") and \
                           not "has generated code" in msg.content and \
                           not "Review Feedback:" in msg.content and \
                           not "The generated code is not yet acceptable." in msg.content and \
                           not "The generated code has been reviewed and is acceptable." in msg.content and \
                           not "Max code review cycles" in msg.content:
                            final_user_facing_message_content = msg.content
                            break
                    elif isinstance(msg, ToolMessage) and msg.content and msg.content.strip():
                        final_user_facing_message_content = f"Tool Result: {msg.content}"
                        break
            
            if final_user_facing_message_content:
                escaped_message = final_user_facing_message_content.replace('"', '\\"').replace('\n', '\\n')
                yield f"data: {{\"type\": \"final_message\", \"data\": {{\"message\": \"{escaped_message}\"}}}}\n\n"
                print(f"Server manually yielded final message: {final_user_facing_message_content[:100]}...")
            elif not final_generated_code_output:
                yield f"data: {{\"type\": \"final_message\", \"data\": {{\"message\": \"Processing completed with no specific final message.\"}}}}\n\n"
                print("Server manually yielded fallback final message: Processing completed with no specific final message.")

            yield f"data: {{\"type\": \"end_stream\", \"data\": {{}}}}\n\n"
            print("Server yielding end_stream after final message.")

        except Exception as e:
            print(f"Error during LangGraph streaming or final message retrieval: {e}")
            import traceback
            traceback.print_exc()
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