# supervisor/supervisor_node.py

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from state import AgentState
from llms.ollama_llms import llm_supervisor
from supervisor.prompts import SUPERVISOR_PROMPT, format_chat_history
import json
from typing import List, Dict

VALID_NEXT_STATES = [
    "text_processing_agent",
    "data_analysis_agent",
    "calculator_agent",
    "stock_news_agent",
    "web_search_agent",
    "END"
]

def supervisor_node(state: AgentState) -> Dict[str, str]:
    print("---Executing Supervisor Node---")
    messages = state['messages']

    latest_message_content = ""
    # We need to correctly identify the content that represents the *latest output/answer*
    # from the previous agent, or the user's new query.
    if messages:
        last_msg = messages[-1]

        # Prioritize the final *content* from an AI message (which should be the answer)
        if isinstance(last_msg, AIMessage) and last_msg.content:
            latest_message_content = last_msg.content
        # If it's a ToolMessage, that's the output of a tool
        elif isinstance(last_msg, ToolMessage) and last_msg.content:
            latest_message_content = f"Tool Result: {last_msg.content}"
        # If it's a HumanMessage, it's a new user query
        elif isinstance(last_msg, HumanMessage) and last_msg.content:
            latest_message_content = last_msg.content
        elif hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
            # If AI message just contained tool calls but no direct content
            latest_message_content = f"AI called tool(s): {', '.join([tc['name'] for tc in last_msg.tool_calls])}"

    # The chat history for the prompt should include all messages *except* the very last one.
    # The last one is passed as 'latest_message_content'.
    chat_history_for_prompt = messages[:-1] if messages else []

    formatted_history_str = format_chat_history(chat_history_for_prompt)

    prompt_content = SUPERVISOR_PROMPT.format(
        chat_history=formatted_history_str,
        # Ensure 'latest_message' is never empty to prevent LLM confusion
        latest_message=latest_message_content if latest_message_content else "No explicit new message. Assess completion based on history."
    )

    try:
        llm_response = llm_supervisor.invoke([HumanMessage(content=prompt_content)])
        raw_decision_content = llm_response.content.strip()

        print(f"---Supervisor LLM raw response: '{raw_decision_content}'---")

        next_action = ""
        try:
            parsed_response = json.loads(raw_decision_content)
            next_action = parsed_response.get("next_action", "").strip()
        except json.JSONDecodeError:
            print(f"---Warning: Supervisor LLM did not return valid JSON. Attempting fallback parsing. Raw response: '{raw_decision_content}'---")
            
            found_action = None
            for action in VALID_NEXT_STATES:
                if action.lower() in raw_decision_content.lower():
                    found_action = action
                    break
            
            if found_action:
                next_action = found_action
            else:
                cleaned_content = raw_decision_content.lower()
                for action in VALID_NEXT_STATES:
                    if action.lower() in cleaned_content:
                        next_action = action
                        break
                if not next_action:
                     next_action = raw_decision_content.split(' ')[0].strip().replace('"', '').replace('}', '')

        if next_action not in VALID_NEXT_STATES:
            print(f"---Warning: Supervisor LLM returned an invalid decision: '{next_action}'. Forcing to END.---")
            state['messages'].append(AIMessage(content=f"Supervisor could not determine a valid next step based on conversation, or LLM output was malformed ('{next_action}'). Forcing conversation end."))
            next_action = "END"
            
        state['next'] = next_action
        print(f"---Supervisor decided next action: {next_action}---")
        state['messages'].append(AIMessage(content=f"Supervisor routing to: {next_action}"))
        
        return state # LangGraph expects the entire state dictionary to be returned here
        # It seems your LangGraph setup is expecting the full state object, not just {'next': 'agent_name'}

    except Exception as e:
        print(f"---Error in supervisor_node: {e}---")
        state['messages'].append(AIMessage(content=f"Supervisor encountered a critical internal error: {e}. Cannot determine next step. Forcing END."))
        state['next'] = "END" # Ensure 'next' is set even on error
        return state # Return the modified state on error