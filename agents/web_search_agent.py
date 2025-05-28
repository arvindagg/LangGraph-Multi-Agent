# agents/web_search_agent.py

from state import AgentState
from llms.ollama_llms import llm_agent_with_tools, llm_supervisor # Assuming you have a tool-calling LLM instance
from tools.tools import tavily_search # Import the specific tool
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def web_search_agent(state: AgentState) -> AgentState:
    print("---Executing Web Search Agent---")
    messages = state['messages']

    # The agent's LLM will decide if it needs to use the tool
    # We use llm_agent_with_tools which is configured to call tools
    response = llm_agent_with_tools.invoke(messages)
    messages.append(response)

    tool_calls = response.tool_calls if hasattr(response, 'tool_calls') else []

    if tool_calls:
        # Assuming only one tool call for simplicity, or handle multiple
        for tool_call in tool_calls:
            if tool_call['name'] == "tavily_search":
                try:
                    # Execute the tavily_search tool
                    search_result = tavily_search.invoke(tool_call['args'])
                    messages.append(ToolMessage(content=search_result, tool_call_id=tool_call['id']))

                    # After getting search results, re-invoke the LLM to summarize/answer
                    # This is crucial for Perplexity-like behavior: LLM analyzes search results
                    print("---Web Search Agent: Analyzing search results with LLM---")
                    summary_prompt = ChatPromptTemplate.from_messages([
                        ("system", "You are an expert at analyzing web search results and providing concise, direct answers to user queries. Summarize the provided search results to answer the user's latest question. If the search results don't directly answer, state what you found or if more search is needed."),
                        ("user", "Here are the search results for the last query:\n{search_results}\n\nUser's original question: {original_question}"),
                    ])

                    # Get the last HumanMessage for the original question
                    original_question = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "No original question found.")

                    summarizer_chain = summary_prompt | llm_supervisor | StrOutputParser() # Using supervisor LLM for summarization, could be separate
                    final_answer = summarizer_chain.invoke({
                        "search_results": search_result,
                        "original_question": original_question
                    })
                    messages.append(AIMessage(content=final_answer))

                except Exception as e:
                    messages.append(ToolMessage(content=f"Error during web search: {e}", tool_call_id=tool_call['id']))
            else:
                messages.append(AIMessage(content=f"Web Search Agent received an unexpected tool call: {tool_call['name']}"))
    else:
        # If LLM didn't call a tool, it might have responded directly or asked clarifying question
        print("---Web Search Agent: LLM responded directly (no tool call)---")
        # The response is already added to messages
    
    return {"messages": messages}