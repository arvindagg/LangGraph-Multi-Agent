# agents/code_review_agent.py

from state import AgentState
from llms.ollama_llms import llm_agent_with_tools # Using the same LLM for consistency
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json

def code_review_agent(state: AgentState) -> AgentState:
    print("---Executing Code Review Agent---")
    messages = state['messages']
    generated_code = state.get('generated_code', '')
    code_review_count = state.get('code_review_count', 0)
    original_user_query = next((m.content for m in messages if isinstance(m, HumanMessage)), "User's request.")


    if not generated_code:
        # Fallback if somehow no code was generated
        state['messages'].append(AIMessage(content="Code Review Agent: No code found to review. Ending code generation cycle."))
        state['next'] = "supervisor" # Go back to supervisor for a fresh start or END
        return state

    review_prompt = f"""
    You are an expert Python code reviewer.
    Your task is to review the provided Python code based on the user's original request.
    Assess the code for:
    1.  **Correctness**: Does it fulfill the user's request accurately?
    2.  **Readability**: Is it clear, well-structured, and easy to understand?
    3.  **Efficiency**: Is there a more optimal way to achieve the goal?
    4.  **Best Practices**: Does it follow common Python best practices (e.g., PEP 8)?

    After your review, provide your assessment in a JSON object with two keys:
    -   `feedback`: (string) Your detailed feedback on the code. If the code is acceptable, state "Code is acceptable.".
    -   `is_acceptable`: (boolean) True if the code is acceptable and requires no further changes, False otherwise.

    DO NOT include any conversational text outside the JSON object.
    STRICTLY adhere to the JSON format.

    User's Original Request: "{original_user_query}"

    Code to Review:
    ```python
    {generated_code}
    ```

    Your Review (JSON format only):
    """

    # Invoke the LLM for code review
    review_response = llm_agent_with_tools.invoke([HumanMessage(content=review_prompt)])
    raw_review_content = review_response.content.strip()

    print(f"---Code Review Agent raw response: '{raw_review_content}'---")

    feedback = "Error: Could not parse review."
    is_acceptable = False

    try:
        review_data = json.loads(raw_review_content)
        feedback = review_data.get('feedback', 'No feedback provided.')
        is_acceptable = review_data.get('is_acceptable', False)
    except json.JSONDecodeError:
        print("---Warning: Code Review LLM did not return valid JSON. Attempting fallback for acceptance.---")
        # Fallback for ill-formatted JSON: simple keyword search
        if "code is acceptable" in raw_review_content.lower() or "no further changes" in raw_review_content.lower():
            is_acceptable = True
            feedback = raw_review_content # Use raw content as feedback
        else:
            feedback = "Received malformed review response. Assuming un-acceptable."
            is_acceptable = False # Default to false if malformed and no clear acceptance

    state['messages'].append(AIMessage(content=f"Code Review Agent has completed review."))
    state['messages'].append(AIMessage(content=f"Review Feedback: {feedback}"))
    state['code_review_feedback'] = feedback

    MAX_REVIEW_CYCLES = 5
    if is_acceptable:
        print("---Code is acceptable. Returning to Supervisor.---")
        state['messages'].append(AIMessage(content="The generated code has been reviewed and is acceptable."))
        state['next'] = "supervisor" # Code is good, back to supervisor to decide next step (likely END)
    elif code_review_count >= MAX_REVIEW_CYCLES:
        print(f"---Max review cycles ({MAX_REVIEW_CYCLES}) reached. Code not acceptable. Returning to Supervisor.---")
        state['messages'].append(AIMessage(content=f"Max code review cycles ({MAX_REVIEW_CYCLES}) reached. Code still has issues. User intervention may be required."))
        state['next'] = "supervisor" # Max cycles reached, return to supervisor
    else:
        print("---Code is NOT acceptable. Looping back to Code Generation.---")
        state['messages'].append(AIMessage(content="The generated code is not yet acceptable. Re-generating based on feedback."))
        state['next'] = "code_generation_agent" # Loop back to generation

    return state