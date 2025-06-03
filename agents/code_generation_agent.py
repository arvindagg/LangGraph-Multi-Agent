# agents/code_generation_agent.py

from state import AgentState
from llms.ollama_llms import llm_code_generator # CHANGED: Use the dedicated code generator LLM
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import re # IMPORTANT: Added import for the 're' module

def code_generation_agent(state: AgentState) -> AgentState:
    """
    An agent responsible for generating new code or revising existing code
    based on user input and code review feedback.
    """
    print("---Executing Code Generation Agent---")
    messages = state['messages']
    generated_code = state.get('generated_code', '')
    code_review_feedback = state.get('code_review_feedback', '')
    code_review_count = state.get('code_review_count', 0)

    # Find the original user query to provide context for code generation
    # It's better to get the *initial* HumanMessage that triggered code generation
    initial_human_message = next((m.content for m in messages if isinstance(m, HumanMessage)), "Generate code based on the user's request.")

    # Determine the context for the LLM based on whether it's a first pass or a revision
    if generated_code and code_review_feedback:
        # This is a revision pass: provide original code and feedback
        system_prompt = f"""
        You are an expert software engineer.
        Your task is to revise the provided Python code based on the given review feedback.
        Ensure the revised code directly addresses all feedback points.
        Produce ONLY the complete, revised Python code block.
        Your response MUST start and end with triple backticks (```) with 'python' language tag.
        DO NOT include any conversational text, explanations, or JSON outside the code block.
        If you cannot generate valid code, respond with a clear message indicating so, but try your best.

        User's Original Request: "{initial_human_message}"

        Original Code to Revise:
        ```python
        {generated_code}
        ```

        Review Feedback:
        {code_review_feedback}
        """
        llm_input_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Please revise the code as per the feedback provided.")
        ]
    else:
        # This is the initial code generation pass
        system_prompt = """
        You are an expert software engineer.
        Your task is to write clean, efficient, and well-commented Python code based on the user's request.
        Produce ONLY the complete Python code block.
        Your response MUST start and end with triple backticks (```) with 'python' language tag.
        DO NOT include any conversational text, explanations, or JSON outside the code block.
        If you cannot generate valid code, respond with a clear message indicating so, but try your best.

        User's Request: "{initial_human_message}"
        """
        llm_input_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=initial_human_message) # Pass the original request as the user message
        ]

    # Invoke the LLM to generate or revise code
    response = llm_code_generator.invoke(llm_input_messages) # Using llm_code_generator
    generated_code_content = response.content.strip()

    # --- DEBUGGING STEP ---
    print(f"---Code Generation Agent: Raw LLM Response Content---\n{generated_code_content}\n-----------------------------------------------------")
    # --- END DEBUGGING STEP ---

    # Extract code block using regex
    code_match = re.search(r"```(?:\w+)?\n(.*?)```", generated_code_content, re.DOTALL)
    extracted_code = code_match.group(1).strip() if code_match else generated_code_content

    # Robust check for empty or "task" JSON output
    if not extracted_code or extracted_code.lower().strip() == '{ "task": "write clean, efficient, and well-commented python code based on the user\'s request." }'.lower().strip():
        print("---Code Generation Agent: Warning! No valid code extracted or LLM returned task description.---")
        # If no code is extracted or it's the default task description, set a placeholder and send to supervisor
        state['messages'].append(AIMessage(content="Code Generation Agent: Failed to generate or extract valid code. LLM might be struggling with the request or format. Please try a different request or be more specific."))
        state['next'] = "supervisor" # Directly go to supervisor as no valid code was generated
        return state

    # Update state with the new code and increment review count
    state['generated_code'] = extracted_code
    state['messages'].append(AIMessage(content=f"Code Generation Agent has generated code (Attempt {code_review_count + 1}). Sending for review..."))
    state['messages'].append(AIMessage(content=f"Generated Code:\n```python\n{extracted_code}\n```"))
    state['code_review_feedback'] = "" # Clear previous feedback
    state['code_review_count'] = code_review_count + 1 # Increment counter

    # Route to the code review agent for the next step in the workflow
    state['next'] = "code_review_agent"
    return state

