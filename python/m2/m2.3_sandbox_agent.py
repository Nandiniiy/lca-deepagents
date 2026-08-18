from uuid import uuid4
from dotenv import load_dotenv

from deepagents import create_deep_agent
from deepagents.backends.langsmith import LangSmithSandbox
from langsmith.sandbox import SandboxClient

from models import model


# Load values from .env file
load_dotenv()


# Create LangSmith sandbox client
client = SandboxClient()

# Keep sandbox variable outside try so finally can access it safely
ls_sandbox = None

try:
    # Create a new LangSmith sandbox
    ls_sandbox = client.create_sandbox(
        name=f"lca-deepagents-lab-{uuid4().hex[:8]}"
    )

    print(f"Sandbox: {ls_sandbox.name}  (id: {ls_sandbox.id})")

    # Use LangSmith sandbox as backend
    backend = LangSmithSandbox(sandbox=ls_sandbox)

    # Create deep agent
    agent = create_deep_agent(
        model=model,
        backend=backend,
        system_prompt=(
            "You are a coding assistant. When asked to run code, write the script "
            "to a file first, then execute it. Show the output in your final answer."
        ),
    )

    # Invoke the agent
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Write a Python script that prints the first 15 Fibonacci numbers, "
                        "save it to fib.py, and run it."
                    ),
                }
            ]
        }
    )

    # Get the final message
    final_message = result["messages"][-1]

    # Print output safely for both OpenAI-style and Gemini-style responses
    if hasattr(final_message, "content") and final_message.content:
        print(final_message.content)
    elif hasattr(final_message, "text") and final_message.text:
        print(final_message.text)
    else:
        print(final_message)

finally:
    # Delete sandbox only if it was created successfully
    if ls_sandbox is not None:
        client.delete_sandbox(ls_sandbox.name)
        print(f"Deleted sandbox: {ls_sandbox.name}")