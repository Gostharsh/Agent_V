
from uuid import uuid4

from src.vision_agent.graph import run_agent

# Keep one owner across tests, but use a fresh conversation thread for each.
owner_id = f"live-memory-test-{uuid4().hex[:8]}"

# print("Vision Agent live memory test")
# print(f"Test owner: {owner_id}")
# print("Type 'exit' to quit.\n")

while True:
    message = input("You: ").strip()

    if message.lower() in {"exit", "quit"}:
        break

    if not message:
        continue

    try:
        response = run_agent(
            message,
            thread_id=f"live-test-{uuid4().hex}",
            owner_id=owner_id,
        )
        print(f"Vision Agent: {response}\n")
    except Exception as exc:
        print(f"Agent error: {exc}\n")
