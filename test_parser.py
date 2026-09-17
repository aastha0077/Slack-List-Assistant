from intent_parser import parse_intent
import json

tests = [
    "What is the weather?",
    "Add an action item to fix the login bug.",
    "Make task 123 P1.",
    "Show Aashu's tasks.",
    "Update task 123 to P2",
    "Set the priority of the login task to P1",
    "Mark task 123 as complete",
    "Remove task 123",
]

for t in tests:
    print(f"Input: {t}")
    print(f"Output: {json.dumps(parse_intent(t), indent=2)}")
    print("-" * 40)
