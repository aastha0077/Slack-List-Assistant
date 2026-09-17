import json
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from intent_parser import parse_intent

# Disable logging spam for tests
logging.basicConfig(level=logging.CRITICAL)

today = datetime.now(ZoneInfo("Asia/Kathmandu")).date()
tomorrow = today + timedelta(days=1)

def test_intent_parsing():
    tests = [
        # Deterministic / Read logic
        ("list my tasks", {"intent": "list", "assignee_self": True}),
        ("show me the tasks due today", {"intent": "list", "due_today": True}),
        ("what do I have to work on today?", {"intent": "list", "due_today": True, "assignee_self": True}),
        ("show all tasks", {"intent": "list"}),
        ("list everything assigned to me", {"intent": "list", "assignee_self": True}),
        ("show tasks assigned to me", {"intent": "list", "assignee_self": True}),
        
        # LLM / Write logic
        ("assign New Task to @AasthaA due date 2026-09-17 and P1", {
            "intent": "create", 
            "task_name": "New Task", 
            "assignee": "AasthaA", 
            "priority": "P1", 
            "due_date": "2026-09-17"
        }),
        ("create a task called Client Report", {
            "intent": "create",
            "task_name": "Client Report"
        }),
        ("change docs check priority to P2", {
            "intent": "update",
            "task_name": "docs check"
        }),
        ("mark onboarding complete", {
            "intent": "complete",
            "task_name": "onboarding"
        }),
    ]

    for sentence, expected in tests:
        res = parse_intent(sentence)
        for k, v in expected.items():
            if res.get(k) != v:
                print(f"FAILED: '{sentence}'\nExpected {k}={v}, got {res.get(k)}\nFull: {json.dumps(res)}")
                return False
    print("Intent parsing tests PASSED!")
    return True

if __name__ == "__main__":
    test_intent_parsing()
