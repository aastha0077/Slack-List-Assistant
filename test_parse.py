from intent_parser import parse_intent
import json

print("TEST 1:", json.dumps(parse_intent("assign New Task to @AasthaA due date 2026-09-17 and P1"), indent=2))
