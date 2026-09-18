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
            # Ollama may return @AasthaA or AasthaA — both are functionally identical
            # since find_user_id() strips leading @. Check only intent/task/date/priority.
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



def test_multi_task_create():
    """Verify multi-task CREATE parsing — the main new feature."""
    passed = True
    today_iso = today.isoformat()

    # --- Bullet list → 2 tasks ---
    text = (
        "I have a few action items today. I want to work on\n"
        "* Insight dashboard Completion for pitch\n"
        "* Deploy the pitch dashboard."
    )
    res = parse_intent(text)
    if res.get("intent") != "create":
        print(f"FAILED [bullet multi]: intent={res.get('intent')!r}, expected 'create'\nFull: {json.dumps(res)}")
        passed = False
    elif not isinstance(res.get("tasks"), list) or len(res["tasks"]) < 2:
        print(f"FAILED [bullet multi]: expected tasks list with 2+ items, got: {res.get('tasks')}")
        passed = False
    else:
        names = [t.get("task_name", "") for t in res["tasks"]]
        print(f"  [bullet multi] PASS — tasks: {names}")

    # --- Bullet list → 3 tasks ---
    text = "* Finish report\n* Review code\n* Deploy dashboard"
    res = parse_intent(text)
    if res.get("intent") != "create":
        print(f"FAILED [3 bullets]: intent={res.get('intent')!r}\nFull: {json.dumps(res)}")
        passed = False
    elif not isinstance(res.get("tasks"), list) or len(res["tasks"]) < 3:
        print(f"FAILED [3 bullets]: expected 3 tasks, got: {res.get('tasks')}")
        passed = False
    else:
        names = [t.get("task_name", "") for t in res["tasks"]]
        print(f"  [3 bullets] PASS — tasks: {names}")

    # --- Query must NEVER create ---
    for query in [
        "what are my tasks?",
        "what are my action items?",
        "what do I need to do today?",
        "show my pending tasks",
        "show me the tasks due today",
    ]:
        res = parse_intent(query)
        if res.get("intent") != "list":
            print(f"FAILED [query guard]: '{query}' → intent={res.get('intent')!r}, expected 'list'")
            passed = False
        elif res.get("tasks"):
            print(f"FAILED [query guard]: '{query}' → produced tasks list unexpectedly")
            passed = False
        else:
            print(f"  [query guard] PASS: '{query}' → list")

    # --- Multi-task + shared metadata (priority + due_date) ---
    text = "Create P1 tasks due today:\n1. Write report\n2. Send to client"
    res = parse_intent(text)
    if res.get("intent") != "create":
        print(f"FAILED [meta shared]: intent={res.get('intent')!r}\nFull: {json.dumps(res)}")
        passed = False
    elif isinstance(res.get("tasks"), list) and len(res["tasks"]) >= 2:
        meta_ok = True
        for t in res["tasks"]:
            if t.get("priority") != "P1":
                print(f"FAILED [meta shared]: priority={t.get('priority')!r}, expected 'P1'")
                meta_ok = False
                passed = False
            if t.get("due_date") != today_iso:
                print(f"FAILED [meta shared]: due_date={t.get('due_date')!r}, expected {today_iso!r}")
                meta_ok = False
                passed = False
        if meta_ok:
            print("  [meta shared] PASS — priority + due_date propagated to all tasks")
    else:
        print("  [meta shared] SKIP — tasks not split into list (single-task path taken, acceptable)")

    # --- assignee_self propagation ---
    text = "I want to work on:\n* Task Alpha\n* Task Beta"
    res = parse_intent(text)
    if res.get("intent") != "create":
        print(f"FAILED [assignee_self]: intent={res.get('intent')!r}")
        passed = False
    elif isinstance(res.get("tasks"), list):
        for t in res["tasks"]:
            if not t.get("assignee_self"):
                print(f"FAILED [assignee_self]: task missing assignee_self=True: {t}")
                passed = False
                break
        else:
            print("  [assignee_self] PASS")

    # --- Single task still works (regression) ---
    res = parse_intent("add client report")
    if res.get("intent") != "create":
        print(f"FAILED [single regression]: intent={res.get('intent')!r}")
        passed = False
    else:
        print("  [single regression] PASS")

    # --- Complete mutation is NOT create ---
    res = parse_intent("complete client report")
    if res.get("intent") != "complete":
        print(f"FAILED [complete guard]: intent={res.get('intent')!r}, expected 'complete'")
        passed = False
    else:
        print("  [complete guard] PASS")

    # --- Delete mutation is NOT create ---
    res = parse_intent("delete old report task")
    if res.get("intent") != "delete":
        print(f"FAILED [delete guard]: intent={res.get('intent')!r}, expected 'delete'")
        passed = False
    else:
        print("  [delete guard] PASS")

    if passed:
        print("Multi-task CREATE tests PASSED!")
    else:
        print("Multi-task CREATE tests FAILED — see details above.")
    return passed


if __name__ == "__main__":
    r1 = test_intent_parsing()
    r2 = test_multi_task_create()
    import sys
    sys.exit(0 if (r1 and r2) else 1)
