import pytest
from intent_parser import parse_intent
import slack_tools

class TestNaturalCompletionParsing:
    @pytest.mark.parametrize("phrase,expected_task", [
        ("I finished the login bug.", "login bug"),
        ("I finished login bug.", "login bug"),
        ("Login bug is finished.", "login bug"),
        ("I completed the login bug.", "login bug"),
        ("I have completed the login bug.", "login bug"),
        ("I did the login bug.", "login bug"),
        ("I have done the login bug.", "login bug"),
        ("I'm done with the login bug.", "login bug"),
        ("Login bug is done.", "login bug"),
        ("Mark login bug as completed.", "login bug"),
        ("Mark the login bug done.", "login bug"),
        ("I wrapped up the login bug.", "login bug"),
        ("The login bug is complete.", "login bug"),
        ("I've taken care of the login bug.", "login bug"),
        ("Done with the login bug.", "login bug"),
        ("Finished with login bug.", "login bug"),
        ("I already finished that task.", "__LAST__"),
        ("I already completed that.", "__LAST__"),
    ])
    def test_single_task_completion_phrases(self, phrase, expected_task):
        res = parse_intent(phrase)
        assert res.get("intent") == "complete", f"Failed intent for: {phrase}, got: {res}"
        task_name = (res.get("task_name") or "").lower()
        assert task_name == expected_task.lower(), f"Failed task_name for: {phrase}, got: {res.get('task_name')!r}"

    def test_multi_task_completion(self):
        text = "I finished the login bug and completed the dashboard deployment."
        res = parse_intent(text)
        assert res.get("intent") == "complete", f"Expected complete intent, got: {res}"
        tasks = res.get("tasks")
        assert isinstance(tasks, list), f"Expected tasks list, got: {tasks}"
        assert len(tasks) >= 2, f"Expected 2+ tasks, got: {len(tasks)}"
        names = [t.get("task_name", "").lower() for t in tasks]
        assert any("login bug" in n for n in names)
        assert any("dashboard" in n for n in names)


class TestSemanticMatching:
    def setup_method(self):
        self.mock_schema = {
            "schema": [
                {"id": "col_name", "key": "name", "name": "Task Name", "type": "text"},
                {"id": "col_assignee", "key": "assignee", "name": "Assignee", "type": "user"},
                {"id": "col_completed", "key": "completed", "name": "Completed", "type": "checkbox"},
            ]
        }

    def _make_item(self, id_val, name, completed=False, user_id=None):
        return {
            "id": id_val,
            "fields": [
                {"column_id": "col_name", "text": name},
                {"column_id": "col_completed", "checkbox": completed},
                {"column_id": "col_assignee", "user": [{"id": user_id}]} if user_id else {},
            ]
        }

    def test_exact_match(self):
        items = [self._make_item("1", "Login Bug")]
        matches = slack_tools.find_matches(items, "login bug", self.mock_schema)
        assert len(matches) == 1
        assert matches[0]["id"] == "1"

    def test_query_without_verb_matches_title_with_verb(self):
        items = [self._make_item("1", "Fix the login bug")]
        matches = slack_tools.find_matches(items, "login bug", self.mock_schema)
        assert len(matches) == 1
        assert matches[0]["id"] == "1"

    def test_query_with_verb_matches_title_without_verb(self):
        items = [self._make_item("1", "login bug")]
        matches = slack_tools.find_matches(items, "Fix the login bug", self.mock_schema)
        assert len(matches) == 1
        assert matches[0]["id"] == "1"

    def test_continuous_prefix_matching(self):
        items = [self._make_item("1", "Insight dashboard Completion for pitch")]
        matches = slack_tools.find_matches(items, "Insight dashboard", self.mock_schema)
        assert len(matches) == 1
        assert matches[0]["id"] == "1"

    def test_ambiguity_returns_multiple(self):
        items = [
            self._make_item("1", "Client report draft"),
            self._make_item("2", "Client report review"),
        ]
        matches = slack_tools.find_matches(items, "Client report", self.mock_schema)
        assert len(matches) == 2

    def test_false_positive_prevention(self):
        items = [self._make_item("1", "onboarding docs")]
        matches = slack_tools.find_matches(items, "docs check", self.mock_schema)
        assert len(matches) == 0


class TestListQueryScopes:
    @pytest.mark.parametrize("query", [
        "what are my tasks",
        "what do I need to work on?",
        "what is left for me?",
        "what should I do today?",
        "list my tasks",
        "show tasks",
    ])
    def test_pending_default_queries(self, query):
        res = parse_intent(query)
        assert res.get("intent") == "list"
        assert res.get("completed") is False or res.get("status") == "open" or "completed" not in res

    @pytest.mark.parametrize("query", [
        "what have I completed?",
        "what did I finish?",
        "show my completed tasks",
        "list completed items",
    ])
    def test_completed_queries(self, query):
        res = parse_intent(query)
        assert res.get("intent") == "list"
        assert res.get("completed") is True or res.get("status") == "completed"

    @pytest.mark.parametrize("query", [
        "show all my tasks",
        "show everything",
        "show all action items",
        "list all tasks",
    ])
    def test_all_tasks_queries(self, query):
        res = parse_intent(query)
        assert res.get("intent") == "list"
        assert res.get("all_tasks") is True or res.get("completed") is None
