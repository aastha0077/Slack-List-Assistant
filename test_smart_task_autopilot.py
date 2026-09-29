from datetime import date, timedelta

import action_item_sentinel
import project_intelligence
import smart_task_autopilot as autopilot


TODAY = date(2026, 9, 24)


def payload(risk_type, *, due, owner_ids=("U_OWNER",), task_name="Client report"):
    return {
        "task_id": "I1", "task_ids": ["I1"], "task_name": task_name,
        "risk_type": risk_type, "owner_ids": list(owner_ids), "priority": "P1",
        "due_date": due.isoformat(), "task_state_hash": "state-v1",
    }


def recommendation(risk_type, *, due, owner_ids=("U_OWNER",)):
    return autopilot.prepare_recommendation(
        alert_id="A1", payload=payload(risk_type, due=due, owner_ids=owner_ids),
        requesting_user="U_APPROVER", owner_name="AasthaA" if owner_ids else None,
        today=TODAY, created_at=1.0)


def test_overdue_task_prepares_owner_reminder_without_raw_ids():
    result = recommendation("overdue", due=TODAY - timedelta(days=1))
    assert result.action_type == "send_reminder"
    assert result.executable is True
    assert result.target_user == "U_OWNER"
    assert "Client report" in result.prepared_message
    assert "AasthaA" in result.prepared_message
    assert "due yesterday" in result.prepared_message
    assert "U_OWNER" not in result.prepared_message


def test_due_soon_task_prepares_deadline_reminder():
    result = recommendation("deadline_risk", due=TODAY + timedelta(days=1))
    assert result.action_type == "send_deadline_reminder"
    assert "due tomorrow" in result.prepared_message
    assert "still pending" in result.prepared_message


def test_unassigned_high_priority_task_requires_human_assignment():
    result = recommendation(
        "unassigned_deadline_risk", due=TODAY + timedelta(days=1), owner_ids=())
    assert result.action_type == "assign_owner"
    assert result.executable is False
    assert result.prepared_message is None
    assert result.target_user is None


def test_workload_risk_prepares_aggregate_human_review():
    value = payload("combined_workload_risk", due=TODAY + timedelta(days=2))
    value["task_id"] = "owner:U_OWNER"
    value["task_ids"] = ["I1", "I2"]
    result = autopilot.prepare_recommendation(
        alert_id="A_WORKLOAD", payload=value, requesting_user="U_APPROVER",
        owner_name="Praveen", today=TODAY, created_at=1.0)
    assert result.action_type == "review_workload"
    assert result.executable is False
    assert result.prepared_message is None
    assert result.recommendation == (
        "Praveen — review the 2 P1 tasks and confirm their deadlines.")

    value["task_ids"] = ["I1"]
    singular = autopilot.prepare_recommendation(
        alert_id="A_SINGLE", payload=value, requesting_user="U_APPROVER",
        owner_name="Praveen", today=TODAY, created_at=1.0)
    assert singular.recommendation == (
        "Praveen — review the 1 P1 task and confirm its deadline.")


def test_unknown_risk_does_not_create_unsafe_action():
    result = recommendation("unknown_risk", due=TODAY + timedelta(days=5))
    assert result.executable is False
    assert result.prepared_message is None
    assert result.recommendation == "No safe automated recommendation is available for this risk."


def test_completed_task_never_produces_a_sentinel_recommendation():
    task = project_intelligence.NormalizedTask(
        item_id="I1", item={}, name="Done", owner_ids=("U_OWNER",), priority="P1",
        due_date=TODAY - timedelta(days=1), completed=True, status="Completed",
        created_date=None)
    assert action_item_sentinel.detect_risks([task], TODAY) == []


def test_recommendation_identity_binds_alert_actor_action_and_state():
    first = recommendation("overdue", due=TODAY - timedelta(days=1))
    second = autopilot.prepare_recommendation(
        alert_id="A1", payload={**payload("overdue", due=TODAY - timedelta(days=1)),
                                "task_state_hash": "state-v2"},
        requesting_user="U_APPROVER", owner_name="AasthaA", today=TODAY, created_at=1.0)
    assert first.recommendation_id != second.recommendation_id
    assert first.task_state_version == "state-v1"
