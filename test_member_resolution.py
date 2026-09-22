"""Focused tests for conservative Slack member identity resolution."""
from copy import deepcopy

import pytest

import slack_tools


class MemberClient:
    def __init__(self, members):
        self.members = members

    def users_list(self, **kwargs):
        return {"ok": True, "members": deepcopy(self.members)}


@pytest.fixture
def members(monkeypatch):
    client = MemberClient([
        {
            "id": "U0C2F3CFQ00", "name": "aasthaa", "real_name": "Aastha Acharya",
            "profile": {"display_name": "AasthaA", "real_name": "Aastha Acharya"},
        },
        {
            "id": "UPRAVEEN", "name": "praveen", "real_name": "Praveen",
            "profile": {"display_name": "Praveen", "real_name": "Praveen"},
        },
        {
            "id": "UMARY", "name": "mary.jane", "real_name": "Mary Jane",
            "profile": {"display_name": "Mary Jane", "real_name": "Mary Jane"},
        },
    ])
    monkeypatch.setattr(slack_tools, "_client", client)
    return client


@pytest.mark.parametrize("spoken", ["Aastha", "AUSTA", "AasthaA"])
def test_aastha_variants_resolve_to_aasthaa(members, spoken):
    assert slack_tools.find_user_candidates(spoken) == [
        {"id": "U0C2F3CFQ00", "label": "AasthaA"}]


def test_praveen_exact_name_is_preserved(members):
    assert slack_tools.find_user_id("Praveen") == "UPRAVEEN"


def test_exact_real_name_is_supported(members):
    assert slack_tools.find_user_id("Aastha Acharya") == "U0C2F3CFQ00"


def test_exact_slack_username_is_supported(members):
    assert slack_tools.find_user_id("mary.jane") == "UMARY"


def test_member_matching_is_case_insensitive(members):
    assert slack_tools.find_user_id("aAsThAa") == "U0C2F3CFQ00"


def test_member_matching_normalizes_whitespace(members):
    assert slack_tools.find_user_id("  Mary   Jane  ") == "UMARY"


def test_exact_slack_mention_bypasses_name_matching(members):
    assert slack_tools.find_user_candidates("<@U0C2F3CFQ00>") == [
        {"id": "U0C2F3CFQ00", "label": "<@U0C2F3CFQ00>"}]


def test_unknown_member_never_resolves(members):
    assert slack_tools.find_user_candidates("OSTHO") == []
    assert slack_tools.find_user_id("OSTHO") is None


def test_ambiguous_spoken_variant_returns_every_candidate(monkeypatch):
    client = MemberClient([
        {"id": "UA", "name": "aasthaa", "profile": {"display_name": "AasthaA"}},
        {"id": "UB", "name": "aasthab", "profile": {"display_name": "AasthaB"}},
    ])
    monkeypatch.setattr(slack_tools, "_client", client)
    assert slack_tools.find_user_candidates("Aastha") == [
        {"id": "UA", "label": "AasthaA"},
        {"id": "UB", "label": "AasthaB"},
    ]
    assert slack_tools.find_user_id("Aastha") is None


def test_ambiguous_phonetic_variant_never_selects_arbitrarily(monkeypatch):
    client = MemberClient([
        {"id": "UA", "name": "aasthaa", "profile": {"display_name": "AasthaA"}},
        {"id": "UB", "name": "austhaa", "profile": {"display_name": "AusthaA"}},
    ])
    monkeypatch.setattr(slack_tools, "_client", client)
    matches = slack_tools.find_user_candidates("AUSTA")
    assert {match["id"] for match in matches} == {"UA", "UB"}
    assert slack_tools.find_user_id("AUSTA") is None
