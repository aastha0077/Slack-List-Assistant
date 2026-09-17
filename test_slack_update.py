import os, json
from dotenv import load_dotenv
from slack_sdk import WebClient

load_dotenv()
client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))
list_id = os.getenv("SLACK_LIST_ID")

res = client.api_call("slackLists.items.list", json={"list_id": list_id, "limit": 1})
item_id = res["items"][0]["id"]
print("item_id", item_id)

try:
    res = client.api_call("slackLists.items.update", json={"list_id": list_id, "cells": [{"item_id": item_id, "column_id": "fake", "text": "test"}]})
    print(res)
except Exception as e:
    print(e.response["error"] if hasattr(e, "response") else e)
