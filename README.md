# Slack List Assistant

One shared Slack Bolt application supports local Socket Mode and AWS Lambda HTTP delivery. All task behavior, authorization, parsing, verification, analytics, calendar, transcription, simulation, and Control Tower features use the same existing implementation.

## Local development

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
cp .env.example .env
.venv/bin/python src/local.py
```

Run tests with the same interpreter that received the dependencies:

```bash
.venv/bin/python -m pytest -q
```

The Lambda build installs Linux wheels into `build/`; it does not install
Matplotlib or other runtime dependencies into the local `.venv`.

Local mode requires `SLACK_BOT_TOKEN` and `SLACK_APP_TOKEN`.

## AWS Lambda

The Lambda handler is:

```text
handler.lambda_handler
```

Lambda requires `SLACK_BOT_TOKEN` and `SLACK_SIGNING_SECRET` as environment variables. It does not require `SLACK_APP_TOKEN`. Connect the function through a Lambda Function URL or API Gateway and configure that URL as the Slack request URL.

Build and create/update the function with:

```bash
LAMBDA_FUNCTION_NAME=slack-list-assistant \
LAMBDA_ROLE_ARN=arn:aws:iam::ACCOUNT_ID:role/ROLE_NAME \
./deploy.sh
```

The deployment archive excludes `.env`; configure all secrets in Lambda environment variables.

## Structure

- `src/app.py`: shared Bolt app construction and listener registration
- `src/handler.py`: Lambda HTTP transport
- `src/local.py`: local Socket Mode transport
- `src/graph.py`: facade over existing deterministic-first routing and execution
- `src/tools.py`: stable exports for existing task and mutation tools
- `src/slack_client.py`: reusable Slack API boundary
- `src/prompts.py`: existing prompt and schema exports
- `src/config.py`: runtime-aware environment validation

Root business modules remain compatibility-stable during the incremental package migration. This avoids duplicating or rewriting the production task engine.
