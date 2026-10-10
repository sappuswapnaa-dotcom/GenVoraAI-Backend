# GenVora AI Backend — Gemini Cloud

This backend keeps the Android app's existing API contract:

- POST `/v1/chat`
- GET `/health`
- POST `/v1/report`
- GET `/privacy`
- GET `/terms`

## Render environment variable

Create this secret in Render:

`GEMINI_API_KEY`

Do not put the Gemini API key inside the Android app or commit it to GitHub.

Default model:

`gemini-3.8-flash`

## Render start command

`uvicorn app.main:app --host 0.0.0.0 --port $PORT`
