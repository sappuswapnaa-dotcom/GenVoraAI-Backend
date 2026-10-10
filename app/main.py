import hashlib
import os
import re
import sqlite3
import time
import uuid
from collections import defaultdict, deque
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

APP_NAME = "GenVora AI API"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_BASE_URL = os.getenv(
    "GEMINI_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta",
).rstrip("/")

MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "30000"))
MAX_IMAGE_BASE64 = int(os.getenv("MAX_IMAGE_BASE64", "8000000"))
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
ALLOWED_ORIGINS = [
    x.strip()
    for x in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if x.strip()
]
REPORT_DB = os.getenv("REPORT_DB", "reports.db")

BLOCK_PATTERNS = [
    r"(?:child|minor|underage).{0,30}(?:sexual|nude|porn)",
    r"(?:how to|steps to).{0,20}(?:steal passwords|phish|ransomware)",
    r"(?:encourage|convince).{0,30}(?:suicide|self[- ]harm)",
]


def blocked(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t, re.I | re.S) for p in BLOCK_PATTERNS)


app = FastAPI(title=APP_NAME, version="2.0.0", docs_url="/docs")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_hits: dict[str, deque] = defaultdict(deque)


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_INPUT_CHARS)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=100)
    model: str | None = None
    system_prompt: str = Field(
        default=(
            "You are GenVora AI, a helpful and accurate AI assistant. "
            "Give clear answers, say when you are uncertain, and do not invent sources. "
            "Refuse requests that facilitate exploitation, abuse, fraud, dangerous wrongdoing, "
            "malicious software, or self-harm, and offer safer alternatives when appropriate."
        ),
        max_length=5000,
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    image_base64: str | None = Field(default=None, max_length=MAX_IMAGE_BASE64)


class ChatResponse(BaseModel):
    reply: str
    model: str
    request_id: str


class ReportRequest(BaseModel):
    content: str = Field(min_length=1, max_length=12000)
    reason: str = Field(min_length=1, max_length=200)


def rate_limit(ip: str):
    now = time.time()
    q = _hits[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please try again shortly.",
        )
    q.append(now)


def total_chars(req: ChatRequest) -> int:
    return sum(len(m.content) for m in req.messages) + len(req.system_prompt)


def save_report(content: str, reason: str, ip: str):
    con = sqlite3.connect(REPORT_DB)
    try:
        con.execute(
            "CREATE TABLE IF NOT EXISTS reports("
            "id TEXT PRIMARY KEY, created_at INTEGER, reason TEXT, content TEXT, ip_hash TEXT)"
        )
        salt = os.getenv("REPORT_HASH_SALT", "genvora")
        token = (
            hashlib.sha256((salt + ip).encode()).hexdigest()[:24]
            if ip
            else "unknown"
        )
        con.execute(
            "INSERT INTO reports VALUES(?,?,?,?,?)",
            (str(uuid.uuid4()), int(time.time()), reason, content, token),
        )
        con.commit()
    finally:
        con.close()


@app.get("/")
async def root():
    return {"ok": True, "service": APP_NAME}


@app.get("/health")
async def health():
    return {
        "ok": True,
        "service": APP_NAME,
        "provider": "gemini",
        "text_model": GEMINI_MODEL,
        "vision_model": GEMINI_MODEL,
        "vision_enabled": True,
    }


def build_gemini_payload(req: ChatRequest) -> dict:
    contents = []

    for m in req.messages:
        role = "model" if m.role == "assistant" else "user"
        contents.append({
            "role": role,
            "parts": [{"text": m.content}],
        })

    # Attach the image to the latest user turn.
    if req.image_base64:
        for item in reversed(contents):
            if item.get("role") == "user":
                item["parts"].append({
                    "inlineData": {
                        "mimeType": "image/jpeg",
                        "data": req.image_base64,
                    }
                })
                break

    return {
        "systemInstruction": {
            "parts": [{"text": req.system_prompt}]
        },
        "contents": contents,
    }


def extract_gemini_text(data: dict) -> str:
    candidates = data.get("candidates") or []
    if not candidates:
        prompt_feedback = data.get("promptFeedback") or {}
        reason = prompt_feedback.get("blockReason")
        if reason:
            raise HTTPException(
                status_code=400,
                detail=f"Gemini blocked this request: {reason}",
            )
        return ""

    parts = (
        candidates[0]
        .get("content", {})
        .get("parts", [])
    )
    text_parts = [
        p.get("text", "")
        for p in parts
        if isinstance(p, dict) and p.get("text")
    ]
    return "\n".join(text_parts).strip()


async def call_gemini(req: ChatRequest, model: str) -> str:
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not configured.",
        )

    url = f"{GEMINI_BASE_URL}/models/{model}:generateContent"
    headers = {
        "x-goog-api-key": GEMINI_API_KEY,
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.post(
            url,
            headers=headers,
            json=build_gemini_payload(req),
        )

        if response.status_code >= 400:
            detail = f"Gemini API error: {response.status_code}"
            try:
                body = response.json()
                message = (
                    body.get("error", {}).get("message")
                    if isinstance(body, dict)
                    else None
                )
                if message:
                    detail += f" - {message}"
            except Exception:
                pass
            raise HTTPException(status_code=502, detail=detail)

        return extract_gemini_text(response.json())


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    ip = request.client.host if request.client else "unknown"
    rate_limit(ip)

    if total_chars(req) > MAX_INPUT_CHARS:
        raise HTTPException(status_code=413, detail="Conversation is too large.")

    if any(blocked(m.content) for m in req.messages):
        raise HTTPException(
            status_code=400,
            detail="This request cannot be processed safely.",
        )

    request_id = str(uuid.uuid4())
    model = req.model or GEMINI_MODEL

    try:
        reply = await call_gemini(req, model)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="AI provider timed out.")

    if not reply:
        raise HTTPException(
            status_code=502,
            detail="Gemini returned an empty response.",
        )

    if blocked(reply):
        reply = "I can’t provide that content. I can help with a safer alternative."

    return ChatResponse(
        reply=reply,
        model=model,
        request_id=request_id,
    )


@app.post("/v1/report")
async def report(body: ReportRequest, request: Request):
    ip = request.client.host if request.client else "unknown"
    rate_limit(ip)
    save_report(body.content, body.reason, ip)
    return {"ok": True, "message": "Report received"}


PRIVACY = """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>GenVora AI Privacy Policy</title></head><body><h1>GenVora AI Privacy Policy</h1><p>GenVora AI sends the text and optional images you submit to its backend and configured AI provider in order to generate responses.</p><p>Before publishing, replace this page with your final developer privacy policy and support contact.</p></body></html>"""

TERMS = """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>GenVora AI Terms</title></head><body><h1>GenVora AI Terms of Use</h1><p>GenVora AI can make mistakes. Do not rely on it as a substitute for qualified professional advice in high-stakes situations.</p><p>Before publishing, replace this page with your final developer terms and support contact.</p></body></html>"""


@app.get("/privacy", response_class=HTMLResponse)
async def privacy():
    return PRIVACY


@app.get("/terms", response_class=HTMLResponse)
async def terms():
    return TERMS
