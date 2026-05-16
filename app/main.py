"""
SHL Assessment Recommender — FastAPI application.

GET  /health   → {"status": "ok"}
POST /chat     → ChatResponse (reply + recommendations + end_of_conversation)
"""
from __future__ import annotations

import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.agent import run_agent
from app.models import ChatRequest, ChatResponse, HealthResponse
from app.retrieval import retriever

# ---------------------------------------------------------------------------
# Startup: load the index once
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up — loading catalog and retrieval index...")
    try:
        retriever.load()
    except Exception as exc:
        print(f"[WARNING] Retriever failed to load: {exc}", file=sys.stderr)
    yield
    # Nothing to teardown


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SHL Assessment Recommender",
    description=(
        "Conversational agent that helps hiring managers find SHL "
        "Individual Test Assessments through dialogue."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def _generic_handler(request: Request, exc: Exception):
    print(f"Unhandled exception on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error — please try again."},
    )

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Readiness probe",
    tags=["Ops"],
)
async def health() -> HealthResponse:
    """Returns {"status": "ok"} when the service is ready."""
    return HealthResponse(status="ok")


@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Conversational assessment recommender",
    tags=["Chat"],
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Stateless conversational endpoint.

    Pass the **full** conversation history on every call.
    The service stores no per-conversation state.

    - **messages**: list of ``{role, content}`` pairs; last message must be from "user".
    - Returns **reply** (string), **recommendations** (0–10 items), and **end_of_conversation** (bool).
    """
    if not request.messages:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="`messages` must not be empty.",
        )

    if request.messages[-1].role != "user":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The last message must have role='user'.",
        )

    # Hard turn-cap enforcement — count user turns (not total messages)
    user_turn_count = sum(1 for m in request.messages if m.role == "user")
    if user_turn_count > 8:
        return ChatResponse(
            reply=(
                "We have reached the maximum conversation length. "
                "I hope the recommendations above were useful! "
                "Feel free to start a new conversation any time."
            ),
            recommendations=[],
            end_of_conversation=True,
        )

    t0 = time.monotonic()
    try:
        response = run_agent(request, retriever)
    except Exception as exc:
        print(f"Agent error: {exc}")
        response = ChatResponse(
            reply="I'm having trouble right now. Please try again in a moment.",
            recommendations=[],
            end_of_conversation=False,
        )

    # Hard cap: schema guarantees 0–10 recommendations
    if len(response.recommendations) > 10:
        response = ChatResponse(
            reply=response.reply,
            recommendations=response.recommendations[:10],
            end_of_conversation=response.end_of_conversation,
        )

    elapsed = time.monotonic() - t0
    print(f"[/chat] turns={len(request.messages)} elapsed={elapsed:.2f}s recs={len(response.recommendations)}")
    return response
