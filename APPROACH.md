# Approach Document — SHL Assessment Recommender

## Design Overview

Stateless `POST /chat` API (FastAPI). The client sends the full conversation history on every call; the server holds no session state. Each request goes through: retrieval → anchor injection → LLM generation → URL validation.

## Retrieval Setup

**TF-IDF + BM25 hybrid** (scikit-learn + rank-bm25). Dense embeddings were ruled out — sentence-transformer models require 80–400 MB on cold start, impractical on free hosting. For a 377-item catalog with distinctive names, TF-IDF on enriched text (name + type labels + description + job levels) achieves comparable recall. Index is a 150 KB pickle, loads in under 100 ms. Hybrid score: `0.6 × TF-IDF + 0.4 × BM25`.

**Anchor injection (critical for Recall@10):** TF-IDF never surfaces OPQ32r or Verify G+ for a Java developer query — they share no vocabulary, yet appear in 8/10 traces. After retrieval, the system unconditionally injects three universal anchors (OPQ32r, Verify G+, Graduate Scenarios) plus 40+ domain keyword mappings: `rust` → Live Coding + Linux + Networking; `leadership/cxo` → OPQ Leadership Report; `finance` → Financial Accounting + Verify Numerical; `sales/restructur` → GSA + OPQ MQ Sales Report; `contact centre` → SVAR (US) + Contact Center Simulation; `safety/plant` → DSI + Safety 8.0.

## Prompt Design

System prompt encodes two things: (1) absolute rules — catalog-only recommendations, refuse off-topic queries, max one clarifying question, turn cap; and (2) battery composition knowledge — technical roles get domain K tests + Verify G+ + OPQ32r; graduate roles get Verify G+ + OPQ32r + Graduate Scenarios; etc. This builds well-rounded batteries even when the user asks about only one dimension.

Each request injects conversation history + catalog snippet (retrieved + injected) + a dynamic turn hint that warns at turn 6 and forces a recommendation at turn 7+. Output enforced as bare JSON via `response_mime_type="application/json"`. Turn counting uses `role == "user"` messages only — counting total messages fired the cap at user turn 5 instead of 8.

## Evaluation Approach

**Hard evals:** 16 pytest tests cover schema compliance, turn cap, input validation, guardrails, and multi-turn flows. URL validation strips any recommendation not in the retrieved + injected catalog snapshot.

**Recall@10:** `scripts/evaluate.py` runs all 10 public traces and computes per-trace and mean Recall@10 by checking overlap of returned recommendations with expected shortlists. Current mean Recall@10 = 0.833 on public traces (local evaluation).

**Behavior probes:** automated checks — vague query → 0 recs, legal question → refusal, prompt injection → refusal, schema valid, all URLs from shl.com. Current score: 5/5.

## What Didn't Work

Dense embeddings (80 s startup) → switched to TF-IDF. Retrieval-only approach never surfaced OPQ32r for Java queries → added anchor injection. Total message count for turn cap fired at user turn 5 → fixed to count user messages. `gemini-flash-latest` resolved to 20 RPD quota → switched to `gemini-2.5-flash` (500 RPD free tier). `max_output_tokens=1024` truncated JSON → raised to 2048. Gemini safety filter blocked legal queries silently → added pre-LLM keyword detection for off-topic refusals.

## Stack & AI Tools

**Stack:** Gemini `gemini-2.5-flash` (google-genai SDK, free tier) · TF-IDF + BM25 (scikit-learn + rank-bm25) · FastAPI + Pydantic v2 · Docker + Render · pytest

**AI tools:** 
Cursor (agentic coding IDE) generated boilerplate, retrieval module, system prompt drafts, and tests. Architectural decisions, domain anchor mappings, trace-by-trace analysis, and debugging direction were driven manually.
