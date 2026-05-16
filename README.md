# SHL Assessment Recommender

A conversational agent that guides hiring managers from a vague intent to a grounded shortlist of SHL Individual Test Assessments through natural dialogue.

---

## Quick Start

### 1. Prerequisites

- Python 3.11+ (locally) or Docker
- A [Google Gemini API key](https://aistudio.google.com/app/apikey) (free tier is sufficient)

### 2. Environment

```bash
cp .env.example .env
# edit .env and set GOOGLE_API_KEY=your_key_here
```

### 3. Install & Run (local)

```bash
# One-time setup
bash setup.sh

# Start the server
bash start.sh
```

The API will be available at `http://localhost:8000`.

### 4. Run with Docker

```bash
docker build -t shl-recommender .
docker run -e GOOGLE_API_KEY=your_key -p 8000:8000 shl-recommender
```

---

## API

### `GET /health`

```json
{"status": "ok"}
```

### `POST /chat`

**Request** — pass the **full** conversation history on every call (stateless):

```json
{
  "messages": [
    {"role": "user", "content": "I am hiring a Java developer who works with stakeholders"},
    {"role": "assistant", "content": "What seniority level are you targeting?"},
    {"role": "user", "content": "Mid-level, around 4 years of experience"}
  ]
}
```

**Response**:

```json
{
  "reply": "Here are 5 assessments for a mid-level Java developer...",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "Occupational Personality Questionnaire OPQ32r", "url": "...", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

- `recommendations` is `[]` when the agent is clarifying or refusing.
- `end_of_conversation` is `true` only when the agent has delivered a final shortlist.
- Maximum 8 turns per conversation (user + assistant combined).

---

## Project Structure

```
shl-recommender/
├── app/
│   ├── main.py          # FastAPI app (/health, /chat)
│   ├── agent.py         # Gemini-based conversational agent
│   ├── retrieval.py     # Hybrid TF-IDF + BM25 retrieval
│   └── models.py        # Pydantic request/response schemas
├── scripts/
│   ├── save_catalog.py  # Persist the 377-item scraped catalog
│   ├── build_index.py   # Build TF-IDF index from catalog
│   └── fetch_all_catalog.py  # Full scraper (run to refresh catalog)
├── data/
│   ├── catalog.json     # 377 SHL Individual Test Solutions
│   └── tfidf_index.pkl  # Pre-built retrieval index
├── tests/
│   └── test_api.py      # 16 API-level tests
├── Dockerfile
├── render.yaml          # One-click Render deployment
├── setup.sh             # One-time setup script
└── start.sh             # Server start script
```

---

## Tests

```bash
source venv/bin/activate
pytest tests/ -v
```

All 16 tests cover schema compliance, turn-cap enforcement, input validation, guardrails, and multi-turn flows.

---

## Deployment (Render)

1. Push this repo to GitHub.
2. Create a new **Web Service** on [render.com](https://render.com), connect the repo.
3. Render will use `render.yaml` automatically.
4. Set `GOOGLE_API_KEY` in the Render environment variables dashboard.
5. The `/health` endpoint is used as the health check — Render allows up to 2 minutes for cold start.
