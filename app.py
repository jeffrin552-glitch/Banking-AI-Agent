"""
app.py — BankAssist AI (FastAPI)
Production-ready entry point for Render deployment.
"""
import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Load .env for local development (no-op on Render where env vars are set directly)
load_dotenv()

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── API Key validation ────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY environment variable is not set. "
        "Add it in Render → Environment → Environment Variables."
    )

# ── Lazy-import agent (keeps startup fast; errors surface clearly) ────────────
from agents.banking_agent import BankingConversation

conversation = BankingConversation(groq_api_key=GROQ_API_KEY)
logger.info("BankingConversation agent initialised successfully.")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="BankAssist AI",
    description="LangGraph-powered agentic banking assistant",
    version="1.0.0",
)

templates = Jinja2Templates(directory="templates")


# ── Request / Response models ─────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chat")
async def chat(data: ChatRequest):
    user_message = data.message.strip()
    if not user_message:
        return JSONResponse({"error": "Empty message"}, status_code=400)
    try:
        reply = conversation.chat(user_message)
        return {"reply": reply}
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        logger.error("Error in /chat:\n%s", tb)
        # Return the actual error in development; mask it in prod if needed
        error_detail = str(exc)
        return JSONResponse(
            {"reply": f"⚠️ Error: {error_detail[:200]}"},
            status_code=500,
        )


@app.post("/reset")
async def reset_chat():
    conversation.reset()
    return {"status": "ok"}


@app.get("/accounts")
async def get_accounts():
    from database import db_manager
    return {"accounts": db_manager.get_all_accounts()}


@app.get("/health")
async def health():
    """Health-check endpoint used by Render."""
    return {"status": "ok"}
