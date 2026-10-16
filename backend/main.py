from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel

import llm
import service
import store

app = FastAPI(title="Turkey Weather Chatbot")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["*"], allow_headers=["*"])


class ChatRequest(BaseModel):
    question: str
    history: list[dict] = []


class FeedbackRequest(BaseModel):
    message_id: int
    rating: int
    reason: str | None = None
    correction: str | None = None


@app.get("/api/health")
def health():
    return {"ollama": llm.ready(), "model": llm.MODEL, "mode": llm.TOOL_MODE, **store.stats()}


@app.post("/api/chat")
def chat(req: ChatRequest):
    q = req.question.strip()
    if not q:
        raise HTTPException(400, "question is empty")
    events = (service.sse(e) for e in service.run_chat(q, req.history))
    return StreamingResponse(events, media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/feedback")
def feedback(f: FeedbackRequest):
    try:
        store.set_feedback(f.message_id, f.rating, f.reason, f.correction)
    except KeyError:
        raise HTTPException(404, "message not found")
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True, **store.stats()}


@app.get("/api/export/{kind}", response_class=PlainTextResponse)
def export(kind: str):
    if kind not in ("sft", "dpo", "eval"):
        raise HTTPException(404, "kind must be sft, dpo or eval")
    return store.export(kind)
