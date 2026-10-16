# Turkey Weather Chatbot (Qwen + tools + React)

```
React (Vite, :5173) --/api/chat (SSE stream)--> FastAPI (:8000) --> llm.py: plan -> run tools -> streamed answer -> number check
        |  👍/👎 + reason + correction                |                      |                         |
        +--/api/feedback--> store.py (SQLite) <-------+                  Ollama (Qwen)           tools.py (pandas)
```

## Run (two terminals)
Backend (in `backend/`):
```
pip install -r requirements.txt
set QWEN_MODEL=qwen3:8b            (PowerShell: $env:QWEN_MODEL="qwen3:8b")
uvicorn main:app --port 8000
```
Frontend (in `frontend/`):
```
npm install
npm run dev                        # open http://localhost:5173
```

## Feedback data (feedback.db is created automatically)
- http://localhost:8000/api/export/sft   liked answers -> fine-tuning pairs
- http://localhost:8000/api/export/dpo   disliked answer + your correction -> preference pairs
- http://localhost:8000/api/export/eval  every disliked question + tools used + reason -> add to eval_tools.py
Liked AND number-checked answers are also fed back as style examples in the system prompt (2 latest).
