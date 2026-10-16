import json

import llm
import store


def sse(event):
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def run_chat(question, history):
    answer, calls = "", []
    try:
        for ev in llm.chat_stream(question, history, store.liked_examples(2)):
            t = ev["type"]
            if t == "token":
                answer += ev["text"]
            elif t == "replace":
                answer = ev["text"]
            elif t == "reset":
                answer = ""
            elif t == "tools":
                calls = ev["calls"]
            elif t == "done":
                ev = {**ev, "message_id": store.save_message(question, answer.strip(), calls, ev["meta"])}
            yield ev
    except Exception as e:
        yield {"type": "error", "text": f"{e.__class__.__name__}: {e}"}
