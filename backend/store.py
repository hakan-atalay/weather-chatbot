import json
import os
import sqlite3
import time
from pathlib import Path

DB = Path(os.getenv("FEEDBACK_DB", Path(__file__).parent / "feedback.db"))
REASONS = ("wrong_number", "wrong_scope", "wrong_analysis", "too_long_or_unclear", "off_topic", "other")


def _conn():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    c.execute("""CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, question TEXT, answer TEXT, tool_calls TEXT,
        model TEXT, mode TEXT, grounded INTEGER, rating INTEGER, reason TEXT, correction TEXT)""")
    return c


def save_message(question, answer, tool_calls, meta):
    with _conn() as c:
        cur = c.execute("INSERT INTO messages (ts, question, answer, tool_calls, model, mode, grounded) VALUES (?,?,?,?,?,?,?)",
                        (time.time(), question, answer, json.dumps(tool_calls, ensure_ascii=False),
                         meta.get("model"), meta.get("mode"), int(bool(meta.get("grounded")))))
        return cur.lastrowid


def set_feedback(message_id, rating, reason=None, correction=None):
    if rating not in (1, -1):
        raise ValueError("rating must be 1 (like) or -1 (dislike)")
    if reason and reason not in REASONS:
        raise ValueError(f"reason must be one of {REASONS}")
    with _conn() as c:
        n = c.execute("UPDATE messages SET rating=?, reason=?, correction=? WHERE id=?",
                      (rating, reason, (correction or "").strip() or None, message_id)).rowcount
    if not n:
        raise KeyError("message not found")


def liked_examples(n=2):
    with _conn() as c:
        return [(r["question"], r["answer"]) for r in
                c.execute("SELECT question, answer FROM messages WHERE rating=1 AND grounded=1 ORDER BY id DESC LIMIT ?", (n,))]


def stats():
    with _conn() as c:
        r = c.execute("SELECT COUNT(*) n, COALESCE(SUM(rating=1),0) up, COALESCE(SUM(rating=-1),0) down, "
                      "COALESCE(SUM(correction IS NOT NULL),0) fixes FROM messages").fetchone()
    return {"messages": r["n"], "likes": r["up"], "dislikes": r["down"], "corrections": r["fixes"]}


def export(kind):
    rows = []
    with _conn() as c:
        for r in c.execute("SELECT * FROM messages WHERE rating IS NOT NULL ORDER BY id"):
            if kind == "sft" and (r["rating"] == 1 or r["correction"]):
                rows.append({"messages": [{"role": "user", "content": r["question"]},
                                          {"role": "assistant", "content": r["correction"] or r["answer"]}],
                             "tool_calls": json.loads(r["tool_calls"] or "[]")})
            elif kind == "dpo" and r["rating"] == -1 and r["correction"]:
                rows.append({"prompt": r["question"], "chosen": r["correction"], "rejected": r["answer"]})
            elif kind == "eval" and r["rating"] == -1:
                rows.append({"question": r["question"], "tools_used": json.loads(r["tool_calls"] or "[]"),
                             "reason": r["reason"], "correction": r["correction"]})
    return "\n".join(json.dumps(x, ensure_ascii=False) for x in rows)
