"""Qwen (served by Ollama) + tools.  Three stages instead of hoping the model calls tools on its own:

  1. PLAN    Qwen must output JSON {"calls":[{"tool":..., "args":{...}}]}. Ollama enforces the JSON schema
             (structured output), so a small model cannot "forget" to use a tool or invent a tool name.
  2. EXECUTE we run the planned tools (tools.py). No LLM involved, numbers are exact.
  3. ANSWER  Qwen writes the answer from the tool results only. A grounding check then verifies that every
             number in the answer exists in the results; otherwise Qwen is asked to rewrite once.

TOOL_MODE=native switches to the classic function-calling loop (used automatically if planning fails).
"""
import datetime as dt
import json
import os
import re

import requests

import tools

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("QWEN_MODEL", "qwen2.5:7b-instruct")      # try "qwen3:8b"
TOOL_MODE = os.getenv("TOOL_MODE", "plan")                   # "plan" | "native"
MAX_TOOL_CHARS = 5000
MAX_ROUNDS = 5

SYSTEM_PROMPT = """You are a weather and climate analyst for Turkey. Today is {today}.
Datasets behind the tools:
- 81 provinces, daily max/min temperature and precipitation, 2022-10-08 to 2026-10-07 (4 years).
- Kutahya only: the same variables from 1940-01-01 to 2026-10-13 (86 years).
The data is ERA5 reanalysis (model-based), not official weather-station measurements.

Rules:
1. NEVER calculate, estimate or invent numbers. Every number in your answer must come from the tool results.
2. Describe the period exactly as the results show it ("period", "data_range", "days_covered"). Never claim a
   year or month the results do not cover.
3. Long-term trends, frost calendar and trend projections exist for Kutahya only. For other provinces explain that
   only 4 years exist, which is too short for climate trends, and offer what you CAN do (ranking, comparison,
   seasonal expectation with a warning).
4. Three different kinds of "future" - always say which one you used:
   a) live_forecast = a real weather forecast; b) seasonal_outlook = what is typical for that month (not a
   forecast); c) project_trend = statistical extrapolation of the Kutahya trend (not a climate model).
5. If a result has "warning" or "error", tell the user plainly. If it has "did_you_mean", ask which province they meant.
6. If no tool results are provided, no tool fits this question: do not give any numbers beyond the dataset coverage; explain the limit and what you can do instead.
7. The data is DAILY. Never mention time of day (morning, evening, hourly) - the tools do not know it.
8. If a result contains "plain_summary_tr", base your wording on it; do not swap terms such as last spring frost and first autumn frost.
9. Reply in the language of the user's question (Turkish by default). Be concise: lead with the direct answer,
   then one or two sentences of interpretation. Maximum 4 sentences unless the user asks for detail. Do not repeat raw JSON field names."""


# ---------------------------------------------------------------- Ollama plumbing
def ready():
    try:
        return requests.get(f"{OLLAMA_URL}/api/tags", timeout=1.5).ok
    except requests.RequestException:
        return False


def _payload(payload, stream):
    payload = {"model": MODEL, "stream": stream, **payload}
    # Ollama's default context is 4096 tokens and silently cuts the START of the prompt when it overflows.
    payload["options"] = {**payload.get("options", {}), "num_ctx": int(os.getenv("NUM_CTX", "8192"))}
    if "qwen3" in MODEL:
        payload["think"] = False
    return payload


def _post(payload):
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=_payload(payload, False), timeout=300)
    r.raise_for_status()
    return r.json()["message"]


def _post_stream(payload):
    """Yield the answer text piece by piece while Ollama is still generating it."""
    with requests.post(f"{OLLAMA_URL}/api/chat", json=_payload(payload, True), stream=True, timeout=300) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            piece = chunk.get("message", {}).get("content", "")
            if piece:
                yield piece
            if chunk.get("done"):
                break


def _system(examples=None):
    s = SYSTEM_PROMPT.format(today=dt.date.today().isoformat())
    if examples:
        s += "\n\nExamples of answers users liked (match this style):\n" + "\n".join(f"Q: {q}\nA: {a}" for q, a in examples)
    return s


# ---------------------------------------------------------------- stage 1: plan
def _plan_schema():
    alts = [{"type": "object", "properties": {"tool": {"enum": [t["function"]["name"]]}, "args": t["function"]["parameters"]},
             "required": ["tool", "args"]} for t in tools.build_schema()]
    return {"type": "object", "required": ["calls"],
            "properties": {"calls": {"type": "array", "maxItems": 3, "items": {"anyOf": alts}}}}


def _catalog():
    lines = []
    for t in tools.build_schema():
        f = t["function"]
        params = ", ".join(
            f"{n}: " + ("|".join(p["enum"]) if "enum" in p else p["type"]) + ("" if n in f["parameters"]["required"] else "?")
            for n, p in f["parameters"]["properties"].items())
        lines.append(f"- {f['name']}({params}): {f['description'].split('. ')[0].rstrip('.')}.")
    return "\n".join(lines)


def _planner_prompt():
    today = dt.date.today()
    tomorrow = today + dt.timedelta(days=1)
    last_month = today.replace(day=1) - dt.timedelta(days=1)

    def ex(q, calls):
        return f"Q: {q}\n{json.dumps({'calls': calls}, ensure_ascii=False)}"
    return f"""You turn a user question into tool calls. Output ONLY JSON: {{"calls":[{{"tool":..., "args":{{...}}}}]}}.
Today is {today} (tomorrow is {tomorrow}). Provinces: all 81 in Turkey. Kutahya also has 86 years of history.
Relative dates, already calculated for you - copy these numbers, never calculate dates yourself:
- this year = {today.year}, last year = {today.year - 1}
- this month = year {today.year} month {today.month}; last month ("geçen ay") = year {last_month.year} month {last_month.month}

Rules:
- Put EVERY constraint of the question into args: province, year, month, threshold, date, number of results.
- Seasons: ilkbahar=spring, yaz=summer, sonbahar=autumn, kis=winter -> use get_summary(season=...), NOT month.
- Months: Ocak=1 Subat=2 Mart=3 Nisan=4 Mayis=5 Haziran=6 Temmuz=7 Agustos=8 Eylul=9 Ekim=10 Kasim=11 Aralik=12.
- Use province names exactly as the user wrote them.
- Warming / getting drier / trend → long_term_trend (it works for Kutahya ONLY). Trend of ANOTHER province → no calls.
- Planting, fide, frost, don → frost_calendar.  "In 2035" style questions → project_trend.
- "Tomorrow / next days / this week" → live_forecast (single day: target_date).  "Typical for month X" → seasonal_outlook.
- Greeting, off-topic, or nothing fits → {{"calls": []}}.  At most 3 calls.

Tools:
{_catalog()}

Examples:
{ex("Agustos 2025'te en sicak 5 il hangisi?", [{"tool": "rank_provinces", "args": {"metric": "max_temp", "year": 2025, "month": 8, "top_n": 5, "order": "highest"}}])}
{ex("Kütahya Eskişehir'den serin mi?", [{"tool": "compare_provinces", "args": {"province_a": "Kütahya", "province_b": "Eskişehir"}}])}
{ex("Kütahya ısınıyor mu?", [{"tool": "long_term_trend", "args": {"metric": "mean_temp"}}])}
{ex("Ankara ısınıyor mu?", [])}
{ex("Yarın İzmir'de yağmur var mı?", [{"tool": "live_forecast", "args": {"province": "İzmir", "target_date": str(tomorrow)}}])}
{ex("Kütahya'da fide ne zaman dikilir?", [{"tool": "frost_calendar", "args": {}}])}
{ex("Geçen ay Muğla'da hava nasıldı?", [{"tool": "get_summary", "args": {"province": "Muğla", "year": last_month.year, "month": last_month.month}}])}
{ex("Konya'da 2024 yazı nasıldı?", [{"tool": "get_summary", "args": {"province": "Konya", "year": 2024, "season": "summer"}}])}
{ex("Merhaba", [])}"""


def plan(question, history=None):
    msgs = [{"role": "system", "content": _planner_prompt()}]
    prev = [m["content"] for m in (history or []) if m.get("role") == "user"][-2:]
    if prev:
        msgs.append({"role": "system", "content": "Earlier user questions (ONLY to fill in what a follow-up leaves out, such as the province or period; NEVER plan calls for them): " + " | ".join(prev)})
    msgs.append({"role": "user", "content": "Plan calls ONLY for this question: " + question})
    msg = _post({"messages": msgs, "format": _plan_schema(), "options": {"temperature": 0}})
    calls = json.loads(msg["content"]).get("calls", [])
    return [c for c in calls if c.get("tool") in tools.REGISTRY][:3]


# ---------------------------------------------------------------- stage 3 helper: grounding check
# numbers that are legitimately known from the system prompt (dataset coverage), so they are never "invented"
DATASET_FACTS = "1940 2022 2026 81 4 86 16"          # years and counts only: date parts like "01" must not whitelist the number 1
DATASET_DATES = ("2022-10-08", "2026-10-07", "1940-01-01", "2026-10-13")   # removed from the answer as whole strings
_NUM = re.compile(r"\d+(?:[.,]\d+)?")


def _tokens(text):
    out = []
    for t in _NUM.findall(text):
        t = t.replace(",", ".")
        out.append((float(t), len(t.split(".")[1]) if "." in t else 0))
    return out


def ungrounded_numbers(answer, evidence, question=""):
    """Numbers in `answer` that appear in neither the tool results, the question nor the dataset facts.
    Rounding is allowed (22.414 -> 22.4 or 22). List markers ("1.", "2)") and the constants 10 (decade) and
    100 (percent) are ignored. Small integers are NOT a free pass: "1 day" when the result says 0.152 is flagged."""
    allowed = [v for v, _ in _tokens(evidence + " " + question + " " + DATASET_FACTS + " " + dt.date.today().isoformat())]
    answer = re.sub(r"(?m)^\s*(?:[-*]\s*)?\d+[.)]\s", "", answer)
    for d in DATASET_DATES:
        answer = answer.replace(d, " ")
    bad = []
    for v, d in _tokens(answer):
        if d == 0 and v in (10, 100):
            continue
        if any(abs(v - a) <= 0.5 * 10 ** (-d) + 1e-6 for a in allowed):
            continue
        bad.append(v)
    return bad


# claims the daily data cannot support: time of day. Allowed only if the user's question used the word itself.
_TIME_WORDS = re.compile(r"\b(sabah|akşam|öğle|saat)\w*")


def unsupported_terms(answer, question=""):
    q = question.lower()
    if _TIME_WORDS.search(q):          # the user asked about time of day: the model may legitimately explain there is none
        return []
    return sorted({m.group(0) for m in _TIME_WORDS.finditer(answer.lower()) if m.group(1) not in q})


# ---------------------------------------------------------------- the three-stage chat
def _plan_events(question, history, examples):
    """plan -> execute -> streamed answer -> grounding check. Yields: status | tools | chart | token | replace | done"""
    yield {"type": "status", "text": "planning"}
    calls = plan(question, history)
    results, used, chart = [], [], None
    for c in calls:
        res = tools.call_tool(c["tool"], c.get("args") or {})
        used.append({"name": c["tool"], "args": c.get("args") or {}})
        chart = res.get("chart") or chart
        results.append({"tool": c["tool"], "args": c.get("args") or {}, "result": {k: v for k, v in res.items() if k != "chart"}})
    yield {"type": "tools", "calls": used}
    if chart:
        yield {"type": "chart", "chart": chart}
    evidence = json.dumps(results, ensure_ascii=False)[:MAX_TOOL_CHARS * 2]
    user = f"{question}\n\n[TOOL RESULTS - use only these numbers]\n{evidence if results else '(none: no tool applies to this question)'}"
    msgs = [{"role": "system", "content": _system(examples)}, {"role": "user", "content": user}]
    yield {"type": "status", "text": "writing"}
    answer = ""
    for piece in _post_stream({"messages": msgs, "options": {"temperature": 0.2}}):
        answer += piece
        yield {"type": "token", "text": piece}
    answer = answer.strip()

    bad, terms = ungrounded_numbers(answer, evidence, question), unsupported_terms(answer, question)
    if bad or terms:
        yield {"type": "status", "text": "checking"}
        fix = []
        if bad:
            fix.append(f"These numbers are NOT in the tool results: {bad}. Use only numbers from the results (rounding is fine); do not calculate new ones.")
        if terms:
            fix.append(f"You used {terms}, but the data is daily and has no time-of-day information. Remove such statements.")
        msgs += [{"role": "assistant", "content": answer}, {"role": "user", "content": "Rewrite the answer. " + " ".join(fix)}]
        answer = _post({"messages": msgs, "options": {"temperature": 0.1}}).get("content", "").strip()
        bad, terms = ungrounded_numbers(answer, evidence, question), unsupported_terms(answer, question)
        yield {"type": "replace", "text": answer}
    yield {"type": "done", "meta": {"grounded": not (bad or terms), "ungrounded_numbers": bad, "unsupported_terms": terms,
                                    "mode": "plan", "model": MODEL}}


def _chat_native(question, history, examples):
    """Classic function calling: the model decides alone. Less reliable on small models."""
    messages = [{"role": "system", "content": _system(examples)}] + (history or []) + [{"role": "user", "content": question}]
    schema, used, chart = tools.build_schema(), [], None
    for _ in range(MAX_ROUNDS):
        msg = _post({"messages": messages, "tools": schema, "options": {"temperature": 0.2}})
        msg.setdefault("role", "assistant")
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            return {"answer": (msg.get("content") or "").strip(), "tool_calls": used, "chart": chart, "model": MODEL,
                    "mode": "native", "grounded": bool(used), "ungrounded_numbers": [], "unsupported_terms": []}
        for c in calls:
            name, args = c["function"]["name"], c["function"].get("arguments") or {}
            if isinstance(args, str):
                args = json.loads(args or "{}")
            res = tools.call_tool(name, args)
            used.append({"name": name, "args": args})
            chart = res.get("chart") or chart
            slim = {k: v for k, v in res.items() if k != "chart"}
            messages.append({"role": "tool", "tool_name": name, "content": json.dumps(slim, ensure_ascii=False)[:MAX_TOOL_CHARS]})
    return {"answer": "The analysis needed too many steps. Could you ask a more specific question?", "tool_calls": used,
            "chart": chart, "model": MODEL, "mode": "native", "grounded": False, "ungrounded_numbers": [], "unsupported_terms": []}


def chat_stream(question, history=None, examples=None):
    """Generator of UI events. Falls back to native tool calling if the plan stage fails."""
    history = [m for m in (history or []) if m.get("role") == "user"][-3:]
    if TOOL_MODE == "plan":
        try:
            yield from _plan_events(question, history, examples)
            return
        except (ValueError, KeyError, requests.HTTPError) as e:
            print(f"[plan mode failed: {e.__class__.__name__}: {e}] -> falling back to native tool calling")
            yield {"type": "reset"}
    r = _chat_native(question, history, examples)
    yield {"type": "tools", "calls": r["tool_calls"]}
    if r["chart"]:
        yield {"type": "chart", "chart": r["chart"]}
    yield {"type": "replace", "text": r["answer"]}
    yield {"type": "done", "meta": {k: r[k] for k in ("grounded", "ungrounded_numbers", "unsupported_terms", "mode", "model")}}


def chat(question, history=None, examples=None):
    """Non-streaming interface (CLI, eval): runs chat_stream and returns one dict."""
    out = {"answer": "", "tool_calls": [], "chart": None}
    for ev in chat_stream(question, history, examples):
        t = ev["type"]
        if t == "token":
            out["answer"] += ev["text"]
        elif t == "replace":
            out["answer"] = ev["text"]
        elif t == "reset":
            out["answer"] = ""
        elif t == "tools":
            out["tool_calls"] = ev["calls"]
        elif t == "chart":
            out["chart"] = ev["chart"]
        elif t == "done":
            out.update(ev["meta"])
    out["answer"] = out["answer"].strip()
    return out