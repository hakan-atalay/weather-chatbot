import time

import llm

if not llm.ready():
    raise SystemExit(f"Ollama is not reachable at {llm.OLLAMA_URL}. Start it, then: ollama pull {llm.MODEL}")

print(f"Model: {llm.MODEL} | mode: {llm.TOOL_MODE}  (type 'q' to quit)\n")
history = []
while True:
    q = input("You: ").strip()
    if q.lower() in ("q", "quit", "exit"):
        break
    if not q:
        continue
    t0 = time.time()
    out = llm.chat(q, history[-6:])
    history += [{"role": "user", "content": q}, {"role": "assistant", "content": out["answer"]}]
    for c in out["tool_calls"]:
        print(f"   [tool] {c['name']}({c['args']})")
    print(f"Bot: {out['answer']}")
    if not out["tool_calls"]:
        print("   [i] no tool used (expected for greetings or questions the data cannot answer)")
    if out["ungrounded_numbers"]:
        print(f"   [!] numbers not found in tool results: {out['ungrounded_numbers']}")
    if out.get("unsupported_terms"):
        print(f"   [!] claims the daily data cannot support: {out['unsupported_terms']}")
    print(f"   ({time.time() - t0:.0f}s, mode={out['mode']})\n")
