export async function health() {
  const r = await fetch("/api/health");
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

export async function sendFeedback(body) {
  const r = await fetch("/api/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

/** Calls onEvent(event) for every event as it arrives. Resolves when the stream ends. */
export async function streamChat(question, history, onEvent, signal) {
  const r = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, history }),
    signal,
  });
  if (!r.ok || !r.body) throw new Error(`HTTP ${r.status}`);

  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop();
    for (const block of blocks) {
      if (block.startsWith("data: ")) onEvent(JSON.parse(block.slice(6)));
    }
  }
}
