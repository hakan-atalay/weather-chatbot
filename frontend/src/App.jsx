import { useEffect, useRef, useState } from "react";
import { health, streamChat } from "./api.js";
import Chart from "./Chart.jsx";
import Feedback from "./Feedback.jsx";

const EXAMPLES = [
  "Ağustos 2025'te en sıcak 5 il hangisi?",
  "Kütahya Eskişehir'den serin mi?",
  "Kütahya ısınıyor mu?",
  "Kütahya'da fide ne zaman dikilir?",
  "Yarın Kütahya'da yağmur olacak mı?",
  "2035'te Kütahya'da sıcaklık ne olur?",
  "Van'a iklimce benzeyen iller hangileri?",
  "Kütahya'da kaç gün 35 dereceyi geçmiş?",
];
const STATUS_TEXT = { planning: "Soru analiz ediliyor…", writing: "Cevap yazılıyor…", checking: "Sayılar doğrulanıyor…" };

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [info, setInfo] = useState(null);
  const bottom = useRef(null);

  useEffect(() => { health().then(setInfo).catch(() => setInfo(false)); }, [busy]);
  useEffect(() => { bottom.current?.scrollIntoView({ block: "end" }); }, [messages, status]);

  const patchLast = (fn) => setMessages((all) => all.map((m, i) => (i === all.length - 1 ? fn(m) : m)));

  async function ask(text) {
    const question = text.trim();
    if (!question || busy) return;
    const history = messages.map((m) => ({ role: m.role, content: m.text }));
    setMessages((all) => [...all, { role: "user", text: question }, { role: "assistant", text: "" }]);
    setInput("");
    setBusy(true);
    setStatus(STATUS_TEXT.planning);
    try {
      await streamChat(question, history, (ev) => {
        if (ev.type === "status") setStatus(STATUS_TEXT[ev.text] || "");
        else if (ev.type === "tools") patchLast((m) => ({ ...m, tools: ev.calls }));
        else if (ev.type === "chart") patchLast((m) => ({ ...m, chart: ev.chart }));
        else if (ev.type === "token") patchLast((m) => ({ ...m, text: m.text + ev.text }));
        else if (ev.type === "replace") patchLast((m) => ({ ...m, text: ev.text }));
        else if (ev.type === "reset") patchLast((m) => ({ ...m, text: "" }));
        else if (ev.type === "done") patchLast((m) => ({ ...m, id: ev.message_id, meta: ev.meta }));
        else if (ev.type === "error") patchLast((m) => ({ ...m, error: true, text: `Hata: ${ev.text}. Ollama çalışıyor mu?` }));
      });
    } catch {
      patchLast((m) => ({ ...m, error: true, text: "Sunucuya ulaşılamadı. Backend çalışıyor mu? (uvicorn main:app --port 8000)" }));
    } finally {
      setBusy(false);
      setStatus("");
    }
  }

  return (
    <div className="app">
      <aside>
        <h1>Türkiye Hava Asistanı</h1>
        <p className="sub">81 il (son 4 yıl) ve Kütahya (1940'tan bugüne) verisini analiz eder.</p>
        <h2>Deneyin</h2>
        <ul>
          {EXAMPLES.map((q) => (
            <li key={q}><button disabled={busy} onClick={() => ask(q)}>{q}</button></li>
          ))}
        </ul>
        <p className="health" role="status">
          {info === null ? "Bağlanıyor…" : info === false ? "Sunucu kapalı"
            : `${info.ollama ? "Qwen bağlı" : "Ollama kapalı"} · ${info.model}`}
          {info && <><br />{info.likes} beğeni · {info.dislikes} beğenmeme · {info.corrections} düzeltme</>}
        </p>
        <p className="source">Veri: ERA5 yeniden analizi (Open-Meteo). Resmi istasyon ölçümü değildir.</p>
      </aside>

      <main>
        <div className="stream" aria-live="polite">
          {messages.length === 0 && (
            <div className="empty"><p>Hava verisi hakkında bir şey sorun.</p><p className="hint">Soldan örnek seçebilir ya da yazabilirsiniz.</p></div>
          )}
          {messages.map((m, i) => (
            <article key={i} className={`msg ${m.role}${m.error ? " error" : ""}`}>
              {m.role === "assistant" && !m.text && busy && i === messages.length - 1 ? <p className="wait">{status}</p> : <p>{m.text}</p>}
              {m.chart && <Chart chart={m.chart} />}
              {m.meta && !m.meta.grounded && (
                <p className="warn">⚠️ Bu cevaptaki bazı ifadeler araç sonuçlarıyla doğrulanamadı; sayılara temkinli yaklaşın.</p>
              )}
              {m.tools?.length > 0 && (
                <details className="tools">
                  <summary>Hangi analiz çalıştı?</summary>
                  {m.tools.map((t, k) => (
                    <code key={k}>{t.name}({Object.entries(t.args).map(([a, b]) => `${a}=${b}`).join(", ")})</code>
                  ))}
                </details>
              )}
              {m.id && <Feedback messageId={m.id} />}
            </article>
          ))}
          <div ref={bottom} />
        </div>

        <form className="composer" onSubmit={(e) => { e.preventDefault(); ask(input); }}>
          <label htmlFor="q" className="sr">Sorunuz</label>
          <input id="q" value={input} onChange={(e) => setInput(e.target.value)} autoComplete="off"
                 placeholder="Örn. Geçen ay Konya'da hava nasıldı?" />
          <button type="submit" disabled={busy || !input.trim()}>Sor</button>
        </form>
      </main>
    </div>
  );
}
