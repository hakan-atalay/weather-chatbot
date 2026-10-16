import { useState } from "react";
import { sendFeedback } from "./api.js";

const REASONS = [
  ["wrong_number", "Sayı yanlış"],
  ["wrong_scope", "Yanlış il / dönem"],
  ["wrong_analysis", "Yanlış analiz"],
  ["too_long_or_unclear", "Uzun veya anlaşılmaz"],
  ["off_topic", "Konu dışı"],
  ["other", "Diğer"],
];

export default function Feedback({ messageId }) {
  const [rating, setRating] = useState(0);
  const [reason, setReason] = useState("");
  const [correction, setCorrection] = useState("");
  const [state, setState] = useState("idle");

  async function submit(r, extra = {}) {
    setState("saving");
    try {
      await sendFeedback({ message_id: messageId, rating: r, ...extra });
      setRating(r);
      setState("saved");
    } catch {
      setState("error");
    }
  }

  return (
    <div className="feedback">
      <div className="feedback-buttons">
        <button aria-label="Cevabı beğen" aria-pressed={rating === 1} className={rating === 1 ? "on" : ""}
                disabled={state === "saving"} onClick={() => submit(1)}>👍</button>
        <button aria-label="Cevabı beğenme" aria-pressed={rating === -1} className={rating === -1 ? "on" : ""}
                disabled={state === "saving"} onClick={() => submit(-1)}>👎</button>
        {rating === 1 && <span className="note">Teşekkürler, örnek olarak kaydedildi.</span>}
        {state === "error" && <span className="note err">Kaydedilemedi, tekrar deneyin.</span>}
      </div>

      {rating === -1 && (
        <form className="why" onSubmit={(e) => { e.preventDefault(); submit(-1, { reason: reason || null, correction }); }}>
          <p>Neden beğenmediniz? (isteğe bağlı)</p>
          <div className="chips" role="group" aria-label="Neden">
            {REASONS.map(([value, label]) => (
              <button type="button" key={value} className={reason === value ? "chip on" : "chip"}
                      aria-pressed={reason === value} onClick={() => setReason(value)}>{label}</button>
            ))}
          </div>
          <label htmlFor={`fix-${messageId}`}>Doğrusu ne olmalıydı?</label>
          <textarea id={`fix-${messageId}`} rows={3} value={correction} onChange={(e) => setCorrection(e.target.value)} />
          <button type="submit" className="send" disabled={state === "saving" || (!reason && !correction.trim())}>Gönder</button>
          {state === "saved" && <span className="note">Geri bildiriminiz kaydedildi.</span>}
        </form>
      )}
    </div>
  );
}
