"""Does the model pick the right tool with the right arguments?   Run: python eval_tools.py [--plan-only] [--hard]

--hard      : harder, held-out questions that do NOT resemble the few-shot examples (the honest test).

--plan-only : only runs the planning stage (fast, ~5-15 s per question). Without it, runs full chat.
Compare models:   set QWEN_MODEL=qwen3:8b   then run again.   Compare modes:  set TOOL_MODE=native
"""
import datetime as dt
import sys
import time

import data
import llm

TOMORROW = (dt.date.today() + dt.timedelta(days=1)).isoformat()
# (question, expected tool or None, expected args subset). A list value means "any of these".
EASY_CASES = [
    ("Ağustos 2025'te en sıcak 5 il hangisi?", "rank_provinces", {"metric": ["max_temp", "mean_temp"], "year": 2025, "month": 8, "order": "highest"}),
    ("Kütahya Eskişehir'den serin mi?", "compare_provinces", {"province_a|province_b": ["kutahya", "eskisehir"]}),
    ("Kütahya ısınıyor mu?", "long_term_trend", {"metric": ["mean_temp", "max_temp", "min_temp"]}),
    ("Ankara ısınıyor mu?", None, {}),
    ("Yarın Kütahya'da yağmur olacak mı?", "live_forecast", {"province": "kutahya", "target_date": TOMORROW}),
    ("Kütahya'da fide ne zaman dikilir?", "frost_calendar", {}),
    ("2035'te Kütahya'da sıcaklık ne olur?", "project_trend", {"metric": ["mean_temp", "max_temp"], "target_year": 2035}),
    ("Kütahya'da kaç gün 35 dereceyi geçmiş?", "count_threshold_days", {"province": "kutahya", "metric": "max_temp", "threshold": 35, "direction": "above"}),
    ("Ocak ayında Kütahya'da ortalama sıcaklık kaç derece?", "get_summary", {"province": "kutahya", "month": 1}),
    ("Türkiye'de en yağışlı 3 il hangisi?", "rank_provinces", {"metric": "precip", "order": "highest", "top_n": 3}),
    ("Van'a iklimce en çok benzeyen iller hangileri?", "similar_climates", {"province": "van"}),
    ("Hatay'da kaydedilen en yüksek sıcaklık ne zaman?", "find_extremes", {"province": "hatay", "metric": "max_temp", "order": "highest"}),
    ("Temmuzda Kütahya'da normalde hava nasıl olur?", "seasonal_outlook", {"province": "kutahya", "month": 7}),
    ("2025-07-20 Kütahya için normal bir gün müydü?", "anomaly_check", {"province": "kutahya", "date": "2025-07-20"}),
    ("Ankara ile İzmir'i 2025 Temmuz'unda karşılaştır", "compare_provinces", {"province_a|province_b": ["ankara", "izmir"], "year": 2025, "month": 7}),
    ("Merhaba, nasılsın?", None, {}),
]

_lm = dt.date.today().replace(day=1) - dt.timedelta(days=1)      # last month, computed from today
# Held-out questions: NOT similar to the few-shot examples in the planner prompt (--hard)
HARD_CASES = [
    ("Kütahya'da 2024 yazı nasıldı?", "get_summary", {"province": "kutahya", "year": 2024, "season": "summer"}),
    ("Geçen ay Konya'da hava nasıldı?", "get_summary", {"province": "konya", "year": _lm.year, "month": _lm.month}),
    ("Antalya'da 2024'te kaç gün 40 derece üstü oldu?", "count_threshold_days", {"province": "antalya", "metric": "max_temp", "threshold": 40, "direction": "above", "year": 2024}),
    ("Erzurum mu Van mı daha soğuk?", "compare_provinces", {"province_a|province_b": ["erzurum", "van"]}),
    ("Türkiye'nin en kurak ili hangisi?", "rank_provinces", {"metric": "precip", "order": "lowest"}),
    ("Konya'ya benzer iklime sahip iller hangileri?", "similar_climates", {"province": "konya"}),
    ("Kütahya'da yağışlar azalıyor mu?", "long_term_trend", {"metric": "precip"}),
    ("Kütahya'da son bahar donu genelde hangi tarihte biter?", "frost_calendar", {}),
    ("Önümüzdeki hafta İstanbul'da hava nasıl olacak?", "live_forecast", {"province": "istanbul"}),
    ("Ocak ayında Kars'ta tipik sıcaklık nedir?", "seasonal_outlook", {"province": "kars", "month": 1}),
    ("Kütahya'da yaz sıcaklıkları artıyor mu?", "long_term_trend", {"metric": ["max_temp", "mean_temp"]}),
    ("Kütahya 2010 yılında nasıldı?", "get_summary", {"province": "kutahya", "year": 2010}),
    ("2050'de Kütahya'nın yıllık yağışı ne olur?", "project_trend", {"metric": "precip", "target_year": 2050}),
    ("Sivas'ta 2025'te hiç 38 dereceyi gördü mü?", "count_threshold_days", {"province": "sivas", "metric": "max_temp", "threshold": 38, "direction": "above", "year": 2025}),
    ("Python ile hello world yaz", None, {}),
]
CASES = HARD_CASES if "--hard" in sys.argv else EASY_CASES


def same(want, got):
    if isinstance(want, list):
        return any(same(w, got) for w in want)
    if isinstance(want, str):
        return data.fold(got) == data.fold(want) if isinstance(got, str) else False
    return got == want


def args_ok(expected, got):
    for key, want in expected.items():
        if "|" in key:
            vals = {data.fold(got.get(k, "")) for k in key.split("|")}
            if vals != set(want):
                return False
        elif key not in got or not same(want, got[key]):
            return False
    return True


def main():
    plan_only = "--plan-only" in sys.argv
    if not llm.ready():
        raise SystemExit("Ollama is not running.")
    print(f"model={llm.MODEL} mode={llm.TOOL_MODE} plan_only={plan_only}\n")
    ok_tool = ok_args = 0
    total_time = 0
    for q, tool, args in CASES:
        t0 = time.time()
        if plan_only:
            calls = [{"name": c["tool"], "args": c.get("args", {})} for c in llm.plan(q)]
            extra = ""
        else:
            out = llm.chat(q)
            calls, extra = out["tool_calls"], ("" if out["grounded"] else f"  [ungrounded numbers {out['ungrounded_numbers']}]")
        dt_ = time.time() - t0
        total_time += dt_
        if tool is None:
            good_tool = not any(c["name"] == "long_term_trend" for c in calls) if "ısın" in q else not calls
            good_args = good_tool
        else:
            hit = next((c for c in calls if c["name"] == tool), None)
            good_tool, good_args = hit is not None, hit is not None and args_ok(args, hit["args"])
        ok_tool += good_tool
        ok_args += good_args
        print(f"{'PASS' if good_args else ('TOOL-OK/ARGS-WRONG' if good_tool else 'FAIL')}  {q}  ({dt_:.0f}s)")
        if not good_args:
            print(f"      expected: {tool} {args}\n      got:      {calls}")
        if extra:
            print(extra)
    n = len(CASES)
    print(f"\nTool choice: {ok_tool}/{n} | Tool + arguments: {ok_args}/{n} | avg {total_time / n:.0f}s per question")


if __name__ == "__main__":
    main()