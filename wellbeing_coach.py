"""WellbeingIQ GenAI wellness coach.

Turns the model's output (predicted category + SHAP factors) into a short, cautious message with general
habit suggestions. Uses the Gemini API when a key is available and falls back to safe built-in wording
if there is no key, the call fails or times out, or the AI answer breaks the safety checks.

The AI never makes the prediction. It only rephrases facts produced by the model and SHAP.
Only habit values and factor names are sent to the API (no names, no identifiers, no free text).
"""
import os, re, json, hashlib, concurrent.futures as cf
import pandas as pd

# Model names change over time: override with the GEMINI_MODEL environment variable if needed.
MODELS = list(dict.fromkeys([m for m in (os.getenv("GEMINI_MODEL"), "gemini-3.8-flash", "gemini-3.7-flash") if m]))

DISCLAIMER = ("This is a general wellness tool trained on synthetic data. It is not a medical tool and cannot diagnose "
              "anything. If you feel overwhelmed or unsafe, please talk to a counsellor, a trusted person or a local helpline.")

SYSTEM_RULES = """You are a supportive student wellness assistant inside a prototype app.
Rules you must follow:
- Use ONLY the facts given in the user message. Do not invent facts, numbers, statistics, percentages or probabilities.
- Give general habit suggestions only (sleep routine, screen time, activity, study planning, reaching out to people).
- Never diagnose, never mention medical or mental-health conditions, medication, therapy or treatment.
- Be warm, calm and non-judgemental. Never alarm or blame the student. Do not say the student 'is' at risk of a condition.
- If the prediction is marked uncertain, say plainly that this is a rough estimate.
- Make each tip concrete and tailored to the student's own numbers; choose the most useful ones from the changes listed.
- Never use the percent sign. If you mention an estimate, say 'points'.
- Reply with JSON only, in this exact shape:
  {"summary": "<2 to 3 short sentences>", "suggestions": ["<tip 1>", "<tip 2>", "<tip 3>"]}
- The summary must stay under 70 words and each tip under 30 words."""

# safe, general tips per factor (used by the built-in fallback and as a safety net)
TIPS = {
 "Late_Night_Usage_Hours": "Try keeping the last hour before bed screen-free and move phone time earlier in the evening.",
 "Avg_Daily_Usage_Hours": "Set a gentle daily limit for social media and use app timers to notice when you pass it.",
 "Sleep_Hours_Per_Night": "Aim for a regular sleep and wake time so you get closer to 7 to 8 hours most nights.",
 "Sleep_Quality": "A dark, cool, quiet room and a short wind-down routine can help you sleep better.",
 "Physical_Activity_Hours": "Add a short daily walk or stretch break; small, regular movement counts.",
 "Social_Support_Score": "Reach out to a friend, family member or mentor; talking things through helps.",
 "Academic_Pressure": "Break big tasks into small steps and talk to a teacher or mentor if the workload feels heavy.",
 "Usage_Sleep_Combination": "Heavy phone use together with short sleep stacks up, so protecting your sleep is a good first step.",
 "Conflicts_Over_Social_Media": "Agree on phone-free times with family and friends to reduce arguments.",
 "Daily_Unlocks": "Turn off non-essential notifications to cut the urge to keep checking your phone.",
 "Notifications_Per_Day": "Mute non-essential notifications, especially in the evening.",
 "Days_Offline_Per_Month": "Plan a few phone-light days each month and do something offline you enjoy.",
 "Caffeine_Cups_Per_Day": "Keep caffeine to the morning so it does not get in the way of sleep.",
 "Total_Screen_Time_Hours": "Take regular screen breaks and look away from the screen every so often.",
 "Gaming_Hours": "Keep gaming to set times so it does not push into sleep or study time.",
 "Study_Hours": "Plan short, focused study blocks with breaks instead of long late sessions.",
 "Academic_Performance_Pct": "Set small weekly study goals and ask for help early if a subject feels hard.",
}
NOT_ACTIONABLE = {"Age", "Gender", "Country", "Academic_Level", "Residence", "Most_Used_Platform", "Purpose_Of_Use"}   # never presented as 'drivers'
MIN_IMPACT = 0.05   # ignore factors with a negligible contribution
UNSAFE = re.compile(r"\b(diagnos\w*|depress\w*|disorder\w*|medicat\w*|prescri\w*|therap\w*|suicid\w*|self[- ]?harm\w*|cure\w*|treatment\w*|illness\w*|clinical\w*|psychiatr\w*)\b|%", re.I)


ACTIONS = [
 ("Sleep_Hours_Per_Night", 1.0, 3.5, 10, "😴", "Sleep about 1 hour more", "You sleep {v:.1f} h a night. Try going to bed 30 to 60 minutes earlier and keep the same wake-up time."),
 ("Late_Night_Usage_Hours", -0.5, 0, 4, "🌙", "Trim late-night phone time by 30 min", "You use your phone {v:.1f} h late at night. Charge it away from your bed and set a wind-down alarm."),
 ("Avg_Daily_Usage_Hours", -1.0, 0.5, 12, "📱", "Cut social media by 1 hour", "You spend {v:.1f} h a day on social media. Use app timers and start by trimming your most-used app."),
 ("Physical_Activity_Hours", 0.5, 0, 4, "🏃", "Move 30 minutes more", "You move {v:.1f} h a day. Add a 20 to 30 minute walk after classes or a short stretch break."),
 ("Days_Offline_Per_Month", 2, 0, 20, "🌿", "Take 2 more phone-free days a month", "You have {v:.0f} phone-free days a month. Plan two, like a hike, a game with friends or a hobby day."),
 ("Social_Support_Score", 1, 1, 10, "🤝", "Lean on your support network", "Your support feels like {v:.0f} out of 10. Message a friend or family member this week and plan something together."),
 ("Academic_Pressure", -1, 1, 10, "📚", "Lighten the academic load", "Your pressure is {v:.0f} out of 10. Break big tasks into small steps and talk to a mentor if it feels heavy."),
 ("Sleep_Quality", 1, 1, 5, "🛏️", "Improve sleep quality", "You rate your sleep quality {v:.0f} out of 5. A dark, cool room and a 15-minute screen-free wind-down can help."),
 ("Caffeine_Cups_Per_Day", -1, 0, 8, "☕", "Cut one caffeine drink", "You have {v:.0f} caffeine drinks a day. Keep them to the morning so they do not cut into your sleep."),
 ("Conflicts_Over_Social_Media", -1, 0, 5, "💬", "Agree on phone-free times", "You rate conflicts over phone use {v:.0f} out of 5. Agree on phone-free meals or study blocks with family and friends."),
]
EMOJI = {a[0]: a[4] for a in ACTIONS}
EMOJI.update({"Usage_Sleep_Combination": "⚡", "Daily_Unlocks": "🔓", "Notifications_Per_Day": "🔔", "Study_Hours": "📖", "Gaming_Hours": "🎮",
              "Total_Screen_Time_Hours": "🖥️", "Age": "🎂", "Gender": "🙂", "Country": "🌍", "Academic_Level": "🎓", "Residence": "🏠",
              "Most_Used_Platform": "📲", "Purpose_Of_Use": "🎯", "Academic_Performance_Pct": "📝"})
SCORE = [("Avg_Daily_Usage_Hours", "📱", "Social media", "h", lambda v: "good" if v <= 4 else "watch" if v <= 6 else "high"),
         ("Late_Night_Usage_Hours", "🌙", "Late-night use", "h", lambda v: "good" if v <= 0.5 else "watch" if v <= 1.5 else "high"),
         ("Sleep_Hours_Per_Night", "😴", "Sleep", "h", lambda v: "good" if v >= 7 else "watch" if v >= 6 else "high"),
         ("Physical_Activity_Hours", "🏃", "Activity", "h", lambda v: "good" if v >= 1 else "watch" if v >= 0.5 else "high"),
         ("Social_Support_Score", "🤝", "Support", "/10", lambda v: "good" if v >= 6 else "watch" if v >= 4 else "high"),
         ("Academic_Pressure", "📚", "Pressure", "/10", lambda v: "good" if v <= 5 else "watch" if v <= 7 else "high"),
         ("Caffeine_Cups_Per_Day", "☕", "Caffeine", " cups", lambda v: "good" if v <= 2 else "watch" if v <= 4 else "high")]

def scorecard(row):
    return [{"emoji": e, "label": l, "value": f"{float(row[f]):.1f}{u}".replace(".0" + u, u) if u != "h" else f"{float(row[f]):.1f} h", "status": fn(float(row[f]))} for f, e, l, u, fn in SCORE]

def bump(row, f, d, lo, hi):
    new = dict(row); v = float(new[f]); nv = min(hi, max(lo, v + d))
    if abs(nv - v) < 1e-9: return None
    new[f] = nv; u = float(new["Avg_Daily_Usage_Hours"])
    new["Late_Night_Usage_Hours"] = min(float(new["Late_Night_Usage_Hours"]), u, 4.0)
    new["Total_Screen_Time_Hours"] = max(float(new["Total_Screen_Time_Hours"]), u + 0.5)
    return new

def meter(ex, row):
    """wellness meter 0 to 100 from the model's class probabilities, plus the predicted category"""
    pr = dict(zip(ex.classes, ex.pipe.predict_proba(pd.DataFrame([row])[ex.features])[0]))
    return int(round(100 * (pr["Good"] + 0.5 * pr["Moderate"]))), max(pr, key=pr.get)

def action_plan(ex, row, n=4):
    """Try realistic small changes, ask the model how the wellness meter would move, keep the biggest helpers."""
    m0, _ = meter(ex, row); out = []
    for f, d, lo, hi, em, title, how in ACTIONS:
        new = bump(row, f, d, lo, hi)
        if new is None: continue
        pts = meter(ex, new)[0] - m0
        if pts >= 1: out.append({"feature": f, "emoji": em, "title": title, "how": how.format(v=float(row[f])), "points": pts, "change": (f, d, lo, hi)})
    return sorted(out, key=lambda x: -x["points"])[:n]

def combo(ex, row, plan, k=3):
    """what happens if the top-k plan changes are made together"""
    new = dict(row)
    for p in plan[:k]: new = bump(new, *p["change"]) or new
    (m0, c0), (m1, c1) = meter(ex, row), meter(ex, new)
    return {"from_meter": m0, "to_meter": m1, "from_cat": c0, "to_cat": c1, "n": min(k, len(plan))}

def _words(s): return len(re.findall(r"\w+", s))

def _facts(expl, plan=()):
    fmt = lambda f, d: f"- {f['label']}: {f['value']} ({d})"
    lines = [f"Predicted wellness category: {expl['predicted_category']}",
             f"Prediction reliability: {'UNCERTAIN, say this is a rough estimate' if expl['uncertain'] else 'reasonably confident'}",
             "Habits pushing toward lower wellbeing:"] + [fmt(f, "raises risk") for f in expl["risk_drivers"]] \
          + ["Habits helping wellbeing:"] + [fmt(f, "protective") for f in expl["protective_factors"]]
    if plan:
        lines += ["Possible changes, with the model's estimated gain on the wellness meter (0 to 100):"]
        lines += [f"- {p['title']} (student's situation: {p['how'].split('. ')[0]}): about {p['points']} points higher on the wellness meter" for p in plan]
    return "\n".join(lines)

def validate(raw):
    """Parse and check the AI answer. Returns (summary, suggestions) or raises ValueError."""
    t = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    d = json.loads(t)
    s, tips = d["summary"], d["suggestions"]
    if not isinstance(s, str) or not isinstance(tips, list) or not (1 <= len(tips) <= 3): raise ValueError("bad shape")
    tips = [str(x) for x in tips]
    if not 8 <= _words(s) <= 90 or any(_words(x) > 40 or _words(x) < 3 for x in tips): raise ValueError("bad length")
    if UNSAFE.search(s) or any(UNSAFE.search(x) for x in tips): raise ValueError("unsafe wording")
    return s.strip(), [x.strip() for x in tips]

def template_advice(expl, plan=()):
    cat, drivers, prot = expl["predicted_category"], expl["risk_drivers"], expl["protective_factors"]
    names = lambda fs: ", ".join(f["label"].lower() for f in fs[:3])
    top = plan[0] if plan else None
    if cat == "Good":
        s = f"Your habits look healthy. {names(prot).capitalize()} are working in your favour." if prot else "Your habits look healthy."
    elif cat == "Moderate":
        s = f"Your habits look mixed. The biggest pulls toward lower wellbeing are {names(drivers)}." if drivers else "Your habits look mixed."
    else:
        s = f"Your current habits suggest your wellbeing could use some care, mostly because of {names(drivers)}." if drivers else "Your wellbeing could use some care."
    if top and cat != "Good": s += f" The good news: {top['title'].lower()} is estimated to help the most."
    if expl["uncertain"]: s += " This is only a rough estimate, so treat it as a starting point."
    tips = [p["how"] for p in plan][:3]
    if not tips: tips = ["Keep up your current routine and check in with yourself from time to time."]
    if cat != "Good" and len(tips) < 3: tips.append("Talk to a friend, mentor or your college counsellor if things feel heavy.")
    return s, tips[:3]

class WellnessCoach:
    def __init__(self, explainer, api_key=None, generate_fn=None, timeout=40):
        self.ex, self.timeout, self._cache = explainer, timeout, {}
        self.key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.generate_fn = generate_fn        # test hook: fn(prompt, system, model) -> str

    @property
    def ai_available(self): return bool(self.key or self.generate_fn)

    def _generate(self, prompt, model):
        if self.generate_fn: return self.generate_fn(prompt, SYSTEM_RULES, model)
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=self.key)
        r = client.models.generate_content(model=model, contents=prompt, config=types.GenerateContentConfig(
            system_instruction=SYSTEM_RULES, temperature=0.4, max_output_tokens=1500, response_mime_type="application/json"))
        return r.text

    def _redact(self, t):
        return t.replace(self.key, "***") if self.key else t

    def _ask_ai(self, expl, plan=()):
        prompt = "Student facts (from the prediction model):\n" + _facts(expl, plan) + "\n\nWrite the JSON now."
        err = "no model tried"
        for m in MODELS:
            ex = cf.ThreadPoolExecutor(max_workers=1)
            try:
                raw = ex.submit(self._generate, prompt, m).result(timeout=self.timeout)
                s, tips = validate(raw); return s, tips, m, None
            except cf.TimeoutError: err = f"{m}: timed out after {self.timeout}s"
            except Exception as e: err = f"{m}: {type(e).__name__}: {self._redact(str(e))[:200]}"
            finally: ex.shutdown(wait=False)
        return None, None, None, err

    def test_connection(self):
        """Tiny call to check the key and model. Never returns the key."""
        if not self.ai_available: return False, "No API key found. Paste one in the sidebar or set GEMINI_API_KEY."
        notes = []
        for m in MODELS:
            ex = cf.ThreadPoolExecutor(max_workers=1)
            try:
                ex.submit(self._generate, "Reply with the single word OK.", m).result(timeout=self.timeout)
                return True, f"Connected using {m}."
            except cf.TimeoutError: notes.append(f"{m}: timed out")
            except Exception as e: notes.append(f"{m}: {type(e).__name__}: {self._redact(str(e))[:200]}")
            finally: ex.shutdown(wait=False)
        return False, " | ".join(notes)

    def advise(self, row, use_ai=True, top_k=3):
        expl = self.ex.explain(row, top_k=10)
        keep = lambda fs, ok: [f for f in fs if f["feature"] not in NOT_ACTIONABLE and ok(f["impact"])][:top_k]
        expl["risk_drivers"] = keep(expl["risk_drivers"], lambda i: i >= MIN_IMPACT)
        expl["protective_factors"] = keep(expl["protective_factors"], lambda i: i <= -MIN_IMPACT)
        plan = action_plan(self.ex, row)
        key = hashlib.md5(json.dumps([expl, [(p["feature"], p["points"]) for p in plan], use_ai, self.ai_available], sort_keys=True, default=str).encode()).hexdigest()
        if key in self._cache: return self._cache[key]
        out = {"explanation": expl, "plan": plan, "combo": combo(self.ex, row, plan) if plan else None, "scorecard": scorecard(row), "disclaimer": DISCLAIMER, "source": "built-in guidance", "model": None, "ai_error": None}
        if use_ai and self.ai_available:
            s, tips, m, err = self._ask_ai(expl, plan)
            if s: out.update(summary=s, suggestions=tips, source="Gemini AI", model=m)
            else: out["ai_error"] = err
        if "summary" not in out: out["summary"], out["suggestions"] = template_advice(expl, plan)
        self._cache[key] = out; return out

if __name__ == "__main__":
    import sys, pandas as pd
    from wellbeing_explain import WellbeingExplainer
    ex = WellbeingExplainer(); coach = WellnessCoach(ex); X = pd.read_csv("X_test.csv")
    print("Gemini key found:", coach.key is not None, "| models tried in order:", MODELS)
    for i in [0, 1, 2]:
        r = coach.advise(X.iloc[i].to_dict())
        print(f"\n[{r['source']}] predicted {r['explanation']['predicted_category']} | uncertain={r['explanation']['uncertain']}")
        print(r["summary"]); [print(" -", t) for t in r["suggestions"]]
        if r["ai_error"]: print("AI problem:", r["ai_error"])
