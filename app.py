import json, time, re, html, datetime, numpy as np, pandas as pd, streamlit as st
from wellbeing_explain import WellbeingExplainer
from wellbeing_coach import WellnessCoach, bump, EMOJI

st.set_page_config(page_title="WellbeingIQ", page_icon="🧠", layout="wide")

@st.cache_resource
def load():
    ex = WellbeingExplainer(); X = pd.read_csv("X_train.csv")[ex.features]
    return ex, X, json.load(open("model_card.json"))
ex, Xtr, card = load()
med = Xtr.median(numeric_only=True); mode = Xtr.mode().iloc[0]
opts = {c: sorted(Xtr[c].dropna().unique()) for c in Xtr.select_dtypes(exclude=np.number)}
COL = {"Good": "#2e9e5b", "Moderate": "#f0a030", "At-Risk": "#e5533d"}
FACE = {"Good": ("😊", "You're in a good place"), "Moderate": ("🙂", "A few habits to tune"), "At-Risk": ("🌧️", "Your habits need some care")}
DOT = {"good": "🟢", "watch": "🟡", "high": "🔴"}

st.markdown("""<style>
@keyframes fadeUp{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:none}}
@keyframes grow{from{width:0}to{width:var(--w)}}
@keyframes floaty{0%,100%{transform:translateY(0)}50%{transform:translateY(-12px)}}
@keyframes glow{0%,100%{box-shadow:0 0 0 0 transparent}50%{box-shadow:0 0 26px 4px var(--c)}}
@keyframes shimmer{from{background-position:0% 50%}to{background-position:300% 50%}}
@keyframes fill{to{stroke-dashoffset:var(--off)}}
.hero{position:relative;overflow:hidden;padding:26px 32px;border-radius:22px;color:#fff;background:linear-gradient(120deg,#5b3cc4,#2a8bd6,#2fbf9b,#5b3cc4);background-size:300% 100%;animation:shimmer 12s linear infinite,fadeUp .8s ease-out}
.hero h1{margin:0;font-size:44px;color:#fff;padding:0}.hero p{margin:6px 0 0;font-size:17px;opacity:.92}
.fl{position:absolute;font-size:34px;animation:floaty 4s ease-in-out infinite;opacity:.9}
.card{background:rgba(128,128,128,.12);border:1px solid rgba(128,128,128,.28);border-radius:16px;padding:16px 18px;margin:8px 0;animation:fadeUp .7s ease-out both}
.cat{border:2px solid var(--c);border-radius:20px;padding:18px;text-align:center;animation:fadeUp .6s ease-out both,glow 2.8s ease-in-out infinite}
.cat .big{font-size:56px;line-height:1.1;animation:floaty 3s ease-in-out infinite;display:inline-block}.cat h2{margin:4px 0;color:var(--c);padding:0}
.pb{display:flex;align-items:center;gap:8px;margin:6px 0;font-size:15px}.pb .n{width:110px}.pb .v{width:44px;text-align:right}
.tr{flex:1;height:12px;border-radius:99px;background:rgba(128,128,128,.25);overflow:hidden}
.br{height:100%;border-radius:99px;width:var(--w);animation:grow 1.1s ease-out both}.br.up{background:linear-gradient(90deg,#e5533d,#ff8a65)}.br.dn{background:linear-gradient(90deg,#2e9e5b,#7be0a3)}
.why{display:flex;align-items:center;gap:10px;margin:7px 0;animation:fadeUp .6s ease-out both}.why .wl{width:48%;font-size:15px}.why .wd{width:92px;font-size:13px;opacity:.85}
.chip{display:inline-block;padding:6px 13px;border-radius:99px;margin:4px 4px 4px 0;font-size:14px;border:1px solid;animation:fadeUp .5s ease-out both}
.chip.good{background:rgba(46,158,91,.18);border-color:#2e9e5b}.chip.watch{background:rgba(240,160,48,.18);border-color:#f0a030}.chip.high{background:rgba(229,83,61,.18);border-color:#e5533d}
.plan{display:flex;gap:14px;flex-wrap:wrap}.pc{flex:1 1 240px;background:rgba(128,128,128,.12);border:1px solid rgba(128,128,128,.28);border-radius:16px;padding:16px;animation:fadeUp .7s ease-out both;transition:transform .25s}
.pc:hover{transform:translateY(-6px) scale(1.02)}.pc .e{font-size:34px;display:inline-block;animation:floaty 3.5s ease-in-out infinite}.pc h4{margin:6px 0;padding:0}.pc p{font-size:14px;margin:4px 0;opacity:.9}
.pill{display:inline-block;margin-top:6px;padding:4px 11px;border-radius:99px;font-size:13px;font-weight:600;background:rgba(46,158,91,.22);color:#59d68c}
.arc{stroke-dashoffset:283;animation:fill 1.8s ease-out forwards}.cmp{display:flex;gap:16px;align-items:stretch;flex-wrap:wrap}.cmp .card{flex:1;text-align:center}
[data-testid="stSidebar"]{background:linear-gradient(160deg,#1b1546,#1f3b73,#0f6b6f,#1b1546);background-size:260% 260%;animation:sbmove 16s ease-in-out infinite;border-right:1px solid rgba(255,255,255,.15)}
[data-testid="stSidebar"]>div:first-child{background:radial-gradient(circle at 15% 0%,rgba(255,255,255,.16),transparent 42%),radial-gradient(circle at 95% 95%,rgba(47,191,155,.28),transparent 45%)}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3,[data-testid="stSidebar"] label,[data-testid="stSidebar"] p,[data-testid="stSidebar"] span{color:#f2f5ff}
[data-testid="stSidebar"] .stButton>button{background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.35);color:#fff;transition:transform .2s,background .2s}
[data-testid="stSidebar"] .stButton>button:hover{background:rgba(255,255,255,.25);transform:translateY(-2px)}
[data-testid="stSidebar"] input{background:rgba(10,12,35,.55)!important;color:#fff!important}
button[kind="primary"]{background:linear-gradient(90deg,#7b5cff,#2fbf9b)!important;border:0!important;color:#fff!important;font-weight:700}
[data-testid="stHeaderActionElements"]{display:none}
@keyframes sbmove{0%,100%{background-position:0% 0%}50%{background-position:100% 100%}}
.av{width:56px;height:56px;flex:0 0 56px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:26px;font-weight:700;color:#fff;background:linear-gradient(135deg,#7b5cff,#2fbf9b);--c:#7b5cff;animation:glow 3s ease-in-out infinite}
.rep{display:flex;align-items:center;gap:16px}.greet{font-size:18px;line-height:1.55;border-left:5px solid #7b5cff}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="hero"><span class="fl" style="right:8%;top:10px">🌙</span><span class="fl" style="right:17%;top:46px;animation-delay:.6s">😴</span><span class="fl" style="right:26%;top:8px;animation-delay:1.2s">📱</span><span class="fl" style="right:34%;top:52px;animation-delay:1.8s">🏃</span><h1>🧠 WellbeingIQ</h1><p>Your habits, explained. Smart guidance for student wellbeing ✨</p></div>', unsafe_allow_html=True)

# ---------------- sidebar ----------------
DEF = dict(usage=round(float(med["Avg_Daily_Usage_Hours"]), 1), late=round(float(med["Late_Night_Usage_Hours"]), 1), sleep=round(float(med["Sleep_Hours_Per_Night"]), 1),
           sq=int(med["Sleep_Quality"]), act=round(float(med["Physical_Activity_Hours"]), 1), sup=int(med["Social_Support_Score"]), press=int(med["Academic_Pressure"]),
           caf=int(med["Caffeine_Cups_Per_Day"]), off=int(med["Days_Offline_Per_Month"]))
PRESETS = {"🌿 Balanced": dict(usage=2.5, late=0.2, sleep=8.0, sq=4, act=1.5, sup=8, press=4, caf=1, off=6),
           "🦉 Night owl": dict(usage=6.5, late=2.5, sleep=5.0, sq=2, act=0.3, sup=5, press=6, caf=3, off=1),
           "📚 Busy": dict(usage=4.5, late=1.0, sleep=5.5, sq=2, act=0.5, sup=4, press=9, caf=4, off=2), "🔄 Reset": DEF}
for k, v in DEF.items(): st.session_state.setdefault(k, v)
def setp(p):
    for k, v in p.items(): st.session_state[k] = v
    st.session_state["result"] = None

with st.sidebar:
    st.markdown('<div style="font-size:13px;opacity:.85;padding:2px 0 8px">🧠 <b>WellbeingIQ</b> </div>', unsafe_allow_html=True)
    st.header("🎭 Quick demo profiles")
    pc = st.columns(2)
    for i, (pname, p) in enumerate(PRESETS.items()): pc[i % 2].button(pname, on_click=setp, args=(p,), use_container_width=True)
    st.header("🧑/👧 About you")
    name = st.text_input("✍️ Enter your name", max_chars=30, placeholder="e.g. Gulshan", help="Used only on this page to personalise your result. It is never sent to the AI.")
    st.header("🎛️ Your habits")
    usage = st.slider("📱 Social media hours per day", 0.5, 12.0, step=0.1, key="usage")
    late = st.slider("🌙 Late-night phone use (hours)", 0.0, 4.0, step=0.1, key="late")
    if late > usage: st.caption("ℹ️ Late-night use can't exceed total use, so it is capped.")
    sleep = st.slider("😴 Sleep per night (hours)", 3.5, 10.0, step=0.1, key="sleep")
    sq = st.slider("🛏️ Sleep quality (1 poor, 5 great)", 1, 5, key="sq")
    act = st.slider("🏃 Physical activity (hours per day)", 0.0, 4.0, step=0.1, key="act")
    sup = st.slider("🤝 Social support (1 low, 10 high)", 1, 10, key="sup")
    press = st.slider("📚 Academic pressure (1 low, 10 high)", 1, 10, key="press")
    caf = st.slider("☕ Caffeine drinks per day", 0, 8, key="caf")
    off = st.slider("🌿 Phone-free days per month", 0, 20, key="off")
    row = {"Avg_Daily_Usage_Hours": usage, "Late_Night_Usage_Hours": min(late, usage), "Sleep_Hours_Per_Night": sleep, "Sleep_Quality": sq, "Physical_Activity_Hours": act,
           "Social_Support_Score": sup, "Academic_Pressure": press, "Caffeine_Cups_Per_Day": caf, "Days_Offline_Per_Month": off}
    with st.expander("➕ More details (optional)"):
        row["Age"] = st.slider("🎂 Age", 17, 30, int(med["Age"]))
        for c, lab in [("Gender","🙂 Gender"),("Country","🌍 Country"),("Academic_Level","🎓 Academic level"),("Residence","🏠 Residence"),("Most_Used_Platform","📲 Most used platform"),("Purpose_Of_Use","🎯 Main purpose of use")]:
            row[c] = st.selectbox(lab, opts[c], index=opts[c].index(mode[c]))
        row["Study_Hours"] = st.slider("📖 Study hours per day", 0.0, 12.0, float(round(med["Study_Hours"], 1)), 0.1)
        row["Gaming_Hours"] = st.slider("🎮 Gaming hours per day", 0.0, 6.0, float(round(med["Gaming_Hours"], 1)), 0.1)
        tot = st.slider("🖥️ Total screen time per day (hours)", 0.5, 16.0, float(round(med["Total_Screen_Time_Hours"], 1)), 0.1)
        row["Total_Screen_Time_Hours"] = max(tot, usage + 0.5)
        row["Daily_Unlocks"] = st.slider("🔓 Phone unlocks per day", 10, 300, int(med["Daily_Unlocks"]))
        row["Notifications_Per_Day"] = st.slider("🔔 Notifications per day", 5, 500, int(med["Notifications_Per_Day"]))
        row["Academic_Performance_Pct"] = st.slider("📝 Academic performance (%)", 40, 100, int(med["Academic_Performance_Pct"]))
        row["Conflicts_Over_Social_Media"] = st.slider("💬 Conflicts over phone use (0 to 5)", 0, 5, int(med["Conflicts_Over_Social_Media"]))

    with st.expander("🤖 Gemini AI settings"):
        key = st.text_input("🔑 Gemini API key (optional)", type="password", help="Kept only in this browser session.")
        use_ai = st.checkbox("Use Gemini for the coach message", value=True)
        key_id = (key or "").strip()
        if st.session_state.get("coach_key") != key_id or "coach" not in st.session_state:
            st.session_state["coach"], st.session_state["coach_key"] = WellnessCoach(ex, api_key=key_id or None), key_id
        coach = st.session_state["coach"]
        if st.button("🔌 Test Gemini connection", use_container_width=True):
            with st.spinner("Calling Gemini..."): ok, msg = coach.test_connection()
            (st.success if ok else st.error)(("✅ " if ok else "❌ ") + msg)

# ---------------- html helpers ----------------
def gauge(idx):
    p = "M20 110 A90 90 0 0 1 200 110"
    return (f'<svg viewBox="0 0 220 128" width="100%" style="max-width:300px"><defs><linearGradient id="g" x1="0" x2="1"><stop offset="0" stop-color="#e5533d"/><stop offset=".5" stop-color="#f0a030"/><stop offset="1" stop-color="#2e9e5b"/></linearGradient></defs>'
            f'<path d="{p}" fill="none" stroke="rgba(128,128,128,.3)" stroke-width="16" stroke-linecap="round"/><path class="arc" d="{p}" fill="none" stroke="url(#g)" stroke-width="16" stroke-linecap="round" stroke-dasharray="283" style="--off:{283*(1-idx/100):.1f}"/>'
            f'<text x="110" y="98" text-anchor="middle" font-size="36" font-weight="700" fill="currentColor">{idx}</text><text x="110" y="120" text-anchor="middle" font-size="12" fill="currentColor" opacity=".8">wellness meter (0 to 100)</text></svg>')
def probbars(pr):
    return "".join(f'<div class="pb"><span class="n">{e} {k}</span><div class="tr"><div class="br" style="--w:{pr[k]*100:.0f}%;background:{COL[k]};animation-delay:{i*.15}s"></div></div><span class="v">{pr[k]:.0%}</span></div>' for i, (k, e) in enumerate([("At-Risk","🌧️"),("Moderate","🙂"),("Good","😊")]))
def whybars(items):
    mx = max(abs(i["impact"]) for i in items) or 1
    return "".join(f'<div class="why" style="animation-delay:{n*.12}s"><span class="wl">{EMOJI.get(i["feature"],"•")} {i["label"]} <i>({i["value"]})</i></span><div class="tr"><div class="br {"up" if i["impact"]>0 else "dn"}" style="--w:{abs(i["impact"])/mx*100:.0f}%"></div></div><span class="wd">{"🔺 raises risk" if i["impact"]>0 else "🟢 lowers risk"}</span></div>' for n, i in enumerate(items))
def chips(sc): return "".join(f'<span class="chip {s["status"]}" style="animation-delay:{n*.08}s">{DOT[s["status"]]} {s["emoji"]} {s["label"]}: {s["value"]}</span>' for n, s in enumerate(sc))
def plancards(plan): return '<div class="plan">' + "".join(f'<div class="pc" style="animation-delay:{n*.15}s"><span class="e">{p["emoji"]}</span><h4>{p["title"]}</h4><p>{p["how"]}</p><span class="pill">⬆️ about +{p["points"]} on your wellness meter</span></div>' for n, p in enumerate(plan)) + "</div>"
def typed(t):
    for w in t.split(" "): yield w + " "; time.sleep(0.025)
def first(n): return n.split()[0] if n.split() else ""
def greeting(cat, who, unsure, idx=None, e=None, plan=(), cb=None):
    f = html.escape(first(who)); hi = f", {f}" if f else ""
    d = e["risk_drivers"][0]["label"].lower() if e and e["risk_drivers"] else None
    h = e["protective_factors"][0]["label"].lower() if e and e["protective_factors"] else None
    base = {"Good": f"Great news{hi}! 🎉 Your habits are working well for you right now.",
            "Moderate": f"Thanks for checking in{hi} 🙂 You're doing several things right, and a few small tweaks could lift you further.",
            "At-Risk": f"Thanks for being open about your habits{hi} 💙 A few of them are weighing on your wellbeing right now, and the good news is that habits can change."}[cat]
    x = []
    if idx is not None: x.append(f"Your wellness meter reads <b>{idx} out of 100</b>.")
    if h: x.append(f"Working in your favour: {h}.")
    if d and cat != "Good": x.append(f"Pulling the other way: {d}.")
    if plan and cat != "Good": x.append(f"A good first step: {plan[0]['title'].lower()}.")
    if cb and cat != "Good" and cb["to_meter"] > cb["from_meter"] + 2: x.append(f"Doing your top {cb['n']} changes together could lift your meter from {cb['from_meter']} to {cb['to_meter']}.")
    return base + " " + " ".join(x) + (" This is a rough estimate, so please use it as a starting point." if unsure else "")
def report_text(res, who):
    e = res["explanation"]; pr = e["probabilities"]; idx = int(round(100 * (pr["Good"] + .5 * pr["Moderate"])))
    L = [f"WellbeingIQ wellness report{' for ' + who if who else ''}", datetime.date.today().strftime("%d %B %Y"), "(Prototype trained on synthetic data. Not a medical tool.)", "",
         f"Category: {e['predicted_category']} (model confidence {e['confidence']:.0%})", f"Wellness meter: {idx}/100", "", "Habits pulling wellbeing down:"] + [f"- {f['label']}: {f['value']}" for f in e["risk_drivers"]] \
        + ["", "Habits helping:"] + [f"- {f['label']}: {f['value']}" for f in e["protective_factors"]] + ["", "Coach note:", res["summary"]] + [f"- {t}" for t in res["suggestions"]] \
        + ["", "Action plan:"] + [f"- {p['title']}: {p['how']}" for p in res["plan"]]
    return "\n".join(L)
def probs_of(r): return dict(zip(ex.classes, ex.pipe.predict_proba(pd.DataFrame([r])[ex.features])[0]))

tab1, tab2, tab3 = st.tabs(["🧭 My wellness check", "🔮 What if I change a habit?", "ℹ️ About"])

# ---------------- tab 1 ----------------
with tab1:
    st.write("Set your habits in the sidebar (or pick a demo profile), then press the button 👇")
    if st.button("✨ Check my wellness", type="primary", use_container_width=True):
        with st.status("🔍 Analysing your habits...", expanded=True) as stt:
            st.write("📊 Reading your habits..."); time.sleep(.3)
            st.write("🧮 Running the wellness model, trying small changes...")
            res = coach.advise(row, use_ai=use_ai)
            st.write("🤖 Writing your coach notes..."); time.sleep(.3)
            stt.update(label="✅ All done!", state="complete", expanded=False)
        st.session_state["result"], st.session_state["fresh"], st.session_state["who"] = res, True, " ".join(w[:1].upper() + w[1:] for w in (name or "").strip()[:30].split())
        if res["explanation"]["predicted_category"] == "Good": st.balloons()
    res = st.session_state.get("result")
    if res:
        e = res["explanation"]; cat = e["predicted_category"]; pr = e["probabilities"]; idx = int(round(100 * (pr["Good"] + .5 * pr["Moderate"])))
        who = st.session_state.get("who", ""); fn = html.escape(first(who)); ini = html.escape(first(who)[:1].upper()) or "🙂"
        title = (html.escape(who) + "'s wellness report") if who else "Your wellness report"
        st.markdown(f'<div class="card rep"><div class="av">{ini}</div><div><h3 style="margin:0;padding:0">{title}</h3><small>🗓️ {datetime.date.today().strftime("%d %B %Y")} · {FACE[cat][0]} {cat}</small></div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card greet">{greeting(cat, who, e["uncertain"], idx, e, res["plan"], res.get("combo"))}</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1.35])
        with c1:
            st.markdown(f'<div class="cat" style="--c:{COL[cat]}"><span class="big">{FACE[cat][0]}</span><h2>{cat}</h2><div>{FACE[cat][1]}</div><small>Model confidence: {e["confidence"]:.0%}</small></div>', unsafe_allow_html=True)
            st.markdown(f'<div style="text-align:center">{gauge(idx)}</div>', unsafe_allow_html=True)
            st.markdown('<div class="card">' + probbars(pr) + '</div>', unsafe_allow_html=True)
            if e["uncertain"]: st.info("🤔 The model is not very sure about this one, so treat it as a rough estimate.")
        with c2:
            st.markdown("### 🔎 Why this result?")
            items = sorted(e["risk_drivers"] + e["protective_factors"], key=lambda r: -abs(r["impact"]))
            st.markdown('<div class="card">' + (whybars(items) if items else "No single habit stands out for this profile 🙂") + '</div>', unsafe_allow_html=True)
            st.markdown("### 🧾 Your habit scorecard")
            st.markdown(chips(res["scorecard"]), unsafe_allow_html=True)
            st.caption("🟢 usually lower risk · 🟡 keep an eye · 🔴 usually higher risk, based on patterns in this dataset")
        st.markdown(f"### 🤖 {fn + ', your' if fn else 'Your'} wellness coach")
        box = st.container(border=True)
        with box:
            if st.session_state.get("fresh"):
                st.write_stream(typed(res["summary"]))
            else:
                st.write(res["summary"])
            for t in res["suggestions"]: st.markdown(f"💡 {t}")
            st.caption(("✨ " if res["source"] == "Gemini AI" else "🛟 ") + f"Written by: {res['source']}" + (f" ({res['model']})" if res["model"] else ""))
        st.session_state["fresh"] = False
        if res["ai_error"] and use_ai:
            with st.expander("ℹ️ Why is this not Gemini?"): st.write("Gemini could not be used, so built-in guidance is shown. Details: " + res["ai_error"])
        elif use_ai and not coach.ai_available: st.caption("🔑 Add a Gemini API key in the sidebar to get AI-written guidance.")
        if res["plan"]:
            st.markdown(f"### 🎯 {fn + chr(39) + 's' if fn else 'Your'} action plan: small changes, biggest effect")
            st.markdown(plancards(res["plan"]), unsafe_allow_html=True)
            cb = res.get("combo")
            if cb and cb["to_cat"] == "At-Risk": st.info("💙 Several habits are high at once, so one change alone won't move the needle much. Try the plan together, and consider talking to a friend, mentor or your college counsellor.")
            if cb and cb["to_meter"] > cb["from_meter"]:
                st.markdown(f'<div class="card" style="--c:{COL[cb["to_cat"]]};border:2px solid {COL[cb["to_cat"]]};text-align:center;font-size:18px">🚀 Do the top {cb["n"]} together: wellness meter <b>{cb["from_meter"]}</b> ➜ <b>{cb["to_meter"]}</b> &nbsp; {FACE[cb["from_cat"]][0]} {cb["from_cat"]} ➜ {FACE[cb["to_cat"]][0]} <b>{cb["to_cat"]}</b></div>', unsafe_allow_html=True)
            st.caption("Estimates come from the prediction model (patterns in synthetic data), not promises.")
        if e["protective_factors"]:
            st.markdown("### 💪 Already working for you")
            st.markdown("".join(f'<span class="chip good">{EMOJI.get(f["feature"],"✅")} {f["label"]}: {f["value"]}</span>' for f in e["protective_factors"]), unsafe_allow_html=True)

        st.markdown(f'<div class="card greet" style="text-align:center">💪 You\'ve got this{", " + fn if fn else ""}! Pick <b>one</b> change from the plan and try it for a week, then check in again.</div>', unsafe_allow_html=True)
        st.download_button("📥 Download my report", report_text(res, who), file_name=f"wellbeing_report_{re.sub(r'[^A-Za-z0-9]', '', first(who)) or 'student'}.txt", mime="text/plain")

# ---------------- tab 2 ----------------
with tab2:
    st.write("Try changes to your current habits and watch the model's view move 🎢. No AI call is made here.")
    plan = (st.session_state.get("result") or {}).get("plan") or []
    use_plan = st.toggle("🎯 Apply my action plan instead of the sliders", value=False, disabled=not plan, help="Press 'Check my wellness' first to get a plan.")
    a, b = st.columns(2)
    d_use = a.slider("📱 Change in social media hours", -4.0, 4.0, -1.0, 0.5); d_late = a.slider("🌙 Change in late-night use (hours)", -2.0, 2.0, -0.5, 0.5)
    d_sleep = b.slider("😴 Change in sleep (hours)", -3.0, 3.0, 1.0, 0.5); d_act = b.slider("🏃 Change in activity (hours)", -1.0, 2.0, 0.5, 0.5)
    new = dict(row)
    if use_plan:
        for p in plan:
            nn = bump(new, *p["change"]); new = nn or new
    else:
        for f, d, lo, hi in [("Avg_Daily_Usage_Hours", d_use, .5, 12), ("Late_Night_Usage_Hours", d_late, 0, 4), ("Sleep_Hours_Per_Night", d_sleep, 3.5, 10), ("Physical_Activity_Hours", d_act, 0, 4)]:
            new = bump(new, f, d, lo, hi) or new
    p0, p1 = probs_of(row), probs_of(new); c0, c1 = max(p0, key=p0.get), max(p1, key=p1.get); d = (p1["At-Risk"] - p0["At-Risk"]) * 100
    arrow = "➡️" if c0 == c1 else ("🎉⬆️" if ["At-Risk","Moderate","Good"].index(c1) > ["At-Risk","Moderate","Good"].index(c0) else "⚠️⬇️")
    st.markdown(f'<div class="cmp"><div class="card" style="--c:{COL[c0]}"><small>Now</small><h2 style="color:{COL[c0]};padding:0">{FACE[c0][0]} {c0}</h2>{probbars(p0)}</div><div class="card" style="font-size:40px;flex:0 0 80px;display:flex;align-items:center;justify-content:center">{arrow}</div><div class="card" style="--c:{COL[c1]}"><small>After the changes</small><h2 style="color:{COL[c1]};padding:0">{FACE[c1][0]} {c1}</h2>{probbars(p1)}</div></div>', unsafe_allow_html=True)
    st.metric("At-Risk chance", f"{p1['At-Risk']:.0%}", delta=f"{d:+.0f} points", delta_color="inverse")
    st.caption("The model describes patterns in synthetic data. These numbers show how its estimate shifts, not what will happen to a real person.")

# ---------------- tab 3 ----------------
with tab3:
    t = card["test_metrics"]
    st.markdown(f"""
### 🧠 What it is
A prototype that predicts a wellness category (At-Risk, Moderate, Good) from lifestyle and stress-related habits, **explains** the prediction with SHAP, tries small changes to build an **action plan**, and writes general guidance with **Gemini** (or built-in wording when AI is unavailable).

### 📊 How well does it work?
Model: **{card['model']}** with engineered features, trained on a **synthetic** student dataset. Held-out test: accuracy **{t['accuracy']:.2f}**, macro-F1 **{t['macro_f1']:.2f}**, At-Risk recall **{t['atrisk_recall']:.2f}**.

### 🛡️ Limits
Synthetic data, so results describe the simulation, not real students. Predictions can be wrong. The AI only rewords the model's output and never makes the prediction. Only habit values are sent to the AI service. Your name stays on your screen and is never sent anywhere.
""")
st.caption("🧠 WellbeingIQ · prototype on synthetic data · not a medical tool · if you ever feel overwhelmed, please talk to a counsellor or someone you trust 💙")
st.caption("👨‍💻 Designed and built by Gulshan Kumar Kushwaha")