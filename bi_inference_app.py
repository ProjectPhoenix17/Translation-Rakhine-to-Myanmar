# -- coding: utf-8 --
"""
ရခိုင် ⇄ မြန်မာ — Bidirectional mBART-50 Inference Studio
Pairs with: bi_training_code.py
Model dir : ./rakhine_myanmar_bidirectional_model
RUN:
pip install -U streamlit transformers torch sentencepiece protobuf pandas
streamlit run bi_inference_app.py
Requires: streamlit >= 1.32, transformers >= 4.46
"""
import os
import time
import html as html_lib
import datetime
import streamlit as st

# ============================================================
# PAGE CONFIG  (must be the first Streamlit call)
# ============================================================
st.set_page_config(
    page_title="ရခိုင် ⇄ မြန်မာ · mBART Translator",
    page_icon="🪶",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Native widget accent (gold) — degrades gracefully on older versions
for _k, _v in {
    "theme.base": "dark",
    "theme.primaryColor": "#f0b429",
    "theme.backgroundColor": "#0b0e1a",
    "theme.secondaryBackgroundColor": "#141a2e",
    "theme.textColor": "#f4ecd9",
}.items():
    try:
        st._config.set_option(_k, _v)
    except Exception:
        pass

# ============================================================
# ART DIRECTION — pagoda gold · coastal blue · lacquer crimson
# ============================================================
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Myanmar:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root{
--gold:#f0b429; --gold-soft:#ffd97a; --crimson:#e5484d; --jade:#2fbf71;
--ocean:#4cc3ff; --ink:#0b0e1a; --surface:#121732; --line:#2a3357;
--text:#f4ecd9; --muted:#9aa3c7;
}
html, body, .stApp, [class*="st-"]{
font-family:'Space Grotesk','Noto Sans Myanmar',sans-serif;
}
.stApp{
background:
radial-gradient(1100px 620px at 88% -12%, rgba(76,195,255,.15), transparent 60%),
radial-gradient(950px 560px at -8% 12%, rgba(240,180,41,.14), transparent 55%),
radial-gradient(820px 620px at 55% 112%, rgba(229,72,77,.13), transparent 60%),
linear-gradient(160deg,#0b0e1a 0%,#0e1226 55%,#131022 100%);
color:var(--text);
}
#MainMenu{visibility:hidden} footer{visibility:hidden}
header[data-testid="stHeader"]{background:transparent}
/* ---- scrollbars, in-palette ---- */
::-webkit-scrollbar{width:10px;height:10px}
::-webkit-scrollbar-track{background:#0b0e1a}
::-webkit-scrollbar-thumb{background:linear-gradient(180deg,var(--gold),var(--crimson));
border-radius:8px;border:2px solid #0b0e1a}
::-webkit-scrollbar-thumb:hover{background:linear-gradient(180deg,var(--gold-soft),var(--crimson))}
/* ---- hero ---- */
.hero{padding:.4rem 0 1.1rem}
.hero-kicker{font-size:.72rem;letter-spacing:.3em;color:var(--ocean);font-weight:700}
.hero h1{margin:.15rem 0 .2rem;font-size:clamp(2.1rem,4.5vw,3.3rem);font-weight:700;line-height:1.35;
background:linear-gradient(90deg,var(--gold-soft),var(--gold),var(--crimson),var(--ocean),var(--gold-soft));
background-size:300% 100%;-webkit-background-clip:text;background-clip:text;color:transparent;
animation:shimmer 9s linear infinite}
@keyframes shimmer{to{background-position:300% 0}}
.hero-sub{color:var(--muted);font-size:.95rem;margin-bottom:.7rem}
.pill{display:inline-flex;align-items:center;gap:.45rem;padding:.28rem .85rem;border-radius:999px;
border:1px solid var(--line);background:rgba(18,23,50,.72);font-size:.78rem;color:var(--muted);
margin:0 .4rem .4rem 0}
.dot{width:.5rem;height:.5rem;border-radius:50%}
.dot-jade{background:var(--jade);box-shadow:0 0 8px var(--jade)}
.dot-gold{background:var(--gold);box-shadow:0 0 8px var(--gold)}
.dot-ocean{background:var(--ocean);box-shadow:0 0 8px var(--ocean)}
.dot-crimson{background:var(--crimson);box-shadow:0 0 8px var(--crimson)}
/* ---- badges / labels ---- */
.badge{display:inline-block;padding:.2rem .75rem;border-radius:999px;font-size:.76rem;font-weight:700}
.badge-rkh{background:rgba(76,195,255,.14);color:var(--ocean);border:1px solid rgba(76,195,255,.45)}
.badge-mm{background:rgba(240,180,41,.14);color:var(--gold-soft);border:1px solid rgba(240,180,41,.45)}
.col-cap{font-size:.7rem;letter-spacing:.22em;color:var(--muted);font-weight:700;margin-left:.5rem}
/* Hide the Streamlit sidebar and its expand/collapse control */
[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"]{display:none!important}
div[role="slider"]{background:var(--gold)!important;border-color:var(--gold)!important;
box-shadow:0 0 0 4px rgba(240,180,41,.16)}
/* ---- buttons ---- */
.stButton>button,.stDownloadButton>button{border-radius:12px;border:1px solid var(--line);
background:var(--surface);color:var(--text);font-weight:600;transition:all .18s ease}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:var(--gold);color:var(--gold-soft);
transform:translateY(-1px);box-shadow:0 6px 18px rgba(240,180,41,.16)}
button[kind="primary"],[data-testid="stBaseButton-primary"]{
background:linear-gradient(135deg,var(--gold) 0%,#f2762e 55%,var(--crimson) 100%)!important;
color:#1a1206!important;border:none!important;font-weight:700;
box-shadow:0 8px 24px rgba(229,72,77,.28)}
button[kind="primary"]:hover{filter:brightness(1.09);transform:translateY(-1px)}
button:disabled{opacity:.45;transform:none!important;box-shadow:none!important}
/* ---- inputs ---- */
.stTextArea textarea{background:rgba(13,17,38,.88)!important;border:1px solid var(--line)!important;
border-radius:14px!important;color:var(--text)!important;
font-family:'Noto Sans Myanmar','Space Grotesk',sans-serif!important;
font-size:1.06rem!important;line-height:2!important;padding:14px 16px!important}
.stTextArea textarea:focus{border-color:var(--gold)!important;box-shadow:0 0 0 3px rgba(240,180,41,.18)!important}
.stTextInput input{background:rgba(13,17,38,.88)!important;border:1px solid var(--line)!important;
border-radius:10px!important;color:var(--text)!important}
/* ---- output panel ---- */
[data-testid="stCodeBlock"],[data-testid="stCode"]{border:1px solid rgba(240,180,41,.38);border-radius:14px;
background:linear-gradient(160deg,rgba(240,180,41,.08),rgba(76,195,255,.06)),#0d1124!important}
[data-testid="stCodeBlock"] pre,[data-testid="stCode"] pre{
font-family:'Noto Sans Myanmar','Space Grotesk',sans-serif!important;
font-size:1.08rem!important;line-height:2!important;max-height:300px;overflow:auto}
[data-testid="stCodeBlock"] code,[data-testid="stCode"] code{color:var(--text)!important;background:transparent!important}
/* ---- tabs ---- */
.stTabs [data-baseweb="tab-list"]{gap:.4rem;border-bottom:1px solid var(--line)}
.stTabs [data-baseweb="tab"]{font-weight:600;color:var(--muted)}
.stTabs [aria-selected="true"]{color:var(--gold-soft)!important}
.stTabs [data-baseweb="tab-highlight"]{background:var(--gold)!important}
/* ---- metrics  & cards ---- */
[data-testid="stMetric"]{background:rgba(18,23,50,.78);border:1px solid var(--line);
border-radius:14px;padding:.8rem 1rem}
[data-testid="stMetricLabel"]{color:var(--muted)!important}
[data-testid="stMetricValue"]{color:var(--gold-soft)!important;font-weight:700}
[data-testid="stVerticalBlockBorderWrapper"]{border:1px solid var(--line)!important;
border-radius:16px!important;background:rgba(15,19,42,.62)}
.hist-meta{font-size:.76rem;color:var(--muted);margin-left:.5rem}
.hist-src{color:var(--muted);font-size:.92rem;line-height:1.9;
font-family:'Noto Sans Myanmar',sans-serif}
.hist-out{color:var(--text);font-size:1.02rem;line-height:2;
font-family:'Noto Sans Myanmar',sans-serif}
.empty{border:1px dashed var(--line);border-radius:16px;padding:2.4rem 1.4rem;text-align:center;
color:var(--muted);background:rgba(13,17,38,.5)}
.empty .big{font-size:2.1rem;margin-bottom:.4rem}
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

# ============================================================
# HEAVY IMPORTS (graceful failure)
# ============================================================
try:
    import torch
    import pandas as pd
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except ImportError as e:
    st.error(f"Missing dependency: `{e}`")
    st.code("pip install -U streamlit transformers torch sentencepiece protobuf pandas")
    st.stop()

# ============================================================
# CONSTANTS — must match bi_training_code.py
# ============================================================
BASE_MODEL = "facebook/mbart-large-50"
DEFAULT_MODEL_DIR = "./rakhine_myanmar_bidirectional_model"
LANG = "my_MM"                       # both sides (Rakhine not in mBART-50 list)
MAX_INPUT = 256
PREFIX_MM_TO_RKH = "translate Myanmar to Rakhine:"
PREFIX_RKH_TO_MM = "translate Rakhine to Myanmar:"
DIR_LABEL = {
    "rkh2mm": "ရခိုင် → မြန်မာ  ·  Rakhine → Myanmar",
    "mm2rkh": "မြန်မာ → ရခိုင်  ·  Myanmar → Rakhine",
}

PARAM_DEFAULTS = {
    "num_beams": 4,  "max_new_tokens": 128,  "length_penalty": 1.0,
    "no_repeat_ngram": 0,  "do_sample": False,  "temperature": 1.0,
    "top_p": 0.9,  "early_stop": True,
}

# ============================================================
# SESSION STATE
# ============================================================
for k, v in PARAM_DEFAULTS.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("direction", "rkh2mm")
st.session_state.setdefault("batch_dir", "rkh2mm")
st.session_state.setdefault("batch_text", "")
st.session_state.setdefault("batch_df", None)
st.session_state.setdefault("history", [])
st.session_state.setdefault("last_result", None)
st.session_state.setdefault("model_path", DEFAULT_MODEL_DIR)
st.session_state.setdefault("device_choice", "auto")
st.session_state.setdefault("allow_base", False)

# ============================================================
# CALLBACKS
# ============================================================
def cb_reset_params():
    for k, v in PARAM_DEFAULTS.items():
        st.session_state[k] = v

def cb_reload():
    st.cache_resource.clear()

def cb_swap():
    st.session_state.direction = (
        "mm2rkh" if st.session_state.direction == "rkh2mm" else "rkh2mm"
    )
    lr = st.session_state.get("last_result")
    if lr:
        st.session_state.src_text = lr["out"]

def cb_clear():
    st.session_state.src_text = ""
    st.session_state.last_result = None

def cb_del_hist(hid):
    st.session_state.history = [h for h in st.session_state.history if h["id"] != hid]

def cb_clear_hist():
    st.session_state.history = []

def cb_clear_batch():
    st.session_state.batch_df = None
    st.session_state.batch_text = ""

# ============================================================
# MODEL LOADER (cached)
# ============================================================
@st.cache_resource(show_spinner=False)
def load_pipeline(path, device_choice, allow_base):
    fine_tuned = os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json"))
    if fine_tuned:
        load_from = path
    elif allow_base:
        load_from = BASE_MODEL
    else:
        return None
    
    tok = AutoTokenizer.from_pretrained(load_from, src_lang=LANG, tgt_lang=LANG)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(load_from)
    
    if device_choice == "cpu":
        device = "cpu"
    elif device_choice == "cuda":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    mdl.to(device)
    mdl.eval()
    mdl.config.use_cache = True
    
    bos = (
        tok.lang_code_to_id[LANG]
        if hasattr(tok, "lang_code_to_id")
        else tok.convert_tokens_to_ids(LANG)
    )
    
    meta = {"fine_tuned": fine_tuned, "load_from": load_from, "device": device}
    return mdl, tok, device, bos, meta

# ============================================================
# TRANSLATION CORE
# ============================================================
def run_translation(text, direction, mdl, tok, device, bos):
    prefix = PREFIX_RKH_TO_MM if direction == "rkh2mm" else PREFIX_MM_TO_RKH
    text = text.strip()
    if not text.startswith("translate "):
        text = f"{prefix} {text}"
        
    enc = tok(text, return_tensors="pt", truncation=True, max_length=MAX_INPUT).to(device)
    
    kw = dict(
        max_new_tokens=st.session_state.max_new_tokens,
        num_beams=st.session_state.num_beams,
        forced_bos_token_id=bos,
    )
    
    if st.session_state.no_repeat_ngram > 0:
        kw["no_repeat_ngram_size"] = st.session_state.no_repeat_ngram
        
    if st.session_state.num_beams > 1:
        kw["length_penalty"] = st.session_state.length_penalty
        if not st.session_state.do_sample:
            kw["early_stopping"] = st.session_state.early_stop
            
    if st.session_state.do_sample:
        kw.update(
            do_sample=True,
            temperature=st.session_state.temperature,
            top_p=st.session_state.top_p,
        )
        
    t0 = time.perf_counter()
    with torch.no_grad():
        out = mdl.generate(**enc, **kw)
    secs = time.perf_counter() - t0
    
    decoded = tok.decode(out[0], skip_special_tokens=True)
    return decoded, secs, int(enc["input_ids"].shape[-1]), int(out.shape[-1])

def lang_badge(which):
    label = "ရခိုင် (Rakhine)" if which == "rkh" else "မြန်မာ (Myanmar)"
    return f'<span class="badge badge-{which}">{label}</span>'

def stamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# ============================================================
# LOAD MODEL
# ============================================================
pipeline, load_error = None, None
try:
    with st.spinner("Loading mBART weights… first load can take a minute"):
        pipeline = load_pipeline(
            st.session_state.model_path,
            st.session_state.device_choice,
            st.session_state.allow_base,
        )
except Exception as e:
    load_error = str(e)

if pipeline:
    model, tokenizer, device, bos_id, meta = pipeline
else:
    model = tokenizer = device = bos_id = meta = None

# ============================================================
# HERO
# ============================================================
if meta and meta["fine_tuned"]:
    model_pill = '<span class="pill"><span class="dot dot-jade"></span>Fine-tuned model</span>'
elif meta:
    model_pill = '<span class="pill"><span class="dot dot-crimson"></span>Base mBART-50 · not fine-tuned</span>'
else:
    model_pill = '<span class="pill"><span class="dot dot-crimson"></span>Model not loaded</span>'

dev_pill = (f'<span class="pill"><span class="dot dot-ocean"></span>{device.upper()}</span>'
            if device else "")

pills = (model_pill + dev_pill +
         '<span class="pill"><span class="dot dot-gold"></span>my_MM ⇄ my_MM</span>'
         f'<span class="pill"><span class="dot dot-gold"></span>beams {st.session_state.num_beams}</span>')

st.markdown(f"""
<div class="hero">
<h2>ရခိုင် ⇄ မြန်မာ</h2>
</div>
""", unsafe_allow_html=True)

if load_error:
    st.error(f"Model load failed: {load_error}")

if pipeline is None and not load_error:
    st.warning(
        f"Fine-tuned model not found at `{st.session_state.model_path}` — "
        "run `bi_training_code.py` first, or set the correct folder in Model & Decoding Settings. "
        "(You may enable the base-model fallback in Model & Decoding Settings, but output will not be proper Rakhine.)"
    )
elif meta and not meta["fine_tuned"]:
    st.warning("⚠ Base-model fallback active — translations will be unreliable until you load the fine-tuned model.")

# ============================================================
# TABS
# ============================================================
tab_single, tab_batch, tab_hist = st.tabs(
    ["🪶 Translate", "📚 Batch", "🕘 History"]
)

# ------------------------------------------------------------
# TAB 1 · SINGLE TRANSLATE
# ------------------------------------------------------------
with tab_single:
    st.radio("Direction", list(DIR_LABEL), format_func=DIR_LABEL.get,
             horizontal=True, key="direction", label_visibility="collapsed")
    
    direction = st.session_state.direction
    src_key = "rkh" if direction == "rkh2mm" else "mm"
    tgt_key = "mm" if direction == "rkh2mm" else "rkh"
    
    col_l, col_r = st.columns(2, gap="large")
    with col_l:
        st.markdown(lang_badge(src_key) + '<span class="col-cap">SOURCE</span>',
                    unsafe_allow_html=True)
        st.text_area("source", key="src_text", height=230,
                     label_visibility="collapsed",
                     placeholder="ဘာသာပြန်လိုသော စာကြောင်းကို ရိုက်ထည့်ပါ…")
        st.caption(f"{len(st.session_state.src_text)} characters")
        
    with col_r:
        st.markdown(lang_badge(tgt_key) + '<span class="col-cap">TRANSLATION</span>',
                    unsafe_allow_html=True)
        lr = st.session_state.last_result
        if lr:
            st.code(lr["out"], language=None)
            st.caption(f"generated with beams {lr['beams']} · {lr['secs']:.2f}s · "
                       f"{DIR_LABEL[lr['dir']]}")
        else:
            st.markdown(
                '<div class="empty"><div class="big">🪶</div>'
                'Translation will appear here.<br>'
                '<span style="font-size:.85rem">Translate ကို နှိပ်ပြီး စမ်းကြည့်ပါ။</span>'
                '</div>', unsafe_allow_html=True)

    b1, b2, b3, _ = st.columns([2, 1, 1, 2])
    go = b1.button("⚡ Translate", type="primary", use_container_width=True,
                   disabled=pipeline is None)
    b2.button("⇄ Swap", on_click=cb_swap, use_container_width=True,
              help="Flip direction and feed the last translation back as input.")
    b3.button("✕ Clear", on_click=cb_clear, use_container_width=True)

    if go:
        if not st.session_state.src_text.strip():
            st.warning("Please enter some text first · စာကြောင်း ထည့်ပါ")
        else:
            with st.spinner("Translating…"):
                out, secs, tin, tout = run_translation(
                    st.session_state.src_text, direction,
                    model, tokenizer, device, bos_id,
                )
            rec = {
                "id": time.time_ns(),
                "ts": datetime.datetime.now().strftime("%H:%M:%S"),
                "dir": direction,
                "src": st.session_state.src_text.strip(),
                "out": out, "secs": secs, "tin": tin, "tout": tout,
                "beams": st.session_state.num_beams,
            }
            st.session_state.last_result = rec
            st.session_state.history.insert(0, rec)
            st.toast("Translation ready ✓", icon="🪶")
            st.rerun()

    if st.session_state.last_result:
        lr = st.session_state.last_result
        m1, m2, m3, m4, m5 = st.columns([1, 1, 1, 1, 1.5])
        m1.metric("Latency", f"{lr['secs']:.2f}s")
        m2.metric("Input tokens", lr["tin"])
        m3.metric("Output tokens", lr["tout"])
        m4.metric("Beams", lr["beams"])
        m5.download_button("⬇ Download .txt", data=lr["out"].encode("utf-8"),
                           file_name=f"translation_{stamp()}.txt",
                           mime="text/plain", use_container_width=True)

# ------------------------------------------------------------
# TAB 2 · BATCH
# ------------------------------------------------------------
with tab_batch:
    st.radio("Batch direction", list(DIR_LABEL), format_func=DIR_LABEL.get,
             horizontal=True, key="batch_dir", label_visibility="collapsed")
    
    up = st.file_uploader("Upload a .txt file (one sentence per line)", type=["txt"])
    st.text_area("…or paste lines here", key="batch_text", height=170,
                 placeholder="မင်း ရီကူး နိုင် ပါလား ။\nရာသီဥတု က ကကောင်း အီးရေ ။")

    if up is not None:
        raw = up.getvalue().decode("utf-8", errors="replace")
        st.caption(f"📄 {up.name} · {len(raw.splitlines())} lines (file takes priority)")
    else:
        raw = st.session_state.batch_text

    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    st.caption(f"{len(lines)} non-empty sentences ready")

    bb1, bb2, _ = st.columns([2, 1, 2])
    run_batch = bb1.button("📚 Translate batch", type="primary",
                           use_container_width=True, disabled=pipeline is None)
    bb2.button("✕ Clear batch", on_click=cb_clear_batch, use_container_width=True)

    if run_batch:
        if not lines:
            st.warning("No input lines found.")
        else:
            if len(lines) > 200:
                st.warning("Capping at 200 lines per run to keep things responsive.")
                lines = lines[:200]
            prog = st.progress(0.0, text="Translating…")
            rows, t0 = [], time.perf_counter()
            for i, line in enumerate(lines):
                out, *_ = run_translation(
                    line, st.session_state.batch_dir,
                    model, tokenizer, device, bos_id,
                )
                rows.append({"Source": line, "Translation": out})
                prog.progress((i + 1) / len(lines),
                              text=f"Translating… {i + 1}/{len(lines)}")
            prog.empty()
            st.session_state.batch_df = pd.DataFrame(rows)
            st.session_state.batch_secs = time.perf_counter() - t0
            st.toast(f"{len(rows)} sentences translated ✓", icon="📚")
            st.rerun()

    if st.session_state.batch_df is not None:
        df = st.session_state.batch_df
        st.caption(f"{len(df)} rows · {st.session_state.get('batch_secs', 0):.1f}s total · "
                   f"{DIR_LABEL[st.session_state.batch_dir]}")
        st.dataframe(df, use_container_width=True, hide_index=True, height=340,
                     column_config={
                         "Source": st.column_config.TextColumn(width="large"),
                         "Translation": st.column_config.TextColumn(width="large"),
                     })
        d1, d2, _ = st.columns([1, 1, 2])
        d1.download_button("⬇ Translations .txt",
                           data="\n".join(df["Translation"]).encode("utf-8"),
                           file_name=f"batch_translations_{stamp()}.txt",
                           mime="text/plain", use_container_width=True)
        d2.download_button("⬇ Parallel .csv",
                           data=df.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"batch_parallel_{stamp()}.csv",
                           mime="text/csv", use_container_width=True)
    else:
        st.markdown('<div class="empty"><div class="big">📚</div>'
                    'Upload a file or paste sentences, then translate the whole batch.'
                    '</div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# TAB 3 · HISTORY
# ------------------------------------------------------------
with tab_hist:
    hist = st.session_state.history
    if not hist:
        st.markdown('<div class="empty"><div class="big">🕘</div>'
                    'No translations yet — everything you translate is logged here.'
                    '</div>', unsafe_allow_html=True)
    else:
        h1, h2, h3, _ = st.columns([1.4, 1, 1, 1.6])
        h1.caption(f"{len(hist)} records this session")
        hist_df = pd.DataFrame([
            {"Time": h["ts"], "Direction": DIR_LABEL[h["dir"]],
             "Source": h["src"], "Translation": h["out"],
             "Beams": h["beams"], "Seconds": round(h["secs"], 2)}
            for h in hist
        ])
        h2.download_button("⬇ Export CSV",
                           data=hist_df.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"history_{stamp()}.csv",
                           mime="text/csv", use_container_width=True)
        h3.button("🗑 Clear all", on_click=cb_clear_hist, use_container_width=True)

        for h in hist[:50]:
            with st.container(border=True):
                cc1, cc2 = st.columns([9, 1])
                badge = lang_badge("rkh" if h["dir"] == "rkh2mm" else "mm")
                arrow = " → "
                badge2 = lang_badge("mm" if h["dir"] == "rkh2mm" else "rkh")
                cc1.markdown(
                    badge + arrow + badge2 +
                    f'<span class="hist-meta">{h["ts"]} · beams {h["beams"]} · '
                    f'{h["secs"]:.2f}s</span>', unsafe_allow_html=True)
                cc2.button("✕", key=f"del_{h['id']}", on_click=cb_del_hist,
                           args=(h["id"],), help="Delete this record")
                st.markdown(f'<div class="hist-src">{html_lib.escape(h["src"])}</div>',
                            unsafe_allow_html=True)
                st.markdown(f'<div class="hist-out">{html_lib.escape(h["out"])}</div>',
                            unsafe_allow_html=True)

st.divider()
st.caption("🪶 Rakhine ⇄ Myanmar Translation Studio · pairs with bi_training_code.py · "
           "facebook/mbart-large-50 · prefix-conditioned bidirectional inference")