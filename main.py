# streamlit_app.py
# Hybrid Cohere + Local HF Buddhist-style chatbot
# Uses UNIVERSAL_SYSTEM_PROMPT for both online and offline

import os
import time
import threading
from pathlib import Path
from typing import List, Dict

import streamlit as st
from dotenv import load_dotenv

# Online (Cohere)
import cohere

# Offline (Transformers)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from huggingface_hub import snapshot_download, login

# =========================
# Environment & Config
# =========================
load_dotenv()

COHERE_API_KEY = st.secrets.get("COHERE_API_KEY", os.getenv("COHERE_API_KEY", ""))
HF_TOKEN = st.secrets.get("HUGGINGFACE_TOKEN", os.getenv("HUGGINGFACE_TOKEN", ""))
HF_HOME = st.secrets.get("HF_HOME", os.getenv("HF_HOME", None))
if HF_HOME:
    os.environ["HF_HOME"] = HF_HOME

DEFAULT_OFFLINE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


COHERE_MODEL = "command-r"
LOCAL_MODELS_DIR = Path("models")

# =========================
# Universal System Prompt
# =========================
UNIVERSAL_SYSTEM_PROMPT = """
Personal AI Buddha — Calm, Friendly, Mindful

ROLE & TONE
You are “Personal AI Buddha,” a gentle, respectful, loving, and lightly humorous spiritual companion.
Reply warmly and directly. For greetings (hi, hello, good morning), respond naturally in 1–2 sentences.
Do not repeat the user’s message in offline mode.

DOCTRINAL STANCE
• Neutral across Theravāda, Mahāyāna, Vajrayāna; explain differences simply.
• Grounded in canonical sources: Pāli Canon, Mahāyāna sūtras, Vajrayāna termas.
• If unsure, admit uncertainty.

LANGUAGE & STYLE
• Use the user’s dominant language (English/Nepali).
• Short glosses for key terms (e.g., anicca = impermanence) if helpful.
• Tone: calm, soft, kind, lightly humorous, accurate, and clear.

SAFETY & SCOPE
• Not a medical, legal, or crisis professional. For urgent matters, suggest seeking local help.
• Avoid advanced practices without a qualified teacher.
• Be accurate with historical facts and sacred sites.

OUTPUT RULES
1) Respond **only with your answer**, do not repeat the user’s question.
2) Keep greetings short, friendly, and mindful.
3) For questions: 1–3 sentences for concise answers, more if needed; add small practical advice if appropriate.
4) For checklists/plans: use clear bullets or numbers.
5) Maintain warmth, accuracy, and gentle humor.
"""

# =========================
# Page Setup
# =========================
st.set_page_config(page_title="Hybrid Buddhist Chatbot", page_icon="🧘", layout="centered")

# =========================
# Utilities
# =========================
def device_hint() -> str:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def device_emoji() -> str:
    dev = device_hint()
    return "🟢 CUDA" if dev == "cuda" else "🟣 MPS" if dev == "mps" else "⚪ CPU"

def ensure_logged_in_to_hf():
    if HF_TOKEN:
        try:
            login(token=HF_TOKEN, add_to_git_credential=False)
        except Exception:
            pass

def repo_local_dir(repo_id: str) -> Path:
    safe = repo_id.replace("/", "__")
    return LOCAL_MODELS_DIR / safe

def has_local_model(repo_id: str) -> bool:
    d = repo_local_dir(repo_id)
    return d.exists() and any((d / f).exists() for f in [
        "config.json", "tokenizer.json", "tokenizer.model", "pytorch_model.bin", "model.safetensors"
    ])

def status_badge(mode: str) -> str:
    color = "green" if mode == "Online" else "red"
    return f'<span class="badge {color}"><span class="dot"></span><span class="label">{mode}</span></span>'

def to_cohere_history(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    mapping = {"user": "USER", "assistant": "CHATBOT"}
    return [{"role": mapping[m["role"]], "message": m["content"]} for m in messages if m["role"] in mapping]

def export_chat_md(messages: List[Dict[str, str]]) -> str:
    lines = ["# Hybrid Chat Transcript\n"]
    for m in messages:
        who = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"**{who}:** {m['content']}")
    return "\n\n".join(lines)

# =========================
# Download & Load (Offline)
# =========================
def download_model_with_status(repo_id: str) -> Path:
    LOCAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    d = repo_local_dir(repo_id)
    with st.status(f"Preparing **{repo_id}** …", expanded=True) as status:
        ensure_logged_in_to_hf()
        done = threading.Event()
        err_holder = {"err": None}

        def _dl():
            try:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=d.as_posix(),
                    local_dir_use_symlinks=False,
                )
            except Exception as e:
                err_holder["err"] = e
            finally:
                done.set()

        t = threading.Thread(target=_dl, daemon=True)
        t.start()
        prog = st.progress(0.0, text="Fetching model files …")
        k = 0
        while not done.is_set():
            prog.progress(min(0.98, (k % 120) / 120.0), text="Fetching model files …")
            time.sleep(0.1)
            k += 1
        prog.progress(1.0, text="Download complete.")
        if err_holder["err"]:
            status.update(label=f"Download failed: {err_holder['err']}", state="error")
            raise err_holder["err"]
        status.update(label=f"**{repo_id}** is ready.", state="complete")
    return d

@st.cache_resource(show_spinner=False)
def load_local_pipeline(repo_id: str, offline: bool = True):
    d = repo_local_dir(repo_id).as_posix()
    dev = device_hint()
    dtype = torch.float16 if dev in ("cuda", "mps") else torch.float32
    tok = AutoTokenizer.from_pretrained(d, use_fast=True, local_files_only=offline)
    model = AutoModelForCausalLM.from_pretrained(
        d,
        torch_dtype=dtype,
        device_map="auto" if dev in ("cuda", "mps") else None,
        local_files_only=offline,
    )
    model.eval()
    return tok, model, dev

# =========================
# Offline Prompt Template (Fixed)
# =========================

def apply_chat_template(tokenizer, messages: List[Dict[str, str]]) -> str:
    """
    Offline prompt template: 
    - System prompt first (guidelines for Personal AI Buddha)
    - Only the latest user message, without any 'User:' prefix
    - Returns a single string to pass to the model

Personal AI Buddha — Calm, Friendly, Mindful

ROLE & TONE
You are “Personal AI Buddha,” a gentle, respectful, loving, and lightly humorous spiritual companion.
Reply warmly and directly. For greetings (hi, hello, good morning), respond naturally in 1–2 sentences.
Do not repeat the user’s message in offline mode.

DOCTRINAL STANCE
• Neutral across Theravāda, Mahāyāna, Vajrayāna; explain differences simply.
• Grounded in canonical sources: Pāli Canon, Mahāyāna sūtras, Vajrayāna termas.
• If unsure, admit uncertainty.

LANGUAGE & STYLE
• Use the user’s dominant language (English/Nepali).
• Short glosses for key terms (e.g., anicca = impermanence) if helpful.
• Tone: calm, soft, kind, lightly humorous, accurate, and clear.

SAFETY & SCOPE
• Not a medical, legal, or crisis professional. For urgent matters, suggest seeking local help.
• Avoid advanced practices without a qualified teacher.
• Be accurate with historical facts and sacred sites.

OUTPUT RULES
1) Respond **only with your answer**, do not repeat the user’s question.
2) Keep greetings short, friendly, and mindful.
3) For questions: 1–3 sentences for concise answers, more if needed; add small practical advice if appropriate.
4) For checklists/plans: use clear bullets or numbers.
5) Maintain warmth, accuracy, and gentle humor.

    """
    system = UNIVERSAL_SYSTEM_PROMPT.strip()
    # get last user message
    last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    prompt_text = f"{system}\n\n{last_user}\n"
    return prompt_text

# =========================
# Generation (Online)
# =========================
def cohere_generate(api_key: str, history: List[Dict[str, str]], user_msg: str,
                    temperature: float = 0.45, stream: bool = True) -> str:
    co = cohere.Client(api_key=api_key)
    chat_history = to_cohere_history(history)

    if stream:
        out = []
        with st.chat_message("assistant", avatar="🧘"):
            ph = st.empty()
            text_so_far = ""
            for event in co.chat_stream(
                model=COHERE_MODEL,
                message=user_msg,
                chat_history=chat_history,
                preamble=UNIVERSAL_SYSTEM_PROMPT,
                temperature=temperature,
            ):
                if event.event_type == "text-generation":
                    token = event.text
                    out.append(token)
                    text_so_far += token
                    ph.markdown(text_so_far)
        return "".join(out)
    else:
        resp = co.chat(
            model=COHERE_MODEL,
            message=user_msg,
            chat_history=chat_history,
            preamble=UNIVERSAL_SYSTEM_PROMPT,
            temperature=temperature,
        )
        return (getattr(resp, "text", "") or "").strip()

# =========================
# State
# =========================
if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, str]] = []

if "mode" not in st.session_state:
    st.session_state.mode = "Online"

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.markdown("## 🎛️ Controls")
    online_toggle = st.toggle("Online mode (Cohere)", value=(st.session_state.mode == "Online"))
    st.session_state.mode = "Online" if online_toggle else "Offline"

    st.markdown("---")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.45, 0.05)
    max_new_tokens = st.slider("Max tokens", 64, 768, 256, 16)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)

    offline_choice = st.selectbox(
        "Offline Model",
        [DEFAULT_OFFLINE_MODEL],
        index=0
    )
    if st.button("📦 Prepare Offline"):
        try:
            download_model_with_status(offline_choice)
            st.success("Model ready for offline use.")
        except Exception as e:
            st.error(f"Prep failed: {e}")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🧹 Clear Chat"):
            st.session_state.messages = []
            st.toast("Chat cleared.", icon="🧼")
    with col_b:
        md = export_chat_md(st.session_state.messages)
        st.download_button("⬇️ Export", data=md, file_name="chat.md", mime="text/markdown")

# =========================
# Header
# =========================
st.markdown(
    f"""
    <div class="header">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
          <div class="header-title">Hybrid Buddhist Chatbot</div>
          <div class="header-sub">Calm, mindful answers — online or offline.</div>
        </div>
        <div>{status_badge(st.session_state.mode)}</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================
# Chat display
# =========================
for m in st.session_state.messages:
    role, avatar = ("user", "👤") if m["role"] == "user" else ("assistant", "🧘")
    bubble_class = "chat-bubble-user" if role == "user" else "chat-bubble-bot"
    with st.chat_message(role, avatar=avatar):
        st.markdown(f"<div class='{bubble_class}'>{m['content']}</div>", unsafe_allow_html=True)

# =========================
# Chat input & routing
# =========================
prompt = st.chat_input("Ask with a calm mind…")
if prompt:
    # Show user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(f"<div class='chat-bubble-user'>{prompt}</div>", unsafe_allow_html=True)
    st.session_state.messages.append({"role": "user", "content": prompt})

    answer = ""
    try:
        if st.session_state.mode == "Online":
            if not COHERE_API_KEY:
                with st.chat_message("assistant", avatar="🧘"):
                    st.warning("Set `COHERE_API_KEY` in Streamlit secrets to use Online mode.")
            else:
                answer = cohere_generate(
                    api_key=COHERE_API_KEY,
                    history=st.session_state.messages[:-1],
                    user_msg=prompt,
                    temperature=temperature,
                    stream=True
                )
        else:
            if not has_local_model(offline_choice):
                download_model_with_status(offline_choice)
            tok, model, dev = load_local_pipeline(offline_choice, offline=True)

            prompt_text = apply_chat_template(tok, st.session_state.messages)
            inputs = tok(prompt_text, return_tensors="pt")
            if dev in ("cuda", "mps"):
                inputs = {k: v.to(dev) for k, v in inputs.items()}

            streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
            gen_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.05,
            )

            thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
            thread.start()

            with st.chat_message("assistant", avatar="🧘"):
                ph = st.empty()
                text_so_far = ""
                for token in streamer:
                    text_so_far += token
                    ph.markdown(f"<div class='chat-bubble-bot'>{text_so_far}</div>", unsafe_allow_html=True)
            answer = text_so_far.strip()

    except Exception as e:
        with st.chat_message("assistant", avatar="🧘"):
            st.error(f"Error: {e}")

    if answer:
        st.session_state.messages.append({"role": "assistant", "content": answer})
