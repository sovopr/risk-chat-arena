import streamlit as st
import time
import concurrent.futures
import io
import fitz  # PyMuPDF
import tempfile
import hashlib
from openai import OpenAI
from google import genai
from google.genai import types

st.set_page_config(page_title="Risk Chat Arena", layout="wide")

# --- CONFIGURATION ---
MODEL_GPT5 = "gpt-5-mini"          # use gpt-5-mini as ChatGPT side
MODEL_B = "gemini-3-pro-preview"   # Gemini for side-by-side comparison

# ✅ API KEYS: now loaded from Streamlit secrets (NO hardcoding)
CHATGPT5_API_KEY = st.secrets.get("OPENAI_API_KEY")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

# --- COST ASSUMPTIONS (per 1M tokens) - EDIT THESE TO REAL PRICES ---
COST_GPT5_INPUT = 0.00
COST_GPT5_OUTPUT = 0.00
COST_B_INPUT = 0.00
COST_B_OUTPUT = 0.00

# --- SDK CLIENTS ---
openai_client = OpenAI(api_key=CHATGPT5_API_KEY) if CHATGPT5_API_KEY else None

# --- SESSION STATE ---
def init_state(key, default):
    if key not in st.session_state:
        st.session_state[key] = default

init_state("messages_gpt5", [])
init_state("messages_b", [])
init_state("tokens_gpt5", {"input": 0, "output": 0})
init_state("tokens_b", {"input": 0, "output": 0})
# cache map: pdf_hash -> openai file_id
init_state("openai_file_cache", {})

# --- HELPERS: PDF text + scanned detection ---
def get_pdf_data(uploaded_file):
    """
    Returns:
    1. text_content (str): Extracted text (if any)
    2. file_bytes (bytes): Raw PDF bytes
    3. is_scanned (bool): True if text extraction failed (likely scanned)
    """
    text_content = ""
    file_bytes = uploaded_file.getvalue()
    is_scanned = False

    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                text_content += page.get_text() + "\n"
    except Exception:
        # if PyMuPDF fails we still proceed with bytes
        pass

    # If almost no text found, treat as scanned
    if len(text_content.strip()) < 50:
        is_scanned = True
        text_content = ""

    return text_content, file_bytes, is_scanned

# --- OPENAI FILE UPLOAD / CACHING ---
def _hash_bytes(b: bytes):
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

def upload_pdf_to_openai(file_bytes: bytes, filename: str = "uploaded.pdf"):
    """
    Uploads PDF bytes to OpenAI Files API and returns file_id.
    Caches the file_id in st.session_state.openai_file_cache keyed by SHA256(file_bytes),
    so repeated uploads of the same PDF reuse the same file on OpenAI.
    """
    if openai_client is None:
        return None, "OpenAI client not configured"

    h = _hash_bytes(file_bytes)
    cache = st.session_state.openai_file_cache
    if h in cache:
        return cache[h], None

    # write to a temp file and upload via client.files.create
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            tmp_name = tmp.name

        # per latest guidance: use purpose="assistants" for Responses + files
        resp = openai_client.files.create(
            file=open(tmp_name, "rb"),
            purpose="assistants"
        )
        # resp should be a file object with 'id' property
        file_id = getattr(resp, "id", None)
        if not file_id and hasattr(resp, "to_dict"):
            file_id = resp.to_dict().get("id")
        cache[h] = file_id
        st.session_state.openai_file_cache = cache
        return file_id, None
    except Exception as e:
        return None, str(e)

# --- HELPERS: GEMINI ---
def build_gemini_history(messages):
    gemini_history = []
    for msg in messages:
        if "Error:" in msg["content"]:
            continue
        role = "user" if msg["role"] == "user" else "model"
        gemini_history.append(types.Content(
            role=role,
            parts=[types.Part.from_text(text=msg["content"])]
        ))
    return gemini_history

def call_gemini(model, api_key, history, sys_instruct, file_bytes=None, is_scanned=False):
    if not api_key:
        return {"success": False, "error": "Gemini API key not configured"}

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(api_version="v1beta")
    )

    # If it's a scan, attach the PDF bytes directly to the LAST user message
    if is_scanned and file_bytes:
        last_content = history[-1]
        last_content.parts.append(
            types.Part.from_bytes(data=file_bytes, mime_type="application/pdf")
        )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=history,
                config=types.GenerateContentConfig(
                    system_instruction=sys_instruct,
                    temperature=0.7,
                    safety_settings=[
                        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                    ]
                )
            )

            if response.text:
                return {
                    "success": True,
                    "content": response.text,
                    "input_tokens": response.usage_metadata.prompt_token_count if response.usage_metadata else 0,
                    "output_tokens": response.usage_metadata.candidates_token_count if response.usage_metadata else 0
                }
            else:
                reason = response.candidates[0].finish_reason if response.candidates else "UNKNOWN"
                return {"success": False, "error": f"BLOCKED. Reason: {reason}"}

        except Exception as e:
            error_str = str(e)
            if ("503" in error_str or "429" in error_str) and attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            return {"success": False, "error": error_str}

# --- ChatGPT-5-mini CALL using OpenAI SDK + file_id ---
def call_gpt5_via_responses(model, client: OpenAI, system_instruction, history_messages, prompt, file_id=None):
    """
    Uses client.responses.create(...) with an input that includes:
    - prior messages (history_messages) as text-only (for conversation continuity)
    - final user message that contains an input_file (if file_id) and the prompt text
    """
    if client is None:
        return {"success": False, "error": "OpenAI client not configured (OPENAI_API_KEY missing)"}

    # build input list
    input_items = []

    # add previous conversation items (if any)
    for m in history_messages:
        role = m["role"]  # 'user' or 'assistant'
        text_part = {"type": "input_text", "text": m["content"]}
        input_items.append({
            "role": role,
            "content": [text_part]
        })

    # build the final user item that contains the file reference (if present) and the prompt
    content_parts = []
    if file_id:
        content_parts.append({"type": "input_file", "file_id": file_id})
    content_parts.append({"type": "input_text", "text": prompt})

    input_items.append({
        "role": "user",
        "content": content_parts
    })

    # call responses.create
    try:
        resp = client.responses.create(
            model=model,
            input=input_items,
            instructions=system_instruction,
            timeout=300
        )
    except Exception as e:
        return {"success": False, "error": str(e)}

    # extract text safely
    text_out = None
    try:
        text_out = getattr(resp, "output_text", None)
        if not text_out:
            data = resp.to_dict() if hasattr(resp, "to_dict") else (resp if isinstance(resp, dict) else None)
            if data:
                out_items = data.get("output", []) or []
                parts = []
                for item in out_items:
                    if isinstance(item, dict):
                        for c in item.get("content", []):
                            if c.get("type") in ("output_text", "text") and c.get("text"):
                                parts.append(c["text"])
                text_out = "\n".join(parts).strip() if parts else "(no textual content returned)"
    except Exception:
        text_out = "(no textual content returned)"

    # extract usage
    input_tokens = 0
    output_tokens = 0
    try:
        data = resp.to_dict() if hasattr(resp, "to_dict") else (resp if isinstance(resp, dict) else None)
        usage = None
        if data:
            usage = data.get("usage") or data.get("meta", {}).get("usage") or None
        if usage and isinstance(usage, dict):
            input_tokens = int(
                usage.get("input_tokens")
                or usage.get("prompt_tokens")
                or usage.get("prompt_token_count")
                or 0
            )
            output_tokens = int(
                usage.get("output_tokens")
                or usage.get("completion_tokens")
                or usage.get("candidates_token_count")
                or 0
            )
    except Exception:
        input_tokens = 0
        output_tokens = 0

    return {
        "success": True,
        "content": text_out,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens
    }

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 1. Intel Feed")
    uploaded_file = st.file_uploader("Upload Fund Fact Sheet (PDF)", type="pdf")

    fact_sheet_text = ""
    fact_sheet_bytes = None
    is_scanned_pdf = False
    openai_file_id = None

    if uploaded_file:
        fact_sheet_text, fact_sheet_bytes, is_scanned_pdf = get_pdf_data(uploaded_file)

        if is_scanned_pdf:
            st.warning("⚠️ Scanned PDF detected!")
            st.caption("• We'll upload the PDF to OpenAI Files API and let GPT-5-mini read it natively (OCR/vision).")
        else:
            st.success(f"Ingested {len(fact_sheet_text)} chars (Text Mode)")

    st.divider()
    st.header("🧠 2. The 'Teacher' Persona")

    default_prompt = """You are a SEBI/AMFI registered investor-facing mutual fund advisor for an investor who lives in India. You need to explain in precise, factful, pleasant, conversational tone about the information provided in a mutual fund factsheet. 

I. Please help an investor understand all required information and financial concepts in a mutual fund factsheet so that the investor can decide suitability in line with SEBI’s disclosure.

II. Please keep the responses as concise but without losing clarity. Generate simplified definitions of complex financial terms, tailored to reading by naive investors and supported by relevant examples. 

III. If any required detail in not present in the document, explicitly state “Not stated in the document”. Refrain from answering questions that are out of syllabus from the mutual fund factsheets.

IV. Please cover all the concepts needed to understand mutual fund factsheets, even if the investor forgets to ask."""

    system_prompt = st.text_area("System Instructions", value=default_prompt, height=250)

    st.divider()
    st.header("💰 Session 'Bill'")

    def calc_cost(tokens, input_cost, output_cost):
        return ((tokens['input'] / 1e6) * input_cost) + ((tokens['output'] / 1e6) * output_cost)

    cost_gpt5 = calc_cost(st.session_state.tokens_gpt5, COST_GPT5_INPUT, COST_GPT5_OUTPUT)
    st.markdown(f"**🟢 {MODEL_GPT5} (ChatGPT-5-mini)**")
    st.markdown(f"💵 **${cost_gpt5:.4f}**")

    st.divider()

    cost_b = calc_cost(st.session_state.tokens_b, COST_B_INPUT, COST_B_OUTPUT)
    st.markdown(f"**🔵 {MODEL_B}**")
    st.markdown(f"💵 **${cost_b:.4f}**")

    if st.button("Reset Conversation"):
        for k in ["messages_gpt5", "messages_b", "tokens_gpt5", "tokens_b", "openai_file_cache"]:
            if k in st.session_state:
                if "messages" in k:
                    st.session_state[k] = []
                if "tokens" in k:
                    st.session_state[k] = {"input": 0, "output": 0}
                if k == "openai_file_cache":
                    st.session_state[k] = {}
        st.rerun()

# --- UI ---
st.title("🛡️ Conversational Risk Arena (GPT-5-mini vs Gemini)")

col1, col2 = st.columns(2)

with col1:
    st.subheader(f"🟢 {MODEL_GPT5} (ChatGPT-5-mini)")
    for msg in st.session_state.messages_gpt5:
        with st.chat_message(msg["role"]):
            if "Error" in msg["content"]:
                st.error(msg["content"])
            else:
                st.markdown(msg["content"])

with col2:
    st.subheader(f"🔵 {MODEL_B}")
    for msg in st.session_state.messages_b:
        with st.chat_message(msg["role"]):
            if "Error" in msg["content"]:
                st.error(msg["content"])
            else:
                st.markdown(msg["content"])

# --- INPUT ---
if prompt := st.chat_input("Ask a question..."):
    if not uploaded_file:
        st.error("⚠️ Please upload a PDF first!")
    else:
        # echo prompt to both side before processing
        for c in [col1, col2]:
            with c:
                with st.chat_message("user"):
                    st.markdown(prompt)

        # If we have raw bytes, upload to OpenAI (or get cached id)
        if fact_sheet_bytes:
            with st.spinner("Uploading PDF to OpenAI (cached) ..."):
                file_id, err = upload_pdf_to_openai(
                    fact_sheet_bytes,
                    filename=getattr(uploaded_file, "name", "uploaded.pdf")
                )
            if err:
                st.sidebar.error(f"OpenAI upload error: {err}")
                file_id = None
            else:
                st.sidebar.success(f"Uploaded to OpenAI as {file_id}")
        else:
            file_id = None

        # Prepare system instruction: include extracted text if available (non-scanned)
        if not is_scanned_pdf and fact_sheet_text:
            gpt5_system = f"{system_prompt}\n\nDOCUMENT CONTEXT:\n{fact_sheet_text}"
        else:
            # If scanned or no text - rely on file input (OpenAI will extract text+images)
            gpt5_system = f"{system_prompt}\n\n[DOCUMENT PROVIDED AS PDF file through file_id. Use that document only. If detail is missing, say 'Not stated in the document'.]"

        # Prepare GPT-5-mini history
        history_gpt5 = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages_gpt5]

        # Prepare Gemini history for duel
        history_b_sdk = build_gemini_history(st.session_state.messages_b)
        history_b_sdk.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))

        if not is_scanned_pdf and fact_sheet_text:
            gemini_sys_instruct = f"{system_prompt}\n\nDOCUMENT CONTEXT:\n{fact_sheet_text}"
        else:
            gemini_sys_instruct = system_prompt

        # Run both concurrently: GPT-5-mini gets file_id (OpenAI-side PDF processing), Gemini as before
        with st.spinner("⚔️ Models are dueling..."):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_gpt5 = executor.submit(
                    call_gpt5_via_responses,
                    MODEL_GPT5,
                    openai_client,
                    gpt5_system,
                    history_gpt5,
                    prompt,
                    file_id
                )

                if GEMINI_API_KEY:
                    future_b = executor.submit(
                        call_gemini,
                        MODEL_B,
                        GEMINI_API_KEY,
                        history_b_sdk,
                        gemini_sys_instruct,
                        fact_sheet_bytes,
                        is_scanned_pdf
                    )
                else:
                    future_b = None

                result_gpt5 = future_gpt5.result()
                result_b = future_b.result() if future_b else None

        # update helper
        def update_state(result, msgs_key, tokens_key):
            if result is None:
                return
            st.session_state[msgs_key].append({'role': 'user', 'content': prompt})
            if result.get("success"):
                st.session_state[msgs_key].append({'role': 'assistant', 'content': result["content"]})
                st.session_state[tokens_key]['input'] += int(result.get("input_tokens", 0) or 0)
                st.session_state[tokens_key]['output'] += int(result.get("output_tokens", 0) or 0)
            else:
                st.session_state[msgs_key].append(
                    {'role': 'assistant', 'content': f"⚠️ Error: {result.get('error', 'Unknown error')}"}
                )

        update_state(result_gpt5, "messages_gpt5", "tokens_gpt5")
        update_state(result_b, "messages_b", "tokens_b")

        # render assistant outputs
        with col1:
            last_msg = st.session_state.messages_gpt5[-1]
            with st.chat_message(last_msg["role"]):
                if "Error" in last_msg["content"]:
                    st.error(last_msg["content"])
                else:
                    st.markdown(last_msg["content"])

        with col2:
            last_msg = st.session_state.messages_b[-1]
            with st.chat_message(last_msg["role"]):
                if "Error" in last_msg["content"]:
                    st.error(last_msg["content"])
                else:
                    st.markdown(last_msg["content"])

# --- BOTTOM: Tokens & Cost Summary ---
st.markdown("---")
st.subheader("Usage summary")

col_t1, col_t2, col_t3 = st.columns(3)

with col_t1:
    st.markdown(f"**{MODEL_GPT5} (ChatGPT-5-mini)**")
    st.markdown(f"- Input tokens: `{st.session_state.tokens_gpt5['input']}`")
    st.markdown(f"- Output tokens: `{st.session_state.tokens_gpt5['output']}`")
    est_cost_gpt5 = ((st.session_state.tokens_gpt5['input'] / 1e6) * COST_GPT5_INPUT) + \
                    ((st.session_state.tokens_gpt5['output'] / 1e6) * COST_GPT5_OUTPUT)
    st.markdown(f"- Estimated cost: **${est_cost_gpt5:.6f}**")

with col_t2:
    st.markdown(f"**{MODEL_B}**")
    st.markdown(f"- Input tokens: `{st.session_state.tokens_b['input']}`")
    st.markdown(f"- Output tokens: `{st.session_state.tokens_b['output']}`")
    est_cost_b = ((st.session_state.tokens_b['input'] / 1e6) * COST_B_INPUT) + \
                 ((st.session_state.tokens_b['output'] / 1e6) * COST_B_OUTPUT)
    st.markdown(f"- Estimated cost: **${est_cost_b:.6f}**")

with col_t3:
    total_tokens_in = st.session_state.tokens_gpt5['input'] + st.session_state.tokens_b['input']
    total_tokens_out = st.session_state.tokens_gpt5['output'] + st.session_state.tokens_b['output']
    total_cost = est_cost_gpt5 + est_cost_b
    st.markdown("**Total**")
    st.markdown(f"- Input tokens: `{total_tokens_in}`")
    st.markdown(f"- Output tokens: `{total_tokens_out}`")
    st.markdown(f"- Estimated cost: **${total_cost:.6f}**")

st.caption("Note: Token & cost numbers are estimates — update COST_* constants to match your actual pricing.")
