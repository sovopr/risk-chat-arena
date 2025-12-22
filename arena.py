import streamlit as st
import time
import concurrent.futures
import io
import fitz  # PyMuPDF
import tempfile
import hashlib
import uuid
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
from openai import OpenAI
from google import genai
from google.genai import types

st.set_page_config(page_title="Risk Chat Arena", layout="wide")

# --- CONFIGURATION ---
MODEL_A = "gemini-3-flash-preview" # Model A (New - Column 1)
MODEL_B = "gemini-3-pro-preview"   # Model B (Column 2)
MODEL_GPT5_1 = "gpt-5.2"           # Model C (Column 3)
MAX_TURNS = 30                     # Max questions per session
GOOGLE_SHEET_NAME = "RiskArenaLogs" # Name of your Google Sheet

# ✅ API KEYS
CHATGPT5_API_KEY = st.secrets.get("OPENAI_API_KEY") 
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

# --- COST ASSUMPTIONS (per 1M tokens) ---
COST_A_INPUT = 0.00
COST_A_OUTPUT = 0.00
COST_B_INPUT = 0.00
COST_B_OUTPUT = 0.00
COST_GPT5_1_INPUT = 0.00
COST_GPT5_1_OUTPUT = 0.00

# --- SDK CLIENTS ---
openai_client = OpenAI(api_key=CHATGPT5_API_KEY) if CHATGPT5_API_KEY else None

# --- SESSION STATE INITIALIZATION ---
def init_state(key, default):
    if key not in st.session_state:
        st.session_state[key] = default

# Model Histories
init_state("messages_a", [])
init_state("tokens_a", {"input": 0, "output": 0})

init_state("messages_b", [])
init_state("tokens_b", {"input": 0, "output": 0})

init_state("messages_gpt5_1", [])
init_state("tokens_gpt5_1", {"input": 0, "output": 0})

# File Caching
init_state("openai_file_cache", {})

# User Logic
init_state("user_id", str(uuid.uuid4())) # Unique ID for this session
init_state("turn_count", 0)              # Counts questions asked (0 to 30)

# --- GOOGLE SHEETS LOGGING ---
def get_gspread_client():
    """
    Authenticates with Google Sheets using the single JSON key in secrets.
    """
    try:
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]
        
        # Load credentials from st.secrets dictionary
        if "gcp_service_account" not in st.secrets:
            st.error("❌ Missing [gcp_service_account] in secrets.")
            return None

        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        st.error(f"❌ Google Auth Error: {e}")
        return None

def log_to_google_sheet(question, r_flash, r_gemini, r_gpt5_1):
    """
    Appends the interaction data to the Google Sheet.
    """
    client = get_gspread_client()
    if not client:
        return

    try:
        # Open the spreadsheet by name
        sheet = client.open(GOOGLE_SHEET_NAME).sheet1
        
        # Check if headers exist (simple check: is A1 empty?)
        try:
            val = sheet.acell('A1').value
            if not val:
                # Updated Headers
                headers = ["User ID", "Turn Number", "Timestamp", "Question", "Gemini-3-Flash", "Gemini-3-Pro", "GPT-5.2"]
                sheet.append_row(headers)
        except:
            pass
        
        # Prepare row data
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [
            st.session_state.user_id,
            st.session_state.turn_count,
            timestamp,
            question,
            r_flash,
            r_gemini,
            r_gpt5_1
        ]
        
        # Append the row
        sheet.append_row(row)
        
    except Exception as e:
        st.warning(f"⚠️ Data Log Warning: Could not save to Google Sheet. ({e})")

# --- PDF HELPERS ---
def get_pdf_data(uploaded_file):
    text_content = ""
    file_bytes = uploaded_file.getvalue()
    is_scanned = False

    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                text_content += page.get_text() + "\n"
    except Exception:
        pass

    if len(text_content.strip()) < 50:
        is_scanned = True
        text_content = ""

    return text_content, file_bytes, is_scanned

def _hash_bytes(b: bytes):
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

def upload_pdf_to_openai(file_bytes: bytes, filename: str = "uploaded.pdf"):
    if openai_client is None:
        return None, "OpenAI client not configured"

    h = _hash_bytes(file_bytes)
    cache = st.session_state.openai_file_cache
    if h in cache:
        return cache[h], None

    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            tmp_name = tmp.name

        resp = openai_client.files.create(
            file=open(tmp_name, "rb"),
            purpose="assistants"
        )
        file_id = getattr(resp, "id", None)
        if not file_id and hasattr(resp, "to_dict"):
            file_id = resp.to_dict().get("id")
        
        cache[h] = file_id
        st.session_state.openai_file_cache = cache
        return file_id, None
    except Exception as e:
        return None, str(e)

# --- GEMINI CLIENT ---
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

    if is_scanned and file_bytes:
        last_content = history[-1]
        last_content.parts.append(
            types.Part.from_bytes(data=file_bytes, mime_type="application/pdf")
        )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            # UPDATED: Added thinking_config with thinking_level="low"
            # This applies to both Flash and Pro as requested
            response = client.models.generate_content(
                model=model,
                contents=history,
                config=types.GenerateContentConfig(
                    system_instruction=sys_instruct,
                    temperature=0.7,
                    thinking_config=types.ThinkingConfig(thinking_level="low"),
                    safety_settings=[
                        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
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
            if ("503" in str(e) or "429" in str(e)) and attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            return {"success": False, "error": str(e)}

# --- OPENAI CLIENT (Standard Chat Completions with 120s Timeout) ---
def call_gpt5_via_responses(model, client: OpenAI, system_instruction, history_messages, prompt, file_id=None):
    if client is None:
        return {"success": False, "error": "OpenAI client not configured"}

    # 1. Build the standard messages list
    messages = [
        {"role": "system", "content": system_instruction}
    ]

    # 2. Add History
    for m in history_messages:
        messages.append({"role": m["role"], "content": str(m["content"])})

    # 3. Add Current User Prompt
    messages.append({"role": "user", "content": prompt})

    try:
        # Standard Chat Completion Call with increased timeout
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            timeout=120 
        )
        
        content = resp.choices[0].message.content
        input_tokens = resp.usage.prompt_tokens if resp.usage else 0
        output_tokens = resp.usage.completion_tokens if resp.usage else 0

        return {
            "success": True, 
            "content": content, 
            "input_tokens": input_tokens, 
            "output_tokens": output_tokens
        }

    except Exception as e:
        return {"success": False, "error": str(e)}

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
            st.caption("• We'll upload the PDF to OpenAI Files API.")
        else:
            st.success(f"Ingested {len(fact_sheet_text)} chars (Text Mode)")

    st.divider()

    # --- PROGRESS TRACKER ---
    st.header("⏳ Progress")
    
    progress_value = min(st.session_state.turn_count / MAX_TURNS, 1.0)
    st.progress(progress_value)
    
    col_p1, col_p2 = st.columns([3, 1])
    with col_p1:
        st.write(f"Questions Used:")
    with col_p2:
        st.write(f"**{st.session_state.turn_count}/{MAX_TURNS}**")
        
    if st.session_state.turn_count >= MAX_TURNS:
        st.error("🛑 Limit Reached. Please reset.")

    st.divider()
    st.header("🧠 2. The 'Teacher' Persona")

    default_prompt = """
You are tasked with the role of a SEBI/AMFI registered mutual fund advisor for a novice investor who lives in India. Your job is to communicate about the information provided in a mutual fund factsheet (you shall be provided with a factsheet), such that the investor feels confident, satisfied, develops trust with the advisor and makes informed investment decisions. Particularly, you shall respond to the investor’s questions using the following list of communication rules.

**List of communication rules**
Simplify complexities by translating jargon into everyday language, prioritize transparency by explicitly disclosing risks and costs upfront, and personalize the data by directly linking scheme features to the client's specific life goals—so the factsheet becomes easy to comprehend.
Rule 1: Decode Jargon into Plain Language
Novice investors often feel intimidated by technical terms found in factsheets, such as "Modified Duration" or "Sharpe Ratio." The advisor must translate these into relatable concepts. For instance, instead of simply stating the "Expense Ratio," explain it as the "annual fee paid to the fund house for managing the money." When discussing the "Exit Load," describe it as a "penalty for withdrawing money too early," ensuring the client understands the liquidity constraints without getting lost in terminology.
Rule 2: Lead with Transparency on Risk and Cost
Building trust requires discussing potential downsides before selling the upside. Advisors should explicitly point out the Riskometer on the factsheet and explain what the specific risk level (e.g., "Very High") means for their capital. Clearly communicating the "Total Expense Ratio (TER)" and any potential tax implications ensures the investor is not surprised by deductions later. This honest approach mitigates mis-selling risks and fosters long-term confidence.
Rule 3: Personalize by Linking Features to Goals
A factsheet contains generic data; the advisor’s job is to make it relevant to the client’s life in Udupi. Connect specific scheme features to the investor's personal financial goals. For example, if the factsheet shows a "3-year lock-in period" (as in ELSS funds), explain how this aligns with their medium-term goal, like saving for a home renovation, while also providing tax benefits. This contextualization shifts the conversation from abstract numbers to tangible life outcomes.
Rule 4: Utilize Visuals and Check for Understanding
Visual aids are powerful tools for explanation. Advisors should use the charts and graphs present in the factsheet—such as the "NAV movement" or "Portfolio Allocation" pie chart—to visually demonstrate where the money is being invested. Crucially, the advisor must practice active listening and frequently pause to ask, "Does this make sense?" or "How do you feel about this risk level?" to verify the novice investor truly comprehends the information rather than just nodding along.
Operational rules for every response:
•   Stay within the mutual fund factsheet (and references inside it, e.g., SID/SAI/KIM if mentioned). Do not answer questions outside the scope of the mutual fund factsheet. If asked, say: “Out of scope for factsheet-based discussion” Or “Not stated in the document.”
•   Keep answers clear and concise. Include a brief real-world example only when it improves understanding (no hype, no guarantees).
•   Do not wait for the investor to ask about features of the mutual fund. Proactively identify and explain essential concepts from the factsheet—to ensure the investor has the foundational knowledge needed to make an informed decision.
"""

    system_prompt = st.text_area("System Instructions", value=default_prompt, height=250)

    st.divider()
    st.header("💰 Session 'Bill'")

    def calc_cost(tokens, input_cost, output_cost):
        return ((tokens['input'] / 1e6) * input_cost) + ((tokens['output'] / 1e6) * output_cost)

    # 1. Gemini Flash
    cost_a = calc_cost(st.session_state.tokens_a, COST_A_INPUT, COST_A_OUTPUT)
    st.markdown(f"**⚡ {MODEL_A}**")
    st.markdown(f"💵 **${cost_a:.4f}**")
    st.markdown("---")

    # 2. Gemini Pro
    cost_b = calc_cost(st.session_state.tokens_b, COST_B_INPUT, COST_B_OUTPUT)
    st.markdown(f"**🔵 {MODEL_B}**")
    st.markdown(f"💵 **${cost_b:.4f}**")
    st.markdown("---")

    # 3. GPT-5.1
    cost_gpt5_1 = calc_cost(st.session_state.tokens_gpt5_1, COST_GPT5_1_INPUT, COST_GPT5_1_OUTPUT)
    st.markdown(f"**🟣 {MODEL_GPT5_1}**")
    st.markdown(f"💵 **${cost_gpt5_1:.4f}**")

    if st.button("Reset Conversation"):
        keys_to_reset = [
            "messages_a", "tokens_a",
            "messages_b", "tokens_b",
            "messages_gpt5_1", "tokens_gpt5_1",
            "openai_file_cache", "turn_count"
        ]
        for k in keys_to_reset:
            if k in st.session_state:
                if "messages" in k: st.session_state[k] = []
                if "tokens" in k: st.session_state[k] = {"input": 0, "output": 0}
                if k == "openai_file_cache": st.session_state[k] = {}
                if k == "turn_count": st.session_state[k] = 0
        
        st.session_state.user_id = str(uuid.uuid4())
        st.rerun()

# --- UI (UPDATED: THREE MODEL LAYOUT) ---
st.title("🛡️ Conversational Risk Arena (Triple-Model Duel)")

# Iterate through history by turn (User -> 3 Assistants)
# We use messages_b as the master list for length (assuming synchronization)
if "messages_b" in st.session_state and len(st.session_state.messages_b) > 0:
    for i in range(0, len(st.session_state.messages_b), 2):
        
        # 1. DISPLAY USER QUESTION (Full Width)
        user_content = st.session_state.messages_b[i]["content"]
        with st.chat_message("user"):
            st.markdown(user_content)
            
        # 2. DISPLAY 3 MODEL ANSWERS (Columns below the question)
        if i + 1 < len(st.session_state.messages_b):
            c1, c2, c3 = st.columns(3)
            
            # Model A (Flash)
            with c1:
                st.subheader(f"⚡ {MODEL_A}")
                if i+1 < len(st.session_state.messages_a):
                    msg_a = st.session_state.messages_a[i+1]
                    with st.chat_message(msg_a["role"]):
                        if "Error" in msg_a["content"]: st.error(msg_a["content"])
                        else: st.markdown(msg_a["content"])

            # Model B (Pro)
            with c2:
                st.subheader(f"🔵 {MODEL_B}")
                msg_b = st.session_state.messages_b[i+1]
                with st.chat_message(msg_b["role"]):
                     if "Error" in msg_b["content"]: st.error(msg_b["content"])
                     else: st.markdown(msg_b["content"])
            
            # Model C (GPT 5.2)
            with c3:
                st.subheader(f"🟣 {MODEL_GPT5_1}")
                if i+1 < len(st.session_state.messages_gpt5_1):
                    msg_c = st.session_state.messages_gpt5_1[i+1]
                    with st.chat_message(msg_c["role"]):
                         if "Error" in msg_c["content"]: st.error(msg_c["content"])
                         else: st.markdown(msg_c["content"])
            
            st.divider() # Clean separation for the next turn

# --- INPUT LOGIC ---
if st.session_state.turn_count < MAX_TURNS:
    prompt = st.chat_input(f"Ask question ({st.session_state.turn_count + 1}/{MAX_TURNS})...")
else:
    prompt = None
    st.warning("🛑 You have reached the 30-question limit for this session. Please Reset to start over.")

if prompt:
    if not uploaded_file:
        st.error("⚠️ Please upload a PDF first!")
    else:
        # 1. INCREMENT COUNTER
        st.session_state.turn_count += 1

        # 2. ECHO PROMPT TEMPORARILY (Full Width)
        with st.chat_message("user"):
            st.markdown(prompt)

        # 3. UPLOAD FILE IF NEEDED
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

        # 4. SYSTEM PROMPTS
        if not is_scanned_pdf and fact_sheet_text:
            gpt_system = f"{system_prompt}\n\nDOCUMENT CONTEXT:\n{fact_sheet_text}"
        else:
            gpt_system = f"{system_prompt}\n\n[DOCUMENT PROVIDED AS PDF file. Use that document only.]"

        # Prepare Histories
        history_gpt5_1 = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages_gpt5_1]

        # History for Model A (Flash)
        history_a_sdk = build_gemini_history(st.session_state.messages_a)
        history_a_sdk.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))

        # History for Model B (Pro)
        history_b_sdk = build_gemini_history(st.session_state.messages_b)
        history_b_sdk.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))
        
        if not is_scanned_pdf and fact_sheet_text:
            gemini_sys_instruct = f"{system_prompt}\n\nDOCUMENT CONTEXT:\n{fact_sheet_text}"
        else:
            gemini_sys_instruct = system_prompt

        # 5. RUN MODELS (Show spinners in aligned columns)
        c1, c2, c3 = st.columns(3)
        with c1: st.subheader(f"⚡ {MODEL_A}")
        with c2: st.subheader(f"🔵 {MODEL_B}")
        with c3: st.subheader(f"🟣 {MODEL_GPT5_1}")

        with st.spinner("⚔️ 3 Models are dueling..."):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Gemini Flash (Model A)
                if GEMINI_API_KEY:
                    future_a = executor.submit(
                        call_gemini, MODEL_A, GEMINI_API_KEY, history_a_sdk, gemini_sys_instruct, fact_sheet_bytes, is_scanned_pdf
                    )
                else:
                    future_a = None

                # Gemini Pro (Model B)
                if GEMINI_API_KEY:
                    future_b = executor.submit(
                        call_gemini, MODEL_B, GEMINI_API_KEY, history_b_sdk, gemini_sys_instruct, fact_sheet_bytes, is_scanned_pdf
                    )
                else:
                    future_b = None
                
                # GPT-5.1 (Model C)
                future_gpt5_1 = executor.submit(
                    call_gpt5_via_responses, MODEL_GPT5_1, openai_client, gpt_system, history_gpt5_1, prompt, file_id
                )

                result_a = future_a.result() if future_a else None
                result_b = future_b.result() if future_b else None
                result_gpt5_1 = future_gpt5_1.result()

        # 6. UPDATE STATE
        def update_state(result, msgs_key, tokens_key):
            if result is None:
                return "(model unavailable)"
            st.session_state[msgs_key].append({'role': 'user', 'content': prompt})
            if result.get("success"):
                content = result["content"]
                st.session_state[msgs_key].append({'role': 'assistant', 'content': content})
                st.session_state[tokens_key]['input'] += int(result.get("input_tokens", 0) or 0)
                st.session_state[tokens_key]['output'] += int(result.get("output_tokens", 0) or 0)
                return content
            else:
                err_msg = f"⚠️ Error: {result.get('error', 'Unknown error')}"
                st.session_state[msgs_key].append({'role': 'assistant', 'content': err_msg})
                return err_msg

        txt_flash = update_state(result_a, "messages_a", "tokens_a")
        txt_gemini = update_state(result_b, "messages_b", "tokens_b")
        txt_gpt5_1 = update_state(result_gpt5_1, "messages_gpt5_1", "tokens_gpt5_1")

        # 7. LOG TO GOOGLE SHEETS
        log_to_google_sheet(prompt, txt_flash, txt_gemini, txt_gpt5_1)

        # 8. RERUN TO UPDATE UI (Layout will fix itself here)
        st.rerun()

# --- BOTTOM: COST SUMMARY ---
st.markdown("---")
st.subheader("Usage summary")

col_t1, col_t2, col_t3 = st.columns(3)

with col_t1:
    st.markdown(f"**{MODEL_A}**")
    st.markdown(f"- In: `{st.session_state.tokens_a['input']}` / Out: `{st.session_state.tokens_a['output']}`")
    est_cost_a = ((st.session_state.tokens_a['input'] / 1e6) * COST_A_INPUT) + \
                 ((st.session_state.tokens_a['output'] / 1e6) * COST_A_OUTPUT)
    st.markdown(f"- Cost: **${est_cost_a:.6f}**")

with col_t2:
    st.markdown(f"**{MODEL_B}**")
    st.markdown(f"- In: `{st.session_state.tokens_b['input']}` / Out: `{st.session_state.tokens_b['output']}`")
    est_cost_b = ((st.session_state.tokens_b['input'] / 1e6) * COST_B_INPUT) + \
                 ((st.session_state.tokens_b['output'] / 1e6) * COST_B_OUTPUT)
    st.markdown(f"- Cost: **${est_cost_b:.6f}**")

with col_t3:
    st.markdown(f"**{MODEL_GPT5_1}**")
    st.markdown(f"- In: `{st.session_state.tokens_gpt5_1['input']}` / Out: `{st.session_state.tokens_gpt5_1['output']}`")
    est_cost_gpt5_1 = ((st.session_state.tokens_gpt5_1['input'] / 1e6) * COST_GPT5_1_INPUT) + \
                      ((st.session_state.tokens_gpt5_1['output'] / 1e6) * COST_GPT5_1_OUTPUT)
    st.markdown(f"- Cost: **${est_cost_gpt5_1:.6f}**")

st.markdown("---")
total_cost = est_cost_a + est_cost_b + est_cost_gpt5_1
st.markdown(f"### 📦 Grand Total Cost: ${total_cost:.6f}")