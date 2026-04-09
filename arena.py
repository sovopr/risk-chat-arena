import streamlit as st
import time
import concurrent.futures
import uuid
import gspread
import base64
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import anthropic
from google import genai
from google.genai import types
import os

st.set_page_config(page_title="Risk Chat Arena", layout="wide")

# --- CONFIGURATION ---
MODEL_A = "gemini-3-flash-preview"             # Model A (Column 1)
MODEL_B = "gemini-3.1-flash-lite-preview"      # Model B (Column 2)
MODEL_CLAUDE = "claude-haiku-4-5-20251001"    # Model C (Column 3)
MAX_TURNS = 30                         # Max questions per session
GOOGLE_SHEET_NAME = "RiskArenaLogs"    # Name of your Google Sheet

# ✅ API KEYS
ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY") 
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

# --- COST ASSUMPTIONS (per 1M tokens) ---
COST_A_INPUT = 0.00
COST_A_OUTPUT = 0.00
COST_B_INPUT = 0.00
COST_B_OUTPUT = 0.00
COST_CLAUDE_INPUT = 0.00
COST_CLAUDE_OUTPUT = 0.00

# --- SDK CLIENTS ---
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

# --- SESSION STATE INITIALIZATION ---
def init_state(key, default):
    if key not in st.session_state:
        st.session_state[key] = default

# Model Histories
init_state("messages_a", [])
init_state("tokens_a", {"input": 0, "output": 0})

init_state("messages_b", [])
init_state("tokens_b", {"input": 0, "output": 0})

init_state("messages_claude", [])
init_state("tokens_claude", {"input": 0, "output": 0})

# User Logic
init_state("user_id", str(uuid.uuid4())) # Unique ID for this session
init_state("turn_count", 0)              # Counts questions asked (0 to 30)

# --- GOOGLE SHEETS LOGGING ---
def get_gspread_client():
    try:
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]
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

def log_to_google_sheet(question, r_flash, r_gemini, r_claude):
    client = get_gspread_client()
    if not client:
        return

    try:
        sheet = client.open(GOOGLE_SHEET_NAME).sheet1
        try:
            val = sheet.acell('A1').value
            if not val:
                headers = ["User ID", "Turn Number", "Timestamp", "Question", "Gemini-3-Flash", "Gemini-3.1-Lite", "Claude-Haiku"]
                sheet.append_row(headers)
        except:
            pass
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [
            st.session_state.user_id,
            st.session_state.turn_count,
            timestamp,
            question,
            r_flash,
            r_gemini,
            r_claude
        ]
        sheet.append_row(row)
    except Exception as e:
        st.warning(f"⚠️ Data Log Warning: Could not save to Google Sheet. ({e})")

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

def call_gemini(model, api_key, history, sys_instruct, file_bytes=None):
    if not api_key:
        return {"success": False, "error": "Gemini API key not configured"}

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(api_version="v1beta")
    )

    if file_bytes and len(history) > 0 and history[-1].role == "user":
        history[-1].parts.append(
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

# --- CLAUDE CLIENT ---
def call_claude(model, client: anthropic.Anthropic, system_instruction, history_messages, prompt, file_bytes=None):
    if client is None:
        return {"success": False, "error": "Anthropic client not configured"}

    messages = []
    for m in history_messages:
        messages.append({"role": m["role"], "content": str(m["content"])})

    user_content = []
    
    # Pass the native PDF via base64 encoding to Anthropic
    if file_bytes:
        pdf_data = base64.b64encode(file_bytes).decode("utf-8")
        user_content.append({
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": pdf_data
            }
        })
        
    user_content.append({"type": "text", "text": prompt})
    messages.append({"role": "user", "content": user_content})

    try:
        resp = client.messages.create(
            model=model,
            system=system_instruction,
            messages=messages,
            max_tokens=2048,
            extra_headers={"anthropic-beta": "pdfs-2024-09-25"}
        )
        
        content = resp.content[0].text
        input_tokens = resp.usage.input_tokens
        output_tokens = resp.usage.output_tokens

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
    st.header("📂 1. Kotak Flexicap")
    
    uploaded_file = os.path.join(os.path.dirname(__file__), "KOTAK FLEXICAP FUND.pdf")
    fact_sheet_bytes = None

    try:
        with open(uploaded_file, "rb") as f:
            fact_sheet_bytes = f.read()
        st.success("PDF loaded natively (No extraction required).")
    except FileNotFoundError:
        st.error(f"File not found: {uploaded_file}")

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
You are a mutual fund expert in India. Your task is to analyse the attached mutual fund factsheet and explain it to a novice investor who lives in India, ensuring the investor understands the scheme's features, feels confident, and makes informed decisions. Please respond to the investor’s queries using the following communication guidelines.

* Communication Guidelines*
1. Simplify & Educate
Use plain language and relatable real-world examples. Define technical terms immediately. 
2. Focus on Key Information First
Prioritize fund’s objective, performance and risk concepts and suitability, following SEBI’s emphasis on disclosing fundamental attributes upfront.
3. Visual Risk Communication
Use SEBI's color-coded risk-o-meter system to help investors visually understand risk levels (low to very high) rather than relying on complex numerical indicators.
4. Explain Costs Transparently
Clearly distinguish between expense ratios, exit loads, and differences between direct and regular plans, as mandated by SEBI's disclosure requirements.
5. Put Performance in Context
Present historical returns across multiple timeframes while emphasizing that past performance doesn't guarantee future results and comparing against appropriate benchmarks.
6. Assess Suitability
Connect fund features to the investor's financial goals, time horizon, and risk appetite rather than just presenting data, following AMFI's fiduciary standards.
7. Disclose All Material Information
Ensure completeness by covering portfolio holdings, fund manager details, asset allocation, and any recent changes as required by SEBI regulations.
8. Encourage Questions
Create an open environment where investors feel comfortable asking for clarification on any aspect they don't understand, supporting informed decision-making.

Operational rules for every response:
-*Strict Scope:* Stick to the factsheet. If information is missing/irrelevant, reply: "Out of scope for factsheet-based discussion" or "Not stated in the document." 
-Proactive Education: Do not wait for questions. Proactively identify and explain critical concepts (e.g., Sharpe ratio, CAGR, Beta, Benchmark returns, Volatility, AUM, Taxation, etc.) to build a foundation. 
-Brevity: Keep responses concise and action-oriented.
-**Engagement:** If the user doesn't continue the conversation after a brief pause, proactively ask follow-up questions on critical concepts.
"""

    system_prompt = st.text_area("System Instructions", value=default_prompt, height=250)

    st.divider()
    st.header("💰 Session 'Bill'")

    def calc_cost(tokens, input_cost, output_cost):
        return ((tokens['input'] / 1e6) * input_cost) + ((tokens['output'] / 1e6) * output_cost)

    cost_a = calc_cost(st.session_state.tokens_a, COST_A_INPUT, COST_A_OUTPUT)
    st.markdown(f"**⚡ {MODEL_A}**")
    st.markdown(f"💵 **${cost_a:.4f}**")
    st.markdown("---")

    cost_b = calc_cost(st.session_state.tokens_b, COST_B_INPUT, COST_B_OUTPUT)
    st.markdown(f"**🔵 {MODEL_B}**")
    st.markdown(f"💵 **${cost_b:.4f}**")
    st.markdown("---")

    cost_claude = calc_cost(st.session_state.tokens_claude, COST_CLAUDE_INPUT, COST_CLAUDE_OUTPUT)
    st.markdown(f"**🟣 {MODEL_CLAUDE}**")
    st.markdown(f"💵 **${cost_claude:.4f}**")

    if st.button("Reset Conversation"):
        keys_to_reset = [
            "messages_a", "tokens_a",
            "messages_b", "tokens_b",
            "messages_claude", "tokens_claude",
            "turn_count"
        ]
        for k in keys_to_reset:
            if k in st.session_state:
                if "messages" in k: st.session_state[k] = []
                if "tokens" in k: st.session_state[k] = {"input": 0, "output": 0}
                if k == "turn_count": st.session_state[k] = 0
        
        st.session_state.user_id = str(uuid.uuid4())
        st.rerun()

# --- UI (THREE MODEL LAYOUT) ---
st.title("🛡️ Conversational Risk Arena (Triple-Model Duel)")

if "messages_b" in st.session_state and len(st.session_state.messages_b) > 0:
    for i in range(0, len(st.session_state.messages_b), 2):
        
        user_content = st.session_state.messages_b[i]["content"]
        with st.chat_message("user"):
            st.markdown(user_content)
            
        if i + 1 < len(st.session_state.messages_b):
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.subheader(f"⚡ {MODEL_A}")
                if i+1 < len(st.session_state.messages_a):
                    msg_a = st.session_state.messages_a[i+1]
                    with st.chat_message(msg_a["role"]):
                        if "Error" in msg_a["content"]: st.error(msg_a["content"])
                        else: st.markdown(msg_a["content"])

            with c2:
                st.subheader(f"🔵 {MODEL_B}")
                msg_b = st.session_state.messages_b[i+1]
                with st.chat_message(msg_b["role"]):
                     if "Error" in msg_b["content"]: st.error(msg_b["content"])
                     else: st.markdown(msg_b["content"])
            
            with c3:
                st.subheader(f"🟣 {MODEL_CLAUDE}")
                if i+1 < len(st.session_state.messages_claude):
                    msg_c = st.session_state.messages_claude[i+1]
                    with st.chat_message(msg_c["role"]):
                         if "Error" in msg_c["content"]: st.error(msg_c["content"])
                         else: st.markdown(msg_c["content"])
            
            st.divider()

# --- INPUT LOGIC ---
if st.session_state.turn_count < MAX_TURNS:
    prompt = st.chat_input(f"Ask question ({st.session_state.turn_count + 1}/{MAX_TURNS})...")
else:
    prompt = None
    st.warning("🛑 You have reached the 30-question limit for this session. Please Reset to start over.")

if prompt:
    st.session_state.turn_count += 1

    with st.chat_message("user"):
        st.markdown(prompt)

    history_claude = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages_claude]

    history_a_sdk = build_gemini_history(st.session_state.messages_a)
    history_a_sdk.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))

    history_b_sdk = build_gemini_history(st.session_state.messages_b)
    history_b_sdk.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))

    c1, c2, c3 = st.columns(3)
    with c1: st.subheader(f"⚡ {MODEL_A}")
    with c2: st.subheader(f"🔵 {MODEL_B}")
    with c3: st.subheader(f"🟣 {MODEL_CLAUDE}")

    with st.spinner("⚔️ 3 Models are dueling natively with the PDF..."):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            if GEMINI_API_KEY:
                future_a = executor.submit(
                    call_gemini, MODEL_A, GEMINI_API_KEY, history_a_sdk, system_prompt, fact_sheet_bytes
                )
                future_b = executor.submit(
                    call_gemini, MODEL_B, GEMINI_API_KEY, history_b_sdk, system_prompt, fact_sheet_bytes
                )
            else:
                future_a = None
                future_b = None
            
            future_claude = executor.submit(
                call_claude, MODEL_CLAUDE, anthropic_client, system_prompt, history_claude, prompt, fact_sheet_bytes
            )

            result_a = future_a.result() if future_a else None
            result_b = future_b.result() if future_b else None
            result_claude = future_claude.result()

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
    txt_claude = update_state(result_claude, "messages_claude", "tokens_claude")

    log_to_google_sheet(prompt, txt_flash, txt_gemini, txt_claude)
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
    st.markdown(f"**{MODEL_CLAUDE}**")
    st.markdown(f"- In: `{st.session_state.tokens_claude['input']}` / Out: `{st.session_state.tokens_claude['output']}`")
    est_cost_claude = ((st.session_state.tokens_claude['input'] / 1e6) * COST_CLAUDE_INPUT) + \
                      ((st.session_state.tokens_claude['output'] / 1e6) * COST_CLAUDE_OUTPUT)
    st.markdown(f"- Cost: **${est_cost_claude:.6f}**")

st.markdown("---")
total_cost = est_cost_a + est_cost_b + est_cost_claude
st.markdown(f"### 📦 Grand Total Cost: ${total_cost:.6f}")
