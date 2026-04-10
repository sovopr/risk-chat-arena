
```markdown
# Risk Chat Arena

A Streamlit-based UI for benchmarking LLMs against each other in real-time. I built this to test how different models handle financial document analysis—specifically, evaluating Gemini 3 Flash, Gemini 3.1 Flash-Lite, and Claude 3.5 Haiku on a mutual fund factsheet.

## Features
* **Triple-Model Duel:** Runs user prompts concurrently against Gemini and Claude to compare output quality and generation speed side-by-side.
* **Context Caching:** Implements caching for both Google (via the File API) and Anthropic (prompt caching) to save on tokens and latency when repeatedly querying the same PDF.
* **Live Cost Tracking:** Calculates and displays the exact session cost (accounting for input, output, and cache read/write costs) for each model in the sidebar.
* **Auto-Logging:** Dumps all prompts, timestamps, and model responses directly into a Google Sheet (`RiskArenaLogs`) for later evaluation.

## Setup

You'll need API keys for Anthropic and Google GenAI, plus a GCP Service Account JSON for the Google Sheets integration.

1. Clone the repo.
2. Create a `.streamlit/secrets.toml` file. The app expects the following structure:
   ```toml
   ANTHROPIC_API_KEY = "your-anthropic-key"
   GEMINI_API_KEY = "your-google-key"

   [gcp_service_account]
   type = "service_account"
   project_id = "..."
   private_key_id = "..."
   private_key = "..."
   client_email = "..."
   client_id = "..."
   auth_uri = "[https://accounts.google.com/o/oauth2/auth](https://accounts.google.com/o/oauth2/auth)"
   token_uri = "[https://oauth2.googleapis.com/token](https://oauth2.googleapis.com/token)"
   auth_provider_x509_cert_url = "[https://www.googleapis.com/oauth2/v1/certs](https://www.googleapis.com/oauth2/v1/certs)"
   client_x509_cert_url = "..."
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: Core dependencies include `streamlit`, `google-genai`, `anthropic`, and `gspread`.)*

### Devcontainers
If you use VS Code or GitHub Codespaces, a `.devcontainer` configuration is included. It automatically provisions a Python 3.11 environment, installs the `requirements.txt`, sets up the necessary VS Code extensions, and forwards port 8501. The Streamlit app will launch automatically on `postAttach`.

## Usage

If you aren't using the devcontainer, start the app locally with:
```bash
streamlit run arena.py
```

By default, the app loads the Kotak Flexicap Fund factsheet and applies a "Teacher" persona system prompt. This instructs the models to explain financial concepts (like risk-o-meters and expense ratios) to novice Indian investors using SEBI guidelines. 

The session has a hardcoded limit of 30 conversational turns to prevent accidental runaway costs. Once you hit the limit, just click "Reset Conversation" in the sidebar to clear the history and start fresh.
