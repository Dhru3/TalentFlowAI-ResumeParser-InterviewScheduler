# TalentFlow AI 🚀

**Your one-stop solution for smarter, faster hiring.** Curate roles, rank talent, and schedule interviews—all in one place.

End-to-end recruitment platform powered by Azure OpenAI for intelligent resume parsing and semantic matching, with Google Calendar integration for automated interview scheduling.

---

## ⚡ Quick Setup & Run

### Step 1: Install Dependencies

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install all required packages
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Step 2: Set Up Google Calendar (for Interview Scheduling) (MY CREDENTIALS SHOULD WORK, IF IT DOESNT DO THIS STEP:)

**Get OAuth Credentials:**
1. Visit [Google Cloud Console](https://console.cloud.google.com/)
2. Create/select a project
3. Enable **Google Calendar API** and **Gmail API**
4. Go to "Credentials" → "Create OAuth 2.0 Client ID" → Select **Web Application**
5. Download `credentials.json` and place it in the project root

**Configure File Paths:**

Place `credentials.json` in your project root. The app will generate `token.json` automatically on first run.

```bash
# Optional: Set custom paths in .env (defaults to project root)
 GOOGLE_CREDENTIALS_FILE="./credentials.json"
 GOOGLE_TOKEN_FILE="./token.json"
```

**First-Time Authentication:**
- On first run, a browser window will open
- Log in with your Google account and authorize the app
- Credentials are saved to `token.json` for future use

### Step 3: Launch the Application

```bash
streamlit run app.py
```

The dashboard opens at **http://localhost:8501** 🎉

---

## 🎯 Complete Workflow

### 1️⃣ **Upload Job Descriptions**
- Navigate to the **Job Descriptions** section
- Upload `.txt` files or paste JD text directly
- Save to session library for reuse

### 2️⃣ **Parse & Rank Resumes**
- Go to **Resumes & Matching** tab
- Upload PDF resumes (supports bulk upload)
- App automatically:
  - Extracts structured data (name, email, skills, experience)
  - Caches parsed data for faster re-runs
  - Generates AI insights for each candidate
- Select a job description to rank candidates
- View match scores and detailed breakdowns

### 3️⃣ **Schedule Interviews**
- Go to **Interview Scheduling** tab
- Select candidates from ranked list
- Choose available time slots
- Click **Schedule Interviews**
- App automatically:
  - Creates Google Calendar events
  - Generates Google Meet links
  - Sends email invitations to candidates
  - Shows confirmation with event links

### 4️⃣ **Track & Manage**
- View scheduled interviews in the dashboard
- Check candidate availability responses
- Monitor session stats (resumes parsed, interviews scheduled)

---

## 📁 Project Structure

```
TalentFlow-AI/
├── app.py                    # Main Streamlit application
├── credentials.json          # Google OAuth credentials (you provide)
├── token.json                # Auto-generated auth token (after first login)
├── requirements.txt          # Python dependencies
├── parser.py                 # Resume parsing logic
├── matcher.py                # Semantic matching with Azure OpenAI
├── internal_talent_pool.py   # Internal candidate database
├── scheduler.py              # Google Calendar integration
├── data/                     # Place resume PDFs here
├── cache/                    # Auto-generated parsed resume cache
└── google_scheduler/         # Modular scheduling pipeline
    ├── services/
    │   ├── calendar_service.py   # Google Calendar API
    │   ├── gmail_service.py      # Gmail API for emails
    │   └── pipeline.py           # End-to-end scheduling flow
    └── templates/                # Email templates
```

---

## � How It Works

### Resume Parsing
- Uses **Azure OpenAI** (GPT models) to extract structured data from unstructured PDFs
- Extracts: name, email, phone, skills, experience, education, summary
- Results cached in `cache/` folder (JSON format) to avoid re-processing

### Semantic Matching
- Generates embeddings for job descriptions and resumes using **Azure OpenAI**
- Computes cosine similarity to rank candidates
- Provides AI-generated insights explaining why each candidate is a good/poor match

### Interview Scheduling
- Integrates with **Google Calendar API** to create events
- Auto-generates **Google Meet** video conference links
- Sends email invitations via **Gmail API**
- Tracks candidate availability responses

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| **Azure OpenAI errors** | Verify `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY` are set correctly |
| **Google auth fails** | Ensure `credentials.json` is in project root and APIs are enabled in Google Cloud Console |
| **Token expired** | Delete `token.json` and re-authenticate |
| **Resumes not parsing** | Check PDFs are in `data/` folder and contain readable text (not scanned images) |
| **Module not found** | Run `pip install -r requirements.txt` again |

---

## 💡 Key Features

✅ **Smart Resume Parsing** – Extract structured data from any resume format  
✅ **AI-Powered Matching** – Rank candidates by semantic fit, not just keywords  
✅ **Auto Scheduling** – One-click interview invites with Google Meet links  
✅ **Email Automation** – Send professional invitations via Gmail  
✅ **Intelligent Caching** – Instant re-runs without re-processing resumes  
✅ **Beautiful Dashboard** – Modern, intuitive Streamlit UI  
✅ **Session Management** – Track everything in one place  

---

## 📝 Environment Variables Reference

```bash
# Azure OpenAI (Required)
AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
AZURE_OPENAI_API_KEY="your-key"
AZURE_OPENAI_EMBEDDING_MODEL="text-embedding-3-large"
AZURE_OPENAI_CHAT_MODEL="gpt-4o-mini"
AZURE_OPENAI_API_VERSION="2024-12-01-preview"  # Optional

# Google Calendar/Gmail (Required for scheduling)
GOOGLE_CREDENTIALS_FILE="./credentials.json"  # OAuth client credentials
GOOGLE_TOKEN_FILE="./token.json"              # Auto-generated auth token

# Optional
DEFAULT_INTERVIEWER_EMAIL="recruiter@company.com"
GOOGLE_FORM_LINK="https://forms.gle/your-form"  # For availability collection
```

---

**Built with ❤️ for smarter hiring** | Powered by Azure OpenAI & Google Workspace
