# 🚀 Complete Deployment Guide: TalentFlow AI to Streamlit Cloud

This is a **complete step-by-step guide** to host your app on Streamlit Community Cloud so your professor can access it without you running anything.

---

## 📋 Pre-flight Checklist

Before starting, make sure you have:
- [x] GitHub account (you said you created a new repo)
- [x] Your project working locally with `credentials.json` and `token.json`
- [x] Streamlit account (free at [share.streamlit.io](https://share.streamlit.io))

---

## 🔧 PART 1: Prepare Your Project

### Step 1.1: Generate Base64 Secrets

Open Terminal in your project folder and run:

```bash
cd /Users/dhrutipurushotham/Documents/Projects/TalentFlowAI
source .venv/bin/activate
python bootstrap_secrets.py
```

This outputs two long base64 strings. **Copy them to a text file** — you'll need them later:
- `GOOGLE_CREDENTIALS_JSON_B64` → the credentials.json encoded
- `GOOGLE_TOKEN_JSON_B64` → the token.json encoded

### Step 1.2: Verify .gitignore

Your `.gitignore` should exclude these files (already done):
```
credentials.json
token.json
token.json.backup
.env
.streamlit/secrets.toml
```

---

## 📤 PART 2: Push to GitHub

### Step 2.1: Initialize Git (if not already)

```bash
cd /Users/dhrutipurushotham/Documents/Projects/TalentFlowAI

# Check if git is initialized
git status

# If not initialized:
git init
```

### Step 2.2: Add Your GitHub Remote

Replace `YOUR_USERNAME` and `YOUR_REPO_NAME` with your actual values:

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
```

If remote already exists, update it:
```bash
git remote set-url origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
```

### Step 2.3: Commit and Push

```bash
# Stage all files (secrets are excluded by .gitignore)
git add .

# Commit
git commit -m "Prepare for Streamlit Cloud deployment"

# Push to GitHub
git branch -M main
git push -u origin main
```

### Step 2.4: Verify on GitHub

Go to your GitHub repo in browser and confirm:
- ✅ `app.py` is there
- ✅ `bootstrap_secrets.py` is there
- ✅ `requirements.txt` is there
- ✅ `google_scheduler/templates/` folder with HTML files is there
- ❌ `credentials.json` is NOT there
- ❌ `token.json` is NOT there
- ❌ `.env` is NOT there

---

## ☁️ PART 3: Deploy on Streamlit Cloud

### Step 3.1: Go to Streamlit Cloud

1. Open [share.streamlit.io](https://share.streamlit.io)
2. Click **"Sign in"** → Sign in with GitHub
3. Authorize Streamlit to access your repos

### Step 3.2: Create New App

1. Click **"New app"** button
2. Fill in:
   - **Repository**: Select your repo (e.g., `YOUR_USERNAME/YOUR_REPO_NAME`)
   - **Branch**: `main`
   - **Main file path**: `app.py`

### Step 3.3: Add Secrets (IMPORTANT!)

1. Click **"Advanced settings"**
2. Click the **"Secrets"** tab
3. Paste this entire block (with YOUR actual values):

```toml
# ============================================
# GROQ LLM CONFIGURATION
# Get your free API key at: https://console.groq.com
# ============================================
GROQ_API_KEY = "your-groq-api-key-here"
GROQ_MODEL = "llama-3.1-70b-versatile"

# ============================================
# EMBEDDING MODEL (HuggingFace - runs locally, free!)
# ============================================
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ============================================
# GOOGLE OAUTH CREDENTIALS (base64 encoded)
# Paste the values from Step 1.1
# ============================================
GOOGLE_CREDENTIALS_JSON_B64 = "PASTE_YOUR_CREDENTIALS_BASE64_HERE"
GOOGLE_TOKEN_JSON_B64 = "PASTE_YOUR_TOKEN_BASE64_HERE"

# ============================================
# GOOGLE SCHEDULER CONFIGURATION
# Use RELATIVE paths - they work on cloud!
# ============================================
GOOGLE_CREDENTIALS_FILE = "credentials.json"
GOOGLE_TOKEN_FILE = "token.json"
GOOGLE_DEFAULT_TIMEZONE = "Asia/Kolkata"

# Gmail settings
GMAIL_SENDER_ADDRESS = "dhrutipurushotham@gmail.com"
GMAIL_TEMPLATE_PATH = "google_scheduler/templates/invitation_email.html"
GMAIL_CONFIRMATION_TEMPLATE_PATH = "google_scheduler/templates/confirmation_email.html"

# Google Forms/Sheets
GOOGLE_FORM_LINK = "https://forms.gle/joeVDGQ26ywNqm4w5"
GOOGLE_SHEET_ID = "1_lNF6qZ4i6Shbd2Oi7r-ipZuENIRKhV95iJqqo-7818"
GOOGLE_SHEET_RANGE = "Form_Responses_1!A1:G"

# Default interviewer
DEFAULT_INTERVIEWER_EMAIL = "dhrutipurushotham@gmail.com"
INTERVIEWER_EMAIL = "dhrutipurushotham@gmail.com"
```

**⚠️ IMPORTANT**: Replace `PASTE_YOUR_CREDENTIALS_BASE64_HERE` and `PASTE_YOUR_TOKEN_BASE64_HERE` with the actual base64 strings from Step 1.1!

### Step 3.4: Deploy

1. Click **"Deploy!"**
2. Wait 2-5 minutes for build
3. You'll get a URL like: `https://your-app-name.streamlit.app`

---

## 🧪 PART 4: Test Your Deployment

1. Open the Streamlit URL in browser
2. Try uploading a resume
3. Try scheduling an interview
4. Check that emails send correctly

If something fails, click **"Manage app"** → **"Logs"** to see errors.

---

## 🔑 Key Points About Paths

Your `.env` file has **absolute paths** like:
```
GMAIL_TEMPLATE_PATH="/Users/dhrutipurushotham/Documents/Projects/TalentFlowAI/google_scheduler/templates/invitation_email.html"
```

On Streamlit Cloud, use **relative paths** instead:
```
GMAIL_TEMPLATE_PATH = "google_scheduler/templates/invitation_email.html"
```

The `settings.py` code already handles this — it resolves paths relative to `Path.cwd()` (the project root).

---

## ❓ FAQ

### Will my professor need Google credentials?
**No.** The app uses YOUR Google account (via `token.json`). Your professor just opens the URL and uses the app.

### Does the token expire hourly?
**No.** The `token.json` contains a refresh token that auto-renews access tokens. You only need to regenerate if:
- You revoke access in Google Security settings
- Google expires the refresh token (rare, usually months)

### What LLM and embedding models are used?
- **LLM:** Groq Cloud API with `llama-3.1-70b-versatile` (fast, free tier available)
- **Embeddings:** HuggingFace `sentence-transformers` with `all-MiniLM-L6-v2` (runs locally, completely free)

### Can I update the app after deployment?
Yes! Just push changes to GitHub:
```bash
git add .
git commit -m "Update something"
git push
```
Streamlit Cloud auto-redeploys within a minute.

---

## 🛠️ Troubleshooting

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError` | Check `requirements.txt` has all packages |
| `GOOGLE_CREDENTIALS_FILE must be set` | Add it to Streamlit Secrets |
| `Invalid base64` | Re-run `python bootstrap_secrets.py` and copy fresh |
| `Token has been expired or revoked` | Re-auth locally, regenerate token.json, update base64 secret |
| `File not found: templates/...` | Use relative path in secrets |

---

## 🧹 After the Demo

1. Go to Streamlit Cloud → Delete the app
2. Go to [Google Security](https://myaccount.google.com/permissions) → Revoke "TalentFlow AI" access
3. Optionally rotate your Azure API key

---

## 📝 Quick Command Reference

```bash
# Generate base64 secrets
python bootstrap_secrets.py

# Test locally
streamlit run app.py

# Push updates
git add . && git commit -m "Update" && git push

# Encode a single file
python -c "import base64; print(base64.b64encode(open('credentials.json','rb').read()).decode())"
```

---

**You're all set!** 🎉 Share the Streamlit URL with your professor and the app will work without you being online.

