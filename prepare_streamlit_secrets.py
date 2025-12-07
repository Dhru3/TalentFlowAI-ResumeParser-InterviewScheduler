#!/usr/bin/env python3
"""
Prepare secrets for Streamlit Cloud deployment.
This script encodes your local JSON files to base64 for pasting into Streamlit secrets.

Run: python prepare_streamlit_secrets.py
"""

import base64
import os
from pathlib import Path

def encode_file_to_base64(file_path: str) -> str:
    """Read a file and return its base64-encoded content."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_env_value(env_path: Path, key: str, default: str = "") -> str:
    """Get a value from .env file."""
    if not env_path.exists():
        return default
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(f"{key}="):
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
                return value
    return default

def main():
    project_dir = Path(__file__).parent
    
    print("=" * 60)
    print("🚀 STREAMLIT CLOUD SECRETS GENERATOR")
    print("=" * 60)
    print()
    print("This will generate a MINIMAL secrets file.")
    print("Google features are OPTIONAL - the app works without them!")
    print()
    
    # Check for required files
    credentials_path = project_dir / "credentials.json"
    token_path = project_dir / "token.json"
    env_path = project_dir / ".env"
    
    secrets_toml = []
    secrets_toml.append("# ============================================")
    secrets_toml.append("# STREAMLIT CLOUD SECRETS")
    secrets_toml.append("# ============================================")
    secrets_toml.append("")
    
    # Add Groq API key (REQUIRED)
    secrets_toml.append("# REQUIRED: Groq LLM for resume parsing & AI features")
    groq_key = get_env_value(env_path, "GROQ_API_KEY")
    if groq_key:
        secrets_toml.append(f'GROQ_API_KEY = "{groq_key}"')
        print("✅ GROQ_API_KEY found")
    else:
        secrets_toml.append('GROQ_API_KEY = "YOUR_GROQ_API_KEY_HERE"')
        print("⚠️  GROQ_API_KEY not found - GET ONE FREE at https://console.groq.com")
    
    secrets_toml.append('GROQ_MODEL = "llama-3.1-70b-versatile"')
    secrets_toml.append('EMBEDDING_MODEL = "all-MiniLM-L6-v2"')
    secrets_toml.append("")
    
    # Ask about Google integration
    print()
    include_google = input("Include Google OAuth for email/calendar features? (y/N): ").strip().lower()
    
    if include_google == 'y':
        secrets_toml.append("# ============================================")
        secrets_toml.append("# OPTIONAL: Google Integration")
        secrets_toml.append("# (Remove this section if not using scheduling)")
        secrets_toml.append("# ============================================")
        secrets_toml.append("")
        
        # Encode credentials.json
        if credentials_path.exists():
            creds_b64 = encode_file_to_base64(credentials_path)
            secrets_toml.append(f'GOOGLE_CREDENTIALS_JSON_B64 = "{creds_b64}"')
            print("✅ credentials.json encoded")
        else:
            secrets_toml.append('GOOGLE_CREDENTIALS_JSON_B64 = "YOUR_BASE64_CREDENTIALS_HERE"')
            print("⚠️  credentials.json not found")
        
        # Encode token.json
        if token_path.exists():
            token_b64 = encode_file_to_base64(token_path)
            secrets_toml.append(f'GOOGLE_TOKEN_JSON_B64 = "{token_b64}"')
            print("✅ token.json encoded")
        else:
            secrets_toml.append('GOOGLE_TOKEN_JSON_B64 = "YOUR_BASE64_TOKEN_HERE"')
            print("⚠️  token.json not found - run: python generate_token.py")
        
        secrets_toml.append("")
        secrets_toml.append('GOOGLE_CREDENTIALS_FILE = "credentials.json"')
        secrets_toml.append('GOOGLE_TOKEN_FILE = "token.json"')
        
        # Timezone
        tz = get_env_value(env_path, "GOOGLE_DEFAULT_TIMEZONE", "Asia/Kolkata")
        secrets_toml.append(f'GOOGLE_DEFAULT_TIMEZONE = "{tz}"')
        secrets_toml.append("")
        
        # Gmail configuration
        gmail_sender = get_env_value(env_path, "GMAIL_SENDER_ADDRESS")
        if gmail_sender:
            secrets_toml.append(f'GMAIL_SENDER_ADDRESS = "{gmail_sender}"')
            print("✅ GMAIL_SENDER_ADDRESS found")
        else:
            secrets_toml.append('GMAIL_SENDER_ADDRESS = "your-email@gmail.com"')
            print("⚠️  GMAIL_SENDER_ADDRESS not found")
        
        secrets_toml.append('GMAIL_TEMPLATE_PATH = "google_scheduler/templates/invitation_email.html"')
        secrets_toml.append('GMAIL_CONFIRMATION_TEMPLATE_PATH = "google_scheduler/templates/confirmation_email.html"')
        secrets_toml.append("")
        
        # Google Sheets/Forms
        sheet_id = get_env_value(env_path, "GOOGLE_SHEET_ID")
        if sheet_id:
            secrets_toml.append(f'GOOGLE_SHEET_ID = "{sheet_id}"')
            print("✅ GOOGLE_SHEET_ID found")
        else:
            secrets_toml.append('GOOGLE_SHEET_ID = ""')
        
        sheet_range = get_env_value(env_path, "GOOGLE_SHEET_RANGE", "Form_Responses_1!A1:G")
        secrets_toml.append(f'GOOGLE_SHEET_RANGE = "{sheet_range}"')
        
        form_link = get_env_value(env_path, "GOOGLE_FORM_LINK")
        if form_link:
            secrets_toml.append(f'GOOGLE_FORM_LINK = "{form_link}"')
        else:
            secrets_toml.append('GOOGLE_FORM_LINK = ""')
        
        secrets_toml.append("")
        
        # Interviewer email
        interviewer = get_env_value(env_path, "DEFAULT_INTERVIEWER_EMAIL", gmail_sender)
        if interviewer:
            secrets_toml.append(f'DEFAULT_INTERVIEWER_EMAIL = "{interviewer}"')
        else:
            secrets_toml.append('DEFAULT_INTERVIEWER_EMAIL = ""')
    else:
        secrets_toml.append("")
        secrets_toml.append("# Google Integration disabled - scheduling features won't work")
        secrets_toml.append("# but resume parsing and ranking will work fine!")
        secrets_toml.append('GOOGLE_CREDENTIALS_FILE = ""')
        secrets_toml.append('GOOGLE_TOKEN_FILE = ""')
        print("ℹ️  Skipping Google integration")
    
    # Write to file
    output_path = project_dir / "streamlit_secrets_output.txt"
    with open(output_path, "w") as f:
        f.write("\n".join(secrets_toml))
    
    print()
    print("=" * 60)
    print(f"✅ Secrets saved to: {output_path}")
    print("=" * 60)
    print()
    print("NEXT STEPS:")
    print("1. Open streamlit_secrets_output.txt")
    print("2. Copy the entire content")
    print("3. Go to Streamlit Cloud → Your App → Settings → Secrets")
    print("4. Paste and save")
    print()
    print("⚠️  DELETE streamlit_secrets_output.txt after copying!")
    print("    (It contains your API keys)")

if __name__ == "__main__":
    main()
