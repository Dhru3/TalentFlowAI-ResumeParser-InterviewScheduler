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

def main():
    project_dir = Path(__file__).parent
    
    print("=" * 60)
    print("🚀 STREAMLIT CLOUD SECRETS GENERATOR")
    print("=" * 60)
    print()
    
    # Check for required files
    credentials_path = project_dir / "credentials.json"
    token_path = project_dir / "token.json"
    env_path = project_dir / ".env"
    
    secrets_toml = []
    secrets_toml.append("# ============================================")
    secrets_toml.append("# PASTE THIS INTO STREAMLIT CLOUD SECRETS")
    secrets_toml.append("# ============================================")
    secrets_toml.append("")
    
    # Add Groq API key
    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key and env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.startswith("GROQ_API_KEY="):
                    groq_key = line.split("=", 1)[1].strip().strip('"')
                    break
    
    if groq_key:
        secrets_toml.append(f'GROQ_API_KEY = "{groq_key}"')
    else:
        secrets_toml.append('GROQ_API_KEY = "YOUR_GROQ_API_KEY_HERE"')
        print("⚠️  GROQ_API_KEY not found in .env - you'll need to add it manually")
    
    secrets_toml.append('GROQ_MODEL = "llama-3.1-70b-versatile"')
    secrets_toml.append('EMBEDDING_MODEL = "all-MiniLM-L6-v2"')
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
    secrets_toml.append('GOOGLE_DEFAULT_TIMEZONE = "Asia/Kolkata"')
    
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
