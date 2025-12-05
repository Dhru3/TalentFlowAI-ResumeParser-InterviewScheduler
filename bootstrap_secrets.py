"""
Bootstrap secrets for cloud deployment.

This module decodes base64-encoded credentials from environment variables
and writes them to disk so the app can use them normally. Call this BEFORE
importing any modules that need credentials.json or token.json.

Usage:
    import bootstrap_secrets
    bootstrap_secrets.setup()  # Creates credentials.json and token.json from env vars
"""

import base64
import json
import os
from pathlib import Path

# Where to write the credential files (project root)
PROJECT_ROOT = Path(__file__).parent


def _get_secret(key: str) -> str:
    """Get a secret from environment variable or Streamlit secrets."""
    value = os.getenv(key, "").strip()
    if value:
        return value
    
    # Try Streamlit secrets
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and key in st.secrets:
            return str(st.secrets[key]).strip()
    except Exception:
        pass
    
    return ""


def inject_streamlit_secrets_to_env():
    """
    Copy all Streamlit secrets to environment variables.
    This ensures modules that use os.getenv() can find the values.
    """
    try:
        import streamlit as st
        if hasattr(st, 'secrets'):
            for key in st.secrets:
                if key not in os.environ:
                    os.environ[key] = str(st.secrets[key])
            print("✅ Injected Streamlit secrets into environment")
    except Exception as e:
        print(f"ℹ️  Streamlit secrets not available: {e}")


def decode_and_write(env_var: str, output_path: Path, description: str) -> bool:
    """
    Decode a base64-encoded JSON string from an env var and write to file.
    Returns True if successful, False if env var is missing/empty.
    """
    encoded = _get_secret(env_var)
    
    if not encoded:
        print(f"⚠️  {env_var} not set, skipping {description}")
        return False
    
    try:
        decoded = base64.b64decode(encoded).decode("utf-8")
        # Validate it's valid JSON
        json.loads(decoded)
        output_path.write_text(decoded)
        print(f"✅ Created {output_path.name} from {env_var}")
        return True
    except Exception as e:
        print(f"❌ Failed to decode {env_var}: {e}")
        return False


def setup() -> dict:
    """
    Bootstrap all secrets from environment variables.
    Returns a dict with status of each file.
    """
    results = {}
    
    # First, inject Streamlit secrets into environment variables
    # so that settings.py and other modules can use os.getenv()
    inject_streamlit_secrets_to_env()
    
    # Google OAuth credentials (client ID, client secret, etc.)
    credentials_path = PROJECT_ROOT / "credentials.json"
    if not credentials_path.exists():
        results["credentials.json"] = decode_and_write(
            "GOOGLE_CREDENTIALS_JSON_B64",
            credentials_path,
            "Google OAuth credentials"
        )
    else:
        print(f"ℹ️  credentials.json already exists, skipping")
        results["credentials.json"] = True
    
    # Google OAuth token (access token, refresh token)
    token_path = PROJECT_ROOT / "token.json"
    if not token_path.exists():
        results["token.json"] = decode_and_write(
            "GOOGLE_TOKEN_JSON_B64",
            token_path,
            "Google OAuth token"
        )
    else:
        print(f"ℹ️  token.json already exists, skipping")
        results["token.json"] = True
    
    return results


def encode_file_to_base64(file_path: str) -> str:
    """
    Helper function to encode a file to base64 string.
    Use this locally to generate the values for your env vars.
    
    Usage:
        python -c "from bootstrap_secrets import encode_file_to_base64; print(encode_file_to_base64('credentials.json'))"
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"{file_path} not found")
    
    content = path.read_bytes()
    encoded = base64.b64encode(content).decode("utf-8")
    return encoded


if __name__ == "__main__":
    # When run directly, encode both files and print the values
    print("=" * 60)
    print("BASE64 ENCODED SECRETS FOR DEPLOYMENT")
    print("=" * 60)
    print("\nCopy these values to your Streamlit Cloud secrets or .env\n")
    
    for filename in ["credentials.json", "token.json"]:
        path = PROJECT_ROOT / filename
        if path.exists():
            encoded = encode_file_to_base64(str(path))
            print(f"\n{filename.upper().replace('.', '_')}_B64=")
            print("-" * 40)
            print(encoded)
            print("-" * 40)
        else:
            print(f"\n⚠️  {filename} not found, skipping")
    
    print("\n✅ Done! Add these to Streamlit Cloud > App Settings > Secrets")
