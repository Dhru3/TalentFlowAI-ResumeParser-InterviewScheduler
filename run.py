from google_auth_oauthlib.flow import InstalledAppFlow
import pickle
import os
from pathlib import Path

SCOPES=[
  "https://www.googleapis.com/auth/spreadsheets",
  "https://www.googleapis.com/auth/calendar", 
  "https://www.googleapis.com/auth/gmail.send",
  "https://www.googleapis.com/auth/drive", 
  "https://www.googleapis.com/auth/calendar.events",
  "https://www.googleapis.com/auth/forms.responses.readonly"
]


def find_credentials():
  env = os.environ.get('GOOGLE_CREDENTIALS_FILE')
  if env:
    p = Path(env).expanduser()
    if p.exists():
      return str(p)

  p = Path(__file__).resolve()
  for parent in [p] + list(p.parents):
    cand = Path(parent) / 'credentials.json'
    if cand.exists():
      return str(cand)

  cwd_cand = Path('credentials.json')
  if cwd_cand.exists():
    return str(cwd_cand.resolve())

  raise FileNotFoundError(
    "credentials.json not found. Set GOOGLE_CREDENTIALS_FILE env var or place credentials.json in project root."
  )


creds = None
if os.path.exists("token.pickle"):
  with open("token.pickle", "rb") as token:
    creds = pickle.load(token)

if not creds or not creds.valid:
  cred_path = find_credentials()
  flow = InstalledAppFlow.from_client_secrets_file(cred_path, SCOPES)
  creds = flow.run_local_server(port=8080)
  with open("token.pickle", "wb") as token:
    pickle.dump(creds, token)

print("✅ Token created successfully!")
