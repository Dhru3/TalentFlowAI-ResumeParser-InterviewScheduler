# generate_token.py
import json
from google_auth_oauthlib.flow import InstalledAppFlow

# SCOPES used by the pipeline
SCOPES = [
  'https://www.googleapis.com/auth/spreadsheets',
  'https://www.googleapis.com/auth/calendar',
  'https://www.googleapis.com/auth/gmail.send',            # if you want to write back scheduled slot                # read/write calendar, create events
  'https://www.googleapis.com/auth/calendar.events',
  "https://www.googleapis.com/auth/drive",            # event-level           # send emails
  'https://www.googleapis.com/auth/forms.responses.readonly' 
]

def main():
    flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
    creds = flow.run_local_server(port=8080, prompt='consent', access_type='offline')
    # creds is google.oauth2.credentials.Credentials
    with open('token.json', 'w') as f:
        f.write(creds.to_json())
    print("token.json created. Keep it private.")

if __name__ == '__main__':
    main()
