# Google Interview Scheduling Pipeline

An end-to-end automation flow that sends Google Forms invitations, collects
responses from Google Sheets, schedules interviews via Google Calendar, and
sends Gmail confirmations with Meet links.

## Prerequisites

- Python 3.11+
- A Google Cloud project with OAuth client credentials (desktop application)
- Enabled APIs: Gmail, Google Sheets, Google Calendar
- `credentials.json` downloaded into the project root (path configurable)

## Environment Variables

Populate `.env` (or export the variables) with the following values:

```
GOOGLE_CREDENTIALS_FILE=credentials.json
GOOGLE_TOKEN_FILE=token.json
GOOGLE_DEFAULT_TIMEZONE=UTC

GMAIL_SENDER_ADDRESS=your.email@example.com
GMAIL_TEMPLATE_PATH=templates/invitation_email.html
GMAIL_CONFIRMATION_TEMPLATE_PATH=templates/confirmation_email.html

GOOGLE_SHEET_ID=your_sheet_id
GOOGLE_SHEET_RANGE=FormResponses!A:G
DEFAULT_INTERVIEWER_EMAIL=interviewer@example.com

LOG_LEVEL=INFO
LOG_DIR=logs
```

> Tip: the file `google_scheduler/.env` contains the same keys with placeholder
> values for quick reference.

## Running the Pipeline

Install dependencies and execute the orchestration script:

```bash
pip install -r requirements.txt
python -m google_scheduler.main \
  --forms-link "https://forms.gle/your-form" \
  --candidates-csv data/candidates.csv \
  --job-title "AI Engineer"
```

- Invitations are sent (if both `--forms-link` and `--candidates-csv` are
  supplied).
- Responses are pulled from the configured Google Sheet.
- The scheduler assigns 30-minute slots within the preferred blocks and creates
  Google Calendar events.
- Gmail confirmation emails with Meet links go out automatically.

## CSV Format for Invitations

The CSV supplied via `--candidates-csv` should include at least:

| Column | Purpose |
| ------ | ------- |
| `email` | Candidate email address |
| `name`  | Candidate full name |
| `phone` | Optional contact number |

Additional columns are ignored.

## Google Sheet Expectations

The sheet referenced by `GOOGLE_SHEET_ID` should match the Google Form
responses. The scheduler expects these columns (case-sensitive):

- `Full Name`
- `Email ID`
- `Phone Number`
- `Preferred date of interview`
- `Preferred Time Slot`
- `Any Concerns?`

The service will append/update the following columns as interviews are booked:

- `Scheduled`
- `Assigned Date`
- `Assigned Start Time`
- `Assigned End Time`
- `Calendar Event ID`
- `Meet Link`
- `Status`

## OAuth Tokens

On the first run, the script launches a browser-based OAuth consent flow. The
resulting token is stored at `GOOGLE_TOKEN_FILE` and reused until it expires. If
necessary, delete the token file to re-authenticate.

## Logging

Logs are written to `LOG_DIR` (default `logs/`) with rotation enabled. Console
logging mirrors the same output.

## Extending the Pipeline

- Adjust `INTERVIEW_DURATION_MINUTES` in
  `google_scheduler/services/scheduler.py` to change slot length.
- Provide custom HTML templates and point the environment variables to them.
- For multiple interviewers, update `DEFAULT_INTERVIEWER_EMAIL` before running
  the scheduler or pass a different address into
  `SchedulingService.schedule_pending_interviews`.
