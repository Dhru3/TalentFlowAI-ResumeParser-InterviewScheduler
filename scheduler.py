"""
Google Calendar-based interview scheduler with OAuth 2.0 authentication.
Replaces the Microsoft Graph API implementation with Google Calendar API.
"""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)

# Google Calendar API scopes
SCOPES=[
  "https://www.googleapis.com/auth/spreadsheets",
  "https://www.googleapis.com/auth/calendar", 
  "https://www.googleapis.com/auth/gmail.send", 
  "https://www.googleapis.com/auth/calendar.events",
  "https://www.googleapis.com/auth/forms.responses.readonly"
]

@dataclass
class GoogleCalendarCredentials:
    """Configuration for Google Calendar API authentication."""
    credentials_file: str  # Path to OAuth client credentials JSON
    token_file: str = "token.pickle"  # Where to cache OAuth tokens
    default_timezone: str = "America/New_York"  # Default timezone for events

    @classmethod
    def from_env(cls) -> "GoogleCalendarCredentials":
        """Create credentials from environment variables."""
        try:
            credentials_file = os.environ["GOOGLE_CREDENTIALS_FILE"]
        except KeyError as exc:
            raise RuntimeError(
                "Missing required environment variable: GOOGLE_CREDENTIALS_FILE. "
                "Please set this to the path of your Google OAuth client credentials JSON file."
            ) from exc
        
        token_file = os.getenv("GOOGLE_TOKEN_FILE", "token.pickle")
        default_timezone = os.getenv("GOOGLE_DEFAULT_TIMEZONE", "Asia/Kolkata")
        
        return cls(
            credentials_file=credentials_file,
            token_file=token_file,
            default_timezone=default_timezone,
        )


def get_calendar_service(config: GoogleCalendarCredentials):
    """
    Authenticate with Google Calendar API and return service object.
    
    Uses OAuth 2.0 flow with token caching. On first run, opens browser
    for user consent. Subsequent runs use cached refresh token.
    
    Args:
        config: GoogleCalendarCredentials with paths and settings
        
    Returns:
        Google Calendar API service object
        
    Raises:
        FileNotFoundError: If credentials_file doesn't exist
        Exception: If authentication fails
    """
    creds = None
    
    # Check if we have cached tokens
    if os.path.exists(config.token_file):
        logger.info(f"Loading cached credentials from {config.token_file}")
        with open(config.token_file, 'rb') as token:
            creds = pickle.load(token)
    
    # If no valid credentials, authenticate
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            logger.info("Refreshing expired credentials")
            creds.refresh(Request())
        else:
            if not os.path.exists(config.credentials_file):
                raise FileNotFoundError(
                    f"Google OAuth credentials file not found: {config.credentials_file}\n"
                    "Please download from Google Cloud Console and save to this path."
                )
            
            logger.info("Starting OAuth flow - browser will open")
            flow = InstalledAppFlow.from_client_secrets_file(
                config.credentials_file, SCOPES
            )
            creds = flow.run_local_server(port=0)
        
        # Save credentials for next run
        logger.info(f"Saving credentials to {config.token_file}")
        with open(config.token_file, 'wb') as token:
            pickle.dump(creds, token)
    
    # Build and return Calendar service
    service = build('calendar', 'v3', credentials=creds)
    logger.info("Successfully authenticated with Google Calendar API")
    return service


def schedule_interview(
    candidate_email: str,
    interviewer_email: str,
    time_slot,  # Can be tuple of (start, end) or dict with 'start'/'end'
    *,
    subject: Optional[str] = None,
    description: str = "",
    location: str = "Google Meet",
    config: Optional[GoogleCalendarCredentials] = None,
    additional_attendees: Optional[list[str]] = None,
    make_online_meeting: bool = True,
    send_email: bool = False,
    **kwargs  # Accept and ignore other parameters for compatibility
) -> dict:
    """
    Schedule an interview on Google Calendar with Google Meet video conference.
    
    This function signature is compatible with the old Microsoft Graph version
    for easy migration.
    
    Args:
        candidate_email: Email address of the candidate
        interviewer_email: Email address of the interviewer
        time_slot: Either a tuple of (start_datetime, end_datetime) or a dict with 'start'/'end' keys
        subject: Optional subject line for the calendar event
        description: Event description/notes
        location: Location string (ignored when using Google Meet)
        config: GoogleCalendarCredentials for authentication (or will use from_env())
        additional_attendees: Optional list of additional email addresses
        make_online_meeting: Whether to create Google Meet link (always True for Google Calendar)
        send_email: Whether to send email invitations (handled by Google Calendar API)
        **kwargs: Additional parameters for compatibility (ignored)
        
    Returns:
        dict with event details including 'event_id', 'web_link', 'online_meeting' with 'joinUrl'
        
    Raises:
        HttpError: If Google Calendar API call fails
    """
    try:
        config = config or GoogleCalendarCredentials.from_env()
        service = get_calendar_service(config)
        
        # Parse time_slot into start/end datetimes
        if isinstance(time_slot, (tuple, list)):
            start_time, end_time = time_slot
        elif isinstance(time_slot, dict):
            start_time = time_slot['start']
            end_time = time_slot['end']
        else:
            raise ValueError("time_slot must be tuple/list or dict with start/end")
        
        # Convert string times to datetime if needed
        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        if isinstance(end_time, str):
            end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        
        # Build event summary
        summary = subject or f"Interview with {candidate_email}"
        
        # Build event description
        description_text = description or "Interview scheduled via automation."
        
        # Build attendee list
        attendees = [
            {'email': candidate_email},
            {'email': interviewer_email},
        ]
        if additional_attendees:
            attendees.extend({'email': email} for email in additional_attendees)
        
        # Create event body
        event = {
            'summary': summary,
            'description': description_text,
            'start': {
                'dateTime': start_time.isoformat(),
                'timeZone': config.default_timezone,
            },
            'end': {
                'dateTime': end_time.isoformat(),
                'timeZone': config.default_timezone,
            },
            'attendees': attendees,
            'conferenceData': {
                'createRequest': {
                    'requestId': f"interview-{candidate_email}-{int(start_time.timestamp())}",
                    'conferenceSolutionKey': {'type': 'hangoutsMeet'}
                }
            } if make_online_meeting else None,
            'reminders': {
                'useDefault': False,
                'overrides': [
                    {'method': 'email', 'minutes': 24 * 60},  # 1 day before
                    {'method': 'popup', 'minutes': 30},  # 30 minutes before
                ],
            },
            'guestsCanModify': False,
            'guestsCanInviteOthers': False,
        }
        
        # Remove None conferenceData if not making online meeting
        if not make_online_meeting:
            del event['conferenceData']
        
        # Insert event with conference data (Google Meet)
        logger.info(f"Creating calendar event for {candidate_email} at {start_time}")
        created_event = service.events().insert(
            calendarId='primary',
            body=event,
            conferenceDataVersion=1 if make_online_meeting else 0,
            sendUpdates='all' if send_email else 'none'
        ).execute()
        
        logger.info(f"Event created: {created_event.get('htmlLink')}")
        if make_online_meeting:
            logger.info(f"Google Meet link: {created_event.get('hangoutLink')}")
        
        # Return in format compatible with old Graph API version
        return {
            'event_id': created_event['id'],
            'web_link': created_event['htmlLink'],
            'online_meeting': {
                'joinUrl': created_event.get('hangoutLink')
            } if make_online_meeting else None,
            'start': created_event['start'],
            'end': created_event['end'],
        }
        
    except HttpError as error:
        logger.error(f"Failed to create calendar event: {error}")
        raise RuntimeError(f"Failed to create event: {error}") from error
    except Exception as error:
        logger.error(f"Unexpected error scheduling interview: {error}")
        raise


def delete_event(config: GoogleCalendarCredentials, event_id: str) -> None:
    """
    Delete a calendar event by ID.
    
    Args:
        config: GoogleCalendarCredentials for authentication
        event_id: ID of the event to delete
        
    Raises:
        HttpError: If deletion fails
    """
    try:
        service = get_calendar_service(config)
        service.events().delete(
            calendarId='primary',
            eventId=event_id,
            sendUpdates='all'  # Notify attendees
        ).execute()
        logger.info(f"Event {event_id} deleted successfully")
    except HttpError as error:
        logger.error(f"Failed to delete event: {error}")
        raise


def get_upcoming_events(
    config: GoogleCalendarCredentials,
    max_results: int = 10
) -> list[dict]:
    """
    Get upcoming calendar events.
    
    Args:
        config: GoogleCalendarCredentials for authentication
        max_results: Maximum number of events to retrieve
        
    Returns:
        List of event dictionaries
        
    Raises:
        HttpError: If API call fails
    """
    try:
        service = get_calendar_service(config)
        now = datetime.utcnow().isoformat() + 'Z'  # 'Z' indicates UTC time
        
        events_result = service.events().list(
            calendarId='primary',
            timeMin=now,
            maxResults=max_results,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        events = events_result.get('items', [])
        logger.info(f"Retrieved {len(events)} upcoming events")
        return events
        
    except HttpError as error:
        logger.error(f"Failed to retrieve events: {error}")
        raise
