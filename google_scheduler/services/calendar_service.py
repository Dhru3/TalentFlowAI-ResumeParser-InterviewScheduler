"""Service wrapper for Google Calendar event creation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, Optional

from googleapiclient.discovery import Resource
from googleapiclient.errors import HttpError

from ..config.settings import Settings
from ..utils.logging_utils import get_logger
from .google_client import GoogleClientFactory, execute_with_retry

logger = get_logger(__name__)

CALENDAR_SCOPES = ["https://www.googleapis.com/auth/calendar"]


@dataclass(slots=True)
class CalendarEventResult:
    event_id: str
    hangout_link: Optional[str]
    start: str
    end: str
    attendees: Iterable[str]


class CalendarService:
    """Wrapper for creating Google Calendar events with Meet links."""

    def __init__(self, settings: Settings, factory: GoogleClientFactory | None = None):
        self.settings = settings
        self._factory = factory or GoogleClientFactory(settings)
        self._service: Resource = self._factory.build("calendar", "v3", CALENDAR_SCOPES)

    def create_interview_event(
        self,
        *,
        candidate_name: str,
        candidate_email: str,
        start_time: datetime,
        end_time: datetime,
        interviewer_email: Optional[str] = None,
        subject: Optional[str] = None,
        description: str = "",
        calendar_id: Optional[str] = None,
        attendees: Optional[Iterable[str]] = None,
    ) -> CalendarEventResult:
        calendar = calendar_id or "primary"
        interviewer = interviewer_email or self.settings.default_interviewer_email
        attendees_list = [candidate_email]
        if interviewer:
            attendees_list.append(interviewer)
        if attendees:
            attendees_list.extend(attendees)

        body = {
            "summary": subject or f"Interview with {candidate_name}",
            "description": description or "Interview scheduled via automation pipeline.",
            "start": {"dateTime": start_time.isoformat(), "timeZone": self.settings.google_default_timezone},
            "end": {"dateTime": end_time.isoformat(), "timeZone": self.settings.google_default_timezone},
            "attendees": [{"email": email} for email in attendees_list],
            "conferenceData": {
                "createRequest": {
                    "requestId": f"interview-{candidate_email}-{int(start_time.timestamp())}",
                    "conferenceSolutionKey": {"type": "hangoutsMeet"},
                }
            },
        }

        logger.info("Creating calendar event for %s", candidate_email)
        try:
            event = execute_with_retry(
                lambda: self._service.events()
                .insert(
                    calendarId=calendar,
                    body=body,
                    sendUpdates="all",
                    conferenceDataVersion=1,
                )
                .execute()
            )
        except HttpError:  # pragma: no cover - network dependent
            logger.exception("Failed to create calendar event for %s", candidate_email)
            raise

        return CalendarEventResult(
            event_id=event.get("id", ""),
            hangout_link=event.get("hangoutLink") or event.get("conferenceData", {}).get("entryPoints", [{}])[0].get("uri"),
            start=event.get("start", {}).get("dateTime", start_time.isoformat()),
            end=event.get("end", {}).get("dateTime", end_time.isoformat()),
            attendees=[a.get("email") for a in event.get("attendees", [])],
        )


__all__ = ["CalendarService", "CalendarEventResult"]
