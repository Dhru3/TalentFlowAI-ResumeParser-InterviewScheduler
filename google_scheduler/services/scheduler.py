"""Scheduling logic for assigning interview slots and creating calendar events."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional

from googleapiclient.errors import HttpError

from ..config.settings import Settings
from ..utils.logging_utils import get_logger
from ..utils.time_utils import build_time_window, generate_time_slots, parse_preferred_date
from .calendar_service import CalendarEventResult, CalendarService
from .gmail_service import GmailService
from .sheets_service import SheetRow, SheetsService

logger = get_logger(__name__)

INTERVIEW_DURATION_MINUTES = 30
SCHEDULED_FLAG = "Scheduled"
ASSIGNED_DATE = "Assigned Date"
ASSIGNED_START = "Assigned Start Time"
ASSIGNED_END = "Assigned End Time"
EVENT_ID = "Calendar Event ID"
MEET_LINK_COL = "Meet Link"
STATUS_COL = "Status"


@dataclass(slots=True)
class CandidateResponse:
    row: SheetRow
    name: str
    email: str
    phone: str
    preferred_date: datetime
    preferred_slot_label: str
    concerns: str


@dataclass(slots=True)
class ScheduledInterview:
    candidate: CandidateResponse
    start: datetime
    end: datetime
    calendar_event: CalendarEventResult


@dataclass(slots=True)
class ScheduleProposal:
    candidate: CandidateResponse
    suggested_start: Optional[datetime]
    suggested_end: Optional[datetime]
    status: str
    note: str = ""

    def with_updates(self, *, start: Optional[datetime], end: Optional[datetime], status: Optional[str] = None, note: Optional[str] = None) -> "ScheduleProposal":
        return ScheduleProposal(
            candidate=self.candidate,
            suggested_start=start if start is not None else self.suggested_start,
            suggested_end=end if end is not None else self.suggested_end,
            status=status or self.status,
            note=note or self.note,
        )


class SchedulingService:
    """Coordinate scheduling by combining Sheets, Calendar, and Gmail services."""

    def __init__(
        self,
        settings: Settings,
        sheets: SheetsService,
        calendar: CalendarService,
        gmail: GmailService,
    ):
        self.settings = settings
        self.sheets = sheets
        self.calendar = calendar
        self.gmail = gmail

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def schedule_pending_interviews(
        self,
        *,
        interviewer_email: Optional[str] = None,
        confirmation_template: Optional[str] = None,
    ) -> List[ScheduledInterview]:
        proposals = self.plan_schedule()
        ready = [proposal for proposal in proposals if proposal.status.lower() == "ready"]
        return self.finalize_schedule(
            ready,
            interviewer_email=interviewer_email,
            confirmation_template=confirmation_template,
        )

    def plan_schedule(self, rows: Optional[List[SheetRow]] = None) -> List[ScheduleProposal]:
        rows = rows or self.sheets.fetch_form_responses()
        if not rows:
            logger.info("No form responses found")
            return []

        booked_slots = self._collect_booked_slots(rows)
        proposals: List[ScheduleProposal] = []

        for row in rows:
            if row.data.get(SCHEDULED_FLAG, "").strip().lower() == "yes":
                continue

            try:
                candidate = self._parse_candidate(row)
            except ValueError as exc:
                logger.error("Skipping row %s due to parsing error: %s", row.row_number, exc)
                # Try multiple possible column names for time slot
                slot_label = (
                    row.data.get("Preferred Time Slot (Note that the duration of the interview will be only 20-40mins)")
                    or row.data.get("Preferred Time Slot")
                    or ""
                )
                proposals.append(
                    ScheduleProposal(
                        candidate=CandidateResponse(
                            row=row,
                            name=row.data.get("Full Name", row.row_number),
                            email=row.data.get("Email ID", ""),
                            phone=row.data.get("Phone Number", ""),
                            preferred_date=datetime.now(),
                            preferred_slot_label=slot_label,
                            concerns="",
                        ),
                        suggested_start=None,
                        suggested_end=None,
                        status="Error",
                        note=str(exc),
                    )
                )
                continue

            slot = self._find_available_slot(candidate, booked_slots)
            if slot is None:
                proposals.append(
                    ScheduleProposal(
                        candidate=candidate,
                        suggested_start=None,
                        suggested_end=None,
                        status="Waiting",
                        note="No available slots",
                    )
                )
                continue

            start_time, end_time = slot
            proposals.append(
                ScheduleProposal(
                    candidate=candidate,
                    suggested_start=start_time,
                    suggested_end=end_time,
                    status="Ready",
                )
            )
            booked_slots.add((start_time.strftime("%d/%m/%Y"), start_time.strftime("%H:%M")))

        return proposals

    def finalize_schedule(
        self,
        proposals: Iterable[ScheduleProposal],
        *,
        interviewer_email: Optional[str] = None,
        confirmation_template: Optional[str] = None,
    ) -> List[ScheduledInterview]:
        # Ensure headers are cached and scheduling columns exist
        self.sheets.ensure_scheduling_columns()
        
        scheduled: List[ScheduledInterview] = []
        for proposal in proposals:
            if proposal.status.lower() not in {"ready", "scheduled"}:
                continue
            start_time = proposal.suggested_start
            end_time = proposal.suggested_end
            if not start_time or not end_time:
                logger.warning("Skipping proposal for %s due to missing times", proposal.candidate.email)
                continue

            candidate = proposal.candidate
            try:
                event = self.calendar.create_interview_event(
                    candidate_name=candidate.name,
                    candidate_email=candidate.email,
                    start_time=start_time,
                    end_time=end_time,
                    interviewer_email=interviewer_email,
                    description=candidate.concerns or proposal.note,
                )
            except HttpError as exc:  # pragma: no cover - network dependent
                logger.exception("Failed creating event for %s", candidate.email)
                self.sheets.update_row(
                    candidate.row.row_number,
                    {
                        STATUS_COL: "Calendar Error",
                        SCHEDULED_FLAG: "No",
                        "Error": str(exc),
                    },
                )
                continue

            scheduled.append(ScheduledInterview(candidate, start_time, end_time, event))

            updates = {
                SCHEDULED_FLAG: "Yes",
                ASSIGNED_DATE: start_time.strftime("%d/%m/%Y"),
                ASSIGNED_START: start_time.strftime("%H:%M"),
                ASSIGNED_END: end_time.strftime("%H:%M"),
                EVENT_ID: event.event_id,
                MEET_LINK_COL: event.hangout_link or "",
                STATUS_COL: "Scheduled",
            }
            self.sheets.update_row(candidate.row.row_number, updates)

            if event.hangout_link:
                time_display = f"{start_time.strftime('%d %b %Y %H:%M')} - {end_time.strftime('%H:%M %Z')}"
                self.gmail.send_confirmation_email(
                    candidate_email=candidate.email,
                    candidate_name=candidate.name,
                    interview_date=start_time.strftime("%d %b %Y"),
                    interview_time=time_display,
                    meet_link=event.hangout_link,
                    template_path=confirmation_template,
                )

        return scheduled

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _parse_candidate(self, row: SheetRow) -> CandidateResponse:
        data = row.data
        email = (data.get("Email ID") or data.get("Email") or "").strip()
        if not email:
            raise ValueError("Missing Email ID")

        name = (data.get("Full Name") or data.get("Name") or email).strip()
        phone = (data.get("Phone Number") or "").strip()
        concerns = (data.get("Any Concerns?") or data.get("Concerns") or "").strip()
        
        # Try multiple possible column names for time slot
        slot_label = (
            data.get("Preferred Time Slot (Note that the duration of the interview will be only 20-40mins)")
            or data.get("Preferred Time Slot")
            or "Other"
        )

        preferred_date = parse_preferred_date(data.get("Preferred Interview Date", data.get("Preferred date of interview", "")), self.settings.google_default_timezone)

        return CandidateResponse(
            row=row,
            name=name,
            email=email,
            phone=phone,
            preferred_date=preferred_date,
            preferred_slot_label=slot_label,
            concerns=concerns,
        )

    def _collect_booked_slots(self, rows: Iterable[SheetRow]):
        booked = set()
        for row in rows:
            if row.data.get(SCHEDULED_FLAG, "").strip().lower() != "yes":
                continue
            date_value = row.data.get(ASSIGNED_DATE)
            start = row.data.get(ASSIGNED_START)
            if date_value and start:
                booked.add((date_value, start))
        return booked

    def _find_available_slot(self, candidate: CandidateResponse, booked_slots: set[tuple[str, str]]):
        start, end = build_time_window(
            candidate.preferred_date,
            candidate.preferred_slot_label,
            self.settings.google_default_timezone,
        )
        slots = generate_time_slots(start, end, duration_minutes=INTERVIEW_DURATION_MINUTES)
        for slot_start, slot_end in slots:
            key = (slot_start.strftime("%d/%m/%Y"), slot_start.strftime("%H:%M"))
            if key not in booked_slots:
                return slot_start, slot_end
        return None


__all__ = [
    "SchedulingService",
    "ScheduledInterview",
    "ScheduleProposal",
    "CandidateResponse",
]
