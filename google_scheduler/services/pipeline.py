"""High-level pipeline helper for Streamlit/CLI integrations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional

from ..config import load_settings
from ..config.settings import Settings
from .calendar_service import CalendarService
from .gmail_service import GmailService
from .google_client import GoogleClientFactory
from .scheduler import ScheduleProposal, ScheduledInterview, SchedulingService
from .sheets_service import SheetRow, SheetsService


@dataclass(slots=True)
class SchedulerPipeline:
    settings: Settings
    factory: GoogleClientFactory
    gmail: GmailService
    sheets: SheetsService
    calendar: CalendarService
    scheduler: SchedulingService

    @classmethod
    def from_env(cls, dotenv_path: Optional[str] = None) -> "SchedulerPipeline":
        settings = load_settings(dotenv_path)
        factory = GoogleClientFactory(settings)
        gmail = GmailService(settings, factory)
        sheets = SheetsService(settings, factory)
        calendar = CalendarService(settings, factory)
        scheduler = SchedulingService(settings, sheets, calendar, gmail)
        return cls(settings, factory, gmail, sheets, calendar, scheduler)

    # ------------------------------------------------------------------
    # Invitation step
    # ------------------------------------------------------------------
    def send_form_invitations(
        self,
        candidates: Iterable[Mapping[str, str]],
        *,
        job_title: str,
        forms_link: Optional[str] = None,
    ) -> List[Mapping[str, str]]:
        link = forms_link or self.settings.google_form_link
        if not link:
            raise RuntimeError("Google Form link is not configured. Set GOOGLE_FORM_LINK in your environment.")
        return self.gmail.send_form_invitations(candidates, link, job_title=job_title)

    # ------------------------------------------------------------------
    # Sheets + scheduling
    # ------------------------------------------------------------------
    def ensure_scheduling_columns(self) -> None:
        """Ensure all required scheduling columns exist in the Google Sheet."""
        return self.sheets.ensure_scheduling_columns()

    def fetch_form_responses(self) -> List[SheetRow]:
        return self.sheets.fetch_form_responses()

    def plan_schedule(self, rows: Optional[List[SheetRow]] = None) -> List[ScheduleProposal]:
        return self.scheduler.plan_schedule(rows)

    def finalize_schedule(
        self,
        proposals: Iterable[ScheduleProposal],
        *,
        interviewer_email: Optional[str] = None,
        confirmation_template: Optional[str] = None,
    ) -> List[ScheduledInterview]:
        return self.scheduler.finalize_schedule(
            proposals,
            interviewer_email=interviewer_email,
            confirmation_template=confirmation_template,
        )


__all__ = ["SchedulerPipeline"]
