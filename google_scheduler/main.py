"""Entry point for the Google-based interview scheduling pipeline."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Mapping, Optional

from .config import load_settings
from .services.calendar_service import CalendarService
from .services.gmail_service import GmailService
from .services.google_client import GoogleClientFactory
from .services.scheduler import SchedulingService
from .services.sheets_service import SheetsService
from .utils import get_logger, setup_logging

logger = get_logger(__name__)


def load_candidates_from_csv(csv_path: Path) -> List[Mapping[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Candidate CSV not found: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def send_form_invitations(
    gmail: GmailService,
    *,
    candidates: Iterable[Mapping[str, str]],
    forms_link: str,
    job_title: str,
) -> None:
    candidate_list = list(candidates)
    logger.info("Sending invitation emails to %s candidates", len(candidate_list))
    results = gmail.send_form_invitations(candidate_list, forms_link, job_title=job_title)
    sent = sum(1 for result in results if result["status"] == "sent")
    failed = len(results) - sent
    logger.info("Invitations sent: %s | Failed: %s", sent, failed)


def orchestrate_pipeline(
    *,
    settings_path: Optional[str] = None,
    forms_link: Optional[str] = None,
    candidates_csv: Optional[str] = None,
    job_title: str = "AI Engineer",
) -> None:
    settings = load_settings(settings_path)
    setup_logging(settings)
    logger.info("Configuration loaded. Starting pipeline")

    factory = GoogleClientFactory(settings)
    gmail = GmailService(settings, factory)
    sheets = SheetsService(settings, factory)
    calendar = CalendarService(settings, factory)
    scheduler = SchedulingService(settings, sheets, calendar, gmail)

    if candidates_csv and forms_link:
        candidate_rows = load_candidates_from_csv(Path(candidates_csv))
        send_form_invitations(gmail, candidates=candidate_rows, forms_link=forms_link, job_title=job_title)

    scheduled = scheduler.schedule_pending_interviews(interviewer_email=settings.default_interviewer_email)
    logger.info("Scheduling complete. %s interviews booked.", len(scheduled))

    for record in scheduled:
        logger.info(
            "Scheduled %s (%s) on %s from %s to %s",
            record.candidate.name,
            record.candidate.email,
            record.start.date().isoformat(),
            record.start.time().isoformat(timespec="minutes"),
            record.end.time().isoformat(timespec="minutes"),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Google-based interview scheduling pipeline")
    parser.add_argument("--forms-link", help="Google Forms link to include in invitation emails")
    parser.add_argument("--candidates-csv", help="Path to CSV file containing candidate emails and names")
    parser.add_argument("--job-title", default="AI Engineer", help="Job title to reference in invitation emails")
    parser.add_argument("--env-file", help="Optional path to a .env file with configuration overrides")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    orchestrate_pipeline(
        settings_path=args.env_file,
        forms_link=args.forms_link,
        candidates_csv=args.candidates_csv,
        job_title=args.job_title,
    )


if __name__ == "__main__":
    main()
