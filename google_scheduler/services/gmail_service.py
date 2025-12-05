"""Service for sending emails via the Gmail API."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

from googleapiclient.errors import HttpError

from ..config.settings import Settings
from ..utils.logging_utils import get_logger
from .google_client import GoogleClientFactory, execute_with_retry

logger = get_logger(__name__)

GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.send"]


@dataclass(slots=True)
class CandidateInvite:
    name: str
    email: str
    phone: Optional[str] = None


class GmailService:
    """Wrapper around the Gmail API for sending templated emails."""

    def __init__(self, settings: Settings, factory: GoogleClientFactory | None = None):
        self.settings = settings
        self._factory = factory or GoogleClientFactory(settings)
        self._service = self._factory.build("gmail", "v1", GMAIL_SCOPES)
        self._template_cache: dict[Path, str] = {}

    # ---------------------------------------------------------------------
    # Template handling
    # ---------------------------------------------------------------------
    def _load_template(self, template_path: Path) -> str:
        template_path = template_path.resolve()
        if template_path not in self._template_cache:
            if not template_path.exists():
                raise FileNotFoundError(f"Email template not found: {template_path}")
            self._template_cache[template_path] = template_path.read_text(encoding="utf-8")
        return self._template_cache[template_path]

    def _render_template(self, template: str, context: Mapping[str, str]) -> str:
        rendered = template
        for key, value in context.items():
            rendered = rendered.replace(f"{{{{{key}}}}}", value)
        return rendered

    # ---------------------------------------------------------------------
    # Gmail execution
    # ---------------------------------------------------------------------
    def _send_raw(self, message: EmailMessage) -> Dict:
        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        body = {"raw": encoded_message}
        return execute_with_retry(lambda: self._service.users().messages().send(userId="me", body=body).execute())

    def send_html_email(
        self,
        recipient: str,
        subject: str,
        html_body: str,
        *,
        cc: Optional[Iterable[str]] = None,
        bcc: Optional[Iterable[str]] = None,
        attachments: Optional[List[Path]] = None,
    ) -> Dict:
        """Send an HTML email to a single recipient."""
        msg = EmailMessage()
        msg["To"] = recipient
        msg["From"] = self.settings.gmail_sender_address
        msg["Subject"] = subject
        if cc:
            msg["Cc"] = ", ".join(cc)
        if bcc:
            msg["Bcc"] = ", ".join(bcc)

        msg.set_content("This email requires an HTML capable client.")
        msg.add_alternative(html_body, subtype="html")

        attachments = attachments or []
        for attachment in attachments:
            data = attachment.read_bytes()
            msg.add_attachment(
                data,
                maintype="application",
                subtype="octet-stream",
                filename=attachment.name,
            )

        logger.info("Sending email to %s", recipient)
        return self._send_raw(msg)

    # ------------------------------------------------------------------
    # High level flows
    # ------------------------------------------------------------------
    def send_form_invitations(
        self,
        candidates: Iterable[Mapping[str, str]],
        forms_link: str,
        *,
        job_title: str,
        template_path: Optional[Path] = None,
    ) -> List[Dict[str, str]]:
        """Send Google Form invitations to a list of candidates."""
        template_file = template_path or self.settings.gmail_template_path
        template_content = self._load_template(template_file)

        results: List[Dict[str, str]] = []
        for candidate in candidates:
            email = (candidate.get("email") or candidate.get("Email ID") or "").strip()
            name = (
                candidate.get("name")
                or candidate.get("full_name")
                or candidate.get("Full Name")
                or "Candidate"
            )
            if not email:
                logger.warning("Skipping candidate with missing email: %s", candidate)
                results.append({"status": "skipped", "reason": "missing_email", "candidate": name})
                continue

            context = {
                "candidate_name": name,
                "forms_link": forms_link,
                "job_title": job_title,
                "interview_date": "",
                "interview_time": "",
                "meet_link": "",
            }
            html = self._render_template(template_content, context)
            subject = f"Interview Availability for {job_title}" if job_title else "Interview Availability"

            try:
                self.send_html_email(email, subject, html)
                results.append({"status": "sent", "email": email, "name": name})
            except HttpError as exc:  # pragma: no cover - network dependent
                logger.exception("Failed to send email to %s", email)
                results.append({"status": "failed", "email": email, "error": str(exc)})

        return results

    def send_confirmation_email(
        self,
        candidate_email: str,
        candidate_name: str,
        *,
        interview_date: str,
        interview_time: str,
        meet_link: str,
        template_path: Optional[Path] = None,
    ) -> Dict[str, str]:
        """Send a confirmation email with meeting details."""
        template_file = template_path or self.settings.gmail_confirmation_template_path
        template_content = self._load_template(template_file)

        context = {
            "candidate_name": candidate_name,
            "interview_date": interview_date,
            "interview_time": interview_time,
            "meet_link": meet_link,
            "forms_link": meet_link,
            "job_title": "Interview",
        }
        subject = f"Interview Scheduled on {interview_date}"
        html_body = self._render_template(template_content, context)
        try:
            self.send_html_email(candidate_email, subject, html_body)
            return {"status": "sent", "email": candidate_email}
        except HttpError as exc:  # pragma: no cover - network dependent
            logger.exception("Failed to send confirmation to %s", candidate_email)
            return {"status": "failed", "email": candidate_email, "error": str(exc)}


__all__ = ["GmailService", "CandidateInvite"]
