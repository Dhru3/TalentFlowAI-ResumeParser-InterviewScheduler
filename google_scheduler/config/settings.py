"""Configuration management for the Google interview scheduling pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def _expand_path(value: str | os.PathLike[str], *, base: Optional[Path] = None) -> Path:
    base = base or Path.cwd()
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


@dataclass(slots=True)
class Settings:
    """Resolved configuration values for all pipeline components."""

    google_credentials_file: Path
    google_token_file: Path
    google_default_timezone: str

    gmail_sender_address: str
    gmail_template_path: Path
    gmail_confirmation_template_path: Path
    google_form_link: str

    google_sheet_id: str
    google_sheet_range: str

    default_interviewer_email: str

    log_level: str
    log_directory: Path

    def ensure_paths(self) -> None:
        """Ensure configured directories exist where applicable."""
        self.google_credentials_file.parent.mkdir(parents=True, exist_ok=True)
        self.google_token_file.parent.mkdir(parents=True, exist_ok=True)
        self.gmail_template_path.parent.mkdir(parents=True, exist_ok=True)
        self.gmail_confirmation_template_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def load_settings(dotenv_path: str | os.PathLike[str] | None = None) -> Settings:
    """Load settings from environment variables (and an optional .env file)."""
    base_dir = Path.cwd()
    load_dotenv(dotenv_path, override=False)

    credentials_file = os.getenv("GOOGLE_CREDENTIALS_FILE")
    if not credentials_file:
        raise RuntimeError("GOOGLE_CREDENTIALS_FILE must be set.")

    token_file = os.getenv("GOOGLE_TOKEN_FILE", "token.json")
    timezone = os.getenv("GOOGLE_DEFAULT_TIMEZONE", "UTC")

    gmail_sender = os.getenv("GMAIL_SENDER_ADDRESS")
    if not gmail_sender:
        raise RuntimeError("GMAIL_SENDER_ADDRESS must be set.")

    gmail_template = os.getenv("GMAIL_TEMPLATE_PATH", "templates/invitation_email.html")
    gmail_confirmation_template = os.getenv(
        "GMAIL_CONFIRMATION_TEMPLATE_PATH", "templates/confirmation_email.html"
    )
    forms_link = os.getenv("GOOGLE_FORM_LINK") or os.getenv("GOOGLE_FORMS_LINK", "")

    sheet_id = os.getenv("GOOGLE_SHEET_ID")
    if not sheet_id:
        raise RuntimeError("GOOGLE_SHEET_ID must be set.")

    sheet_range = os.getenv("GOOGLE_SHEET_RANGE", "Form_Responses_1!A1:G")
    interviewer_email = os.getenv("DEFAULT_INTERVIEWER_EMAIL", gmail_sender)

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_dir = os.getenv("LOG_DIR", "logs")

    settings = Settings(
        google_credentials_file=_expand_path(credentials_file, base=base_dir),
        google_token_file=_expand_path(token_file, base=base_dir),
        google_default_timezone=timezone,
        gmail_sender_address=gmail_sender,
        gmail_template_path=_expand_path(gmail_template, base=base_dir),
    gmail_confirmation_template_path=_expand_path(gmail_confirmation_template, base=base_dir),
    google_form_link=forms_link,
        google_sheet_id=sheet_id,
        google_sheet_range=sheet_range,
        default_interviewer_email=interviewer_email,
        log_level=log_level,
        log_directory=_expand_path(log_dir, base=base_dir),
    )

    settings.ensure_paths()
    return settings


__all__ = ["Settings", "load_settings"]
