"""Shared helpers for building authenticated Google API clients."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Iterable, Sequence, TypeVar

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import Resource, build
from googleapiclient.errors import HttpError

from ..config.settings import Settings

T = TypeVar("T")

# All scopes needed across all services - this ensures we request them ALL at once
ALL_REQUIRED_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/forms.responses.readonly",
]


class GoogleClientFactory:
    """Factory responsible for creating and caching Google API services."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._cache: dict[tuple[str, str, tuple[str, ...]], Resource] = {}
        self._credentials: Credentials | None = None

    def build(self, api_name: str, api_version: str, scopes: Sequence[str]) -> Resource:
        key = (api_name, api_version, tuple(sorted(scopes)))
        if key in self._cache:
            return self._cache[key]
        # Always use ALL scopes to avoid token regeneration issues
        creds = self._get_credentials(ALL_REQUIRED_SCOPES)
        service = build(api_name, api_version, credentials=creds, cache_discovery=False)
        self._cache[key] = service
        return service

    def _get_credentials(self, scopes: Sequence[str]) -> Credentials:
        # Return cached credentials if already loaded
        if self._credentials and self._credentials.valid:
            return self._credentials
            
        token_path = self._settings.google_token_file
        creds: Credentials | None = None

        if token_path.exists():
            creds = Credentials.from_authorized_user_file(str(token_path), scopes)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self._settings.google_credentials_file),
                    scopes=scopes,
                )
                # Use port 8080 to match redirect_uri in Google Cloud Console
                creds = flow.run_local_server(port=8080)
            token_path.write_text(creds.to_json())

        # Cache for reuse
        self._credentials = creds
        return creds


def execute_with_retry(func: Callable[[], T], *, retries: int = 3, backoff: float = 1.6) -> T:
    """Execute a callable with simple exponential backoff retry handling."""
    attempt = 0
    while True:
        try:
            return func()
        except HttpError as exc:  # pragma: no cover - network dependent
            attempt += 1
            if attempt > retries:
                raise
            sleep_for = backoff ** attempt
            time.sleep(sleep_for)


__all__ = ["GoogleClientFactory", "execute_with_retry"]
