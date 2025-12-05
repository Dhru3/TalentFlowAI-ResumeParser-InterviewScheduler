"""Service wrapper for Google Sheets interactions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional

from googleapiclient.discovery import Resource
from googleapiclient.errors import HttpError

from ..config.settings import Settings
from ..utils.logging_utils import get_logger
from .google_client import GoogleClientFactory, execute_with_retry

logger = get_logger(__name__)

SHEETS_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

_COLUMN_RE = re.compile(r"([A-Z]+)")


def _column_to_index(column: str) -> int:
    column = column.upper()
    result = 0
    for char in column:
        result = result * 26 + (ord(char) - ord("A") + 1)
    return result - 1


def _index_to_column(index: int) -> str:
    index += 1
    result = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


@dataclass(slots=True)
class SheetRow:
    row_number: int
    data: Dict[str, str]


class SheetsService:
    """Wrapper for reading and updating Google Sheets rows."""

    def __init__(self, settings: Settings, factory: GoogleClientFactory | None = None):
        self.settings = settings
        self._factory = factory or GoogleClientFactory(settings)
        self._service: Resource = self._factory.build("sheets", "v4", SHEETS_SCOPES)
        self._sheet_name, self._range_notation = self._split_range(settings.google_sheet_range)
        self._start_column = self._extract_start_column(self._range_notation)
        self._start_index = _column_to_index(self._start_column)
        self._header: List[str] | None = None

    @staticmethod
    def _split_range(range_value: str) -> tuple[str, str]:
        if "!" in range_value:
            sheet_name, range_notation = range_value.split("!", 1)
        else:
            sheet_name, range_notation = "Sheet1", range_value
        return sheet_name, range_notation

    @staticmethod
    def _extract_start_column(range_notation: str) -> str:
        match = _COLUMN_RE.search(range_notation)
        return match.group(1) if match else "A"

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def fetch_form_responses(self) -> List[SheetRow]:
        """Fetch all form responses with their sheet row numbers."""
        # Use full sheet range to capture all columns including newly added ones
        # Format: "SheetName" or "SheetName!A:Z" to get all columns
        fetch_range = self._sheet_name
        
        result = execute_with_retry(
            lambda: self._service.spreadsheets()
            .values()
            .get(spreadsheetId=self.settings.google_sheet_id, range=fetch_range)
            .execute()
        )
        values = result.get("values", [])
        if not values:
            return []

        header = values[0]
        self._header = header
        rows: List[SheetRow] = []
        for offset, row in enumerate(values[1:], start=2):
            normalized = {header[i]: row[i] if i < len(row) else "" for i in range(len(header))}
            rows.append(SheetRow(row_number=offset, data=normalized))
        return rows

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def ensure_scheduling_columns(self) -> None:
        """Add scheduling columns to the sheet if they don't exist."""
        if self._header is None:
            # Fetch to populate header
            self.fetch_form_responses()
        
        if self._header is None:
            raise RuntimeError("Could not load sheet headers")
        
        required_columns = [
            "Scheduled",
            "Assigned Date",
            "Assigned Start Time",
            "Assigned End Time",
            "Calendar Event ID",
            "Meet Link",
            "Status",
        ]
        
        # Find columns that need to be added
        missing_columns = [col for col in required_columns if col not in self._header]
        
        if not missing_columns:
            logger.info("All scheduling columns already exist")
            return
        
        # Add missing columns to the header row
        start_col_idx = len(self._header)
        new_header_values = self._header + missing_columns
        
        # Calculate the range for the header row
        end_column = _index_to_column(self._start_index + len(new_header_values) - 1)
        header_range = f"{self._sheet_name}!{self._start_column}1:{end_column}1"
        
        logger.info(f"Adding {len(missing_columns)} columns: {', '.join(missing_columns)}")
        
        body = {
            "values": [new_header_values]
        }
        
        try:
            execute_with_retry(
                lambda: self._service.spreadsheets()
                .values()
                .update(
                    spreadsheetId=self.settings.google_sheet_id,
                    range=header_range,
                    valueInputOption="USER_ENTERED",
                    body=body,
                )
                .execute()
            )
            # Update cached header
            self._header = new_header_values
            logger.info("Successfully added scheduling columns")
        except HttpError as exc:
            logger.exception("Failed to add scheduling columns")
            raise

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def update_row(self, row_number: int, updates: Mapping[str, str]) -> None:
        if not updates:
            return
        if self._header is None:
            raise RuntimeError("fetch_form_responses must be called before update_row to cache headers.")

        data_entries = []
        for key, value in updates.items():
            if key not in self._header:
                logger.warning("Skipping unknown column %s", key)
                continue
            idx = self._header.index(key)
            column = _index_to_column(self._start_index + idx)
            data_entries.append(
                {"range": f"{self._sheet_name}!{column}{row_number}", "values": [[value]]}
            )

        if not data_entries:
            return

        body = {"valueInputOption": "USER_ENTERED", "data": data_entries}
        try:
            execute_with_retry(
                lambda: self._service.spreadsheets()
                .values()
                .batchUpdate(
                    spreadsheetId=self.settings.google_sheet_id,
                    body=body,
                )
                .execute()
            )
        except HttpError as exc:  # pragma: no cover - network dependent
            logger.exception("Failed to update row %s", row_number)
            raise


__all__ = ["SheetsService", "SheetRow"]
