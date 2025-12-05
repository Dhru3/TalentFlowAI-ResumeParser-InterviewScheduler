"""Time handling helpers for interview slot generation."""

from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Iterable, List, Optional, Tuple

from zoneinfo import ZoneInfo

DEFAULT_FORMATS = ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%m/%d/%Y")

TIME_BLOCKS = {
    # Standard numeric formats
    "10-12": (time(10, 0), time(12, 0)),
    "10am - 12pm": (time(10, 0), time(12, 0)),
    "10-12pm": (time(10, 0), time(12, 0)),
    "10-12am": (time(10, 0), time(12, 0)),
    "10-12 AM": (time(10, 0), time(12, 0)),
    "10am - 1pm": (time(10, 0), time(13, 0)),
    "10 – 12": (time(10, 0), time(12, 0)),
    "2-4": (time(14, 0), time(16, 0)),
    "2pm - 4pm": (time(14, 0), time(16, 0)),
    "14-16": (time(14, 0), time(16, 0)),
    "4-6": (time(16, 0), time(18, 0)),
    "4pm - 6pm": (time(16, 0), time(18, 0)),
    "16-18": (time(16, 0), time(18, 0)),
    
    # Form response formats with descriptive labels
    "Morning (10:00 AM - 12:00 PM)": (time(10, 0), time(12, 0)),
    "morning (10:00 am - 12:00 pm)": (time(10, 0), time(12, 0)),
    "Morning (10:00 AM - 12:00PM)": (time(10, 0), time(12, 0)),
    "Morning (10AM - 12PM)": (time(10, 0), time(12, 0)),
    "Morning": (time(10, 0), time(12, 0)),
    "morning": (time(10, 0), time(12, 0)),
    
    "Afternoon (2:00 PM - 4:00 PM)": (time(14, 0), time(16, 0)),
    "afternoon (2:00 pm - 4:00 pm)": (time(14, 0), time(16, 0)),
    "Afternoon (2:00 PM - 4:00PM)": (time(14, 0), time(16, 0)),
    "Afternoon (2PM - 4PM)": (time(14, 0), time(16, 0)),
    "Afternoon": (time(14, 0), time(16, 0)),
    "afternoon": (time(14, 0), time(16, 0)),
    
    "Late Afternoon (4:00 PM - 6:00 PM)": (time(16, 0), time(18, 0)),
    "late afternoon (4:00 pm - 6:00 pm)": (time(16, 0), time(18, 0)),
    "Late Afternoon (4:00 PM - 6:00PM)": (time(16, 0), time(18, 0)),
    "Late Afternoon (4PM - 6PM)": (time(16, 0), time(18, 0)),
    "Late Afternoon": (time(16, 0), time(18, 0)),
    "late afternoon": (time(16, 0), time(18, 0)),
    
    # Early morning slots
    "Early Morning (8:00 AM - 10:00 AM)": (time(8, 0), time(10, 0)),
    "early morning (8:00 am - 10:00 am)": (time(8, 0), time(10, 0)),
    "Early Morning": (time(8, 0), time(10, 0)),
    "early morning": (time(8, 0), time(10, 0)),
    
    # Evening slots
    "Evening (6:00 PM - 8:00 PM)": (time(18, 0), time(20, 0)),
    "evening (6:00 pm - 8:00 pm)": (time(18, 0), time(20, 0)),
    "Evening": (time(18, 0), time(20, 0)),
    "evening": (time(18, 0), time(20, 0)),
}

OTHER_BLOCK_DEFAULT = (time(10, 0), time(18, 0))


def parse_preferred_date(value: str, timezone: str = "UTC") -> datetime:
    """Parse a date string from the form into a timezone-aware datetime."""
    cleaned = (value or "").strip()
    if not cleaned:
        raise ValueError("Empty date value")

    for fmt in DEFAULT_FORMATS:
        try:
            dt = datetime.strptime(cleaned, fmt)
            return dt.replace(tzinfo=ZoneInfo(timezone))
        except ValueError:
            continue
    raise ValueError(f"Unable to parse date: {value!r}")


def build_time_window(
    preferred_date: datetime,
    slot_label: str,
    timezone: str,
    *,
    allow_other: bool = True,
) -> Tuple[datetime, datetime]:
    """Return the start/end datetimes for a preferred block."""
    label = (slot_label or "").strip()
    time_range = TIME_BLOCKS.get(label)
    if time_range is None:
        if not allow_other:
            raise ValueError(f"Unknown time slot label: {slot_label!r}")
        time_range = OTHER_BLOCK_DEFAULT

    start_time, end_time = time_range
    tz = ZoneInfo(timezone)
    start_dt = preferred_date.replace(hour=start_time.hour, minute=start_time.minute, second=0, microsecond=0, tzinfo=tz)
    end_dt = preferred_date.replace(hour=end_time.hour, minute=end_time.minute, second=0, microsecond=0, tzinfo=tz)
    if end_dt <= start_dt:
        end_dt += timedelta(days=1)
    return start_dt, end_dt


def generate_time_slots(
    start: datetime,
    end: datetime,
    *,
    duration_minutes: int,
) -> List[Tuple[datetime, datetime]]:
    """Generate sequential slots between start and end with the specified duration."""
    if duration_minutes <= 0:
        raise ValueError("duration_minutes must be positive")

    cursor = start
    results: List[Tuple[datetime, datetime]] = []
    delta = timedelta(minutes=duration_minutes)
    while cursor + delta <= end:
        results.append((cursor, cursor + delta))
        cursor += delta
    return results


__all__ = ["parse_preferred_date", "build_time_window", "generate_time_slots"]
