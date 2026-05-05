# -*- coding: utf-8 -*-
"""
server_multipart.py

stdlib-only multipart/form-data parser.
Replaces the deprecated `cgi.FieldStorage` (removed in Python 3.13).

Usage:
    from server_multipart import _parse_multipart
    form = _parse_multipart(self.rfile, self.headers)
    value = form.get("text", "")
"""

from email import message_from_bytes
from typing import IO, Optional


def _parse_multipart(rfile: IO[bytes], headers: dict) -> dict:
    """Parse a multipart/form-data request body.

    Args:
        rfile:   Readable byte stream positioned at the start of the body.
        headers: Mapping that provides Content-Type and Content-Length.

    Returns:
        dict mapping field name -> field value (str).
        Binary payloads (e.g. base64 image strings) are returned as-is strings.
    """
    content_length = int(headers.get("Content-Length", 0) or 0)
    if content_length <= 0:
        return {}

    body = rfile.read(content_length)
    content_type = headers.get("Content-Type", "")

    # email.message_from_bytes requires a full MIME message with headers.
    mime_msg = message_from_bytes(
        b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body
    )

    result: dict = {}
    payload = mime_msg.get_payload()
    if not isinstance(payload, list):
        # Not multipart — return empty
        return result

    for part in payload:
        disposition = part.get("Content-Disposition", "")
        name = _extract_name(disposition)
        if name is None:
            continue

        # decode=True returns raw bytes (undoes transfer-encoding).
        # decode=False returns str but using latin-1, mangling UTF-8.
        raw_bytes = part.get_payload(decode=True)
        if raw_bytes is None:
            # Fallback: decode=True returns None for non-leaf MIME parts.
            raw_str = part.get_payload(decode=False) or ""
            raw = raw_str.rstrip("\r\n") if isinstance(raw_str, str) else ""
        else:
            raw = raw_bytes.rstrip(b"\r\n").decode("utf-8", errors="replace")

        result[name] = raw

    return result


def _extract_name(disposition: str) -> Optional[str]:
    """Return the `name` parameter from a Content-Disposition header, or None."""
    for segment in disposition.split(";"):
        segment = segment.strip()
        if segment.lower().startswith("name="):
            return segment[5:].strip('"').strip("'")
    return None
