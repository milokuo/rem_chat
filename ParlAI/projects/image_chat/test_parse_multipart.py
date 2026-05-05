# -*- coding: utf-8 -*-
"""
TDD tests for _parse_multipart helper (Phase 3 — replaces deprecated cgi.FieldStorage).

Run: python -m pytest test_parse_multipart.py -v
"""

import io
import pytest
from server_multipart import _parse_multipart


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_multipart(fields: dict, boundary: str = "testboundary123") -> tuple[bytes, str]:
    """Return (body_bytes, content_type_header) for a multipart/form-data request."""
    parts = []
    for name, value in fields.items():
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n'
            f"\r\n"
            f"{value}\r\n"
        )
    parts.append(f"--{boundary}--\r\n")
    body = "".join(parts).encode("utf-8")
    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


def _make_headers(body: bytes, content_type: str) -> dict:
    return {
        "Content-Type": content_type,
        "Content-Length": str(len(body)),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestParseMultipartBasic:
    def test_single_text_field(self):
        body, ct = _build_multipart({"text": "hello"})
        result = _parse_multipart(io.BytesIO(body), _make_headers(body, ct))
        assert result["text"] == "hello"

    def test_multiple_fields(self):
        fields = {"text": "hi", "image_name": "photo.jpg", "img_id": "7"}
        body, ct = _build_multipart(fields)
        result = _parse_multipart(io.BytesIO(body), _make_headers(body, ct))
        assert result["text"] == "hi"
        assert result["image_name"] == "photo.jpg"
        assert result["img_id"] == "7"

    def test_all_server_fields(self):
        """Verify every field used in process_post() is correctly parsed."""
        fields = {
            "text": "how are you",
            "image_name": "family.jpg",
            "image": "base64encodeddata==",
            "metadata": '{"foo": "bar"}',
            "img_id": "42",
            "cate": "family",
        }
        body, ct = _build_multipart(fields)
        result = _parse_multipart(io.BytesIO(body), _make_headers(body, ct))
        for key, val in fields.items():
            assert result[key] == val, f"field {key!r}: expected {val!r}, got {result.get(key)!r}"


class TestParseMultipartEdgeCases:
    def test_empty_body_returns_empty_dict(self):
        body, ct = _build_multipart({})
        result = _parse_multipart(io.BytesIO(body), _make_headers(body, ct))
        assert result == {}

    def test_field_with_empty_value(self):
        body, ct = _build_multipart({"text": "", "image_name": ""})
        result = _parse_multipart(io.BytesIO(body), _make_headers(body, ct))
        assert result["text"] == ""
        assert result["image_name"] == ""

    def test_unicode_value(self):
        body, ct = _build_multipart({"text": "你好世界"})
        result = _parse_multipart(io.BytesIO(body), _make_headers(body, ct))
        assert result["text"] == "你好世界"

    def test_missing_content_length_reads_zero(self):
        """If Content-Length is absent, function must not crash."""
        body, ct = _build_multipart({"text": "x"})
        headers = {"Content-Type": ct}  # no Content-Length
        # With length=0 the body cannot be read; result should be empty, not an exception.
        result = _parse_multipart(io.BytesIO(body), headers)
        assert isinstance(result, dict)
