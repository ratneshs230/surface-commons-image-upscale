"""Integration tests for /commons/image/upscale + /enhance.

Run against a live surface-api at 127.0.0.1:3000 (or SURFACE_BASE_URL) with
a service-key + an existing seeded agent. Same env contract as
test_bg_remove.py:

    SURFACE_BASE_URL          (default: http://127.0.0.1:3000)
    SURFACE_SERVICE_KEY       (or SERVICE_API_KEY)
    SURFACE_TEST_AGENT_ID

Real image data, real engine - no mocks. The first /upscale call cold-loads
the realesrgan-ncnn engine for the requested model index, so the timeout is
generous.
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import httpx
import pytest


FIX = Path(__file__).parent / "fixtures"
BASE = os.environ.get("SURFACE_BASE_URL", "http://127.0.0.1:3000")
SVC_KEY = os.environ.get("SURFACE_SERVICE_KEY") or os.environ.get("SERVICE_API_KEY")
AGENT_ID = os.environ.get("SURFACE_TEST_AGENT_ID")


def _client() -> httpx.Client:
    if not SVC_KEY or not AGENT_ID:
        pytest.skip("SURFACE_SERVICE_KEY + SURFACE_TEST_AGENT_ID required")
    return httpx.Client(
        base_url=BASE,
        headers={"X-Service-Key": SVC_KEY, "X-Agent-Id": AGENT_ID},
        timeout=300.0,
    )


def _png_size(png_bytes: bytes) -> tuple[int, int, str]:
    from PIL import Image
    im = Image.open(io.BytesIO(png_bytes))
    return im.width, im.height, im.mode


def test_health_public():
    r = httpx.get(f"{BASE}/api/v1/commons/image/upscale-health", timeout=10)
    assert r.status_code == 200
    j = r.json()
    assert j["service"] == "commons.image.upscale"
    assert "available" in j
    assert "backend" in j
    assert j["backend"] in ("realesrgan_ncnn", "pil_bicubic")
    assert "sc_cost" in j
    assert j["sc_cost"]["upscale"] == 5.0
    assert j["sc_cost"]["enhance"] == 2.0
    assert j["allowed_scales"] == [2, 3, 4]


def test_upscale_x2():
    with _client() as c:
        body = (FIX / "upscale_input.png").read_bytes()
        files = [("image", ("upscale_input.png", body, "image/png"))]
        data = {"scale": "2"}
        r = c.post("/api/v1/commons/image/upscale", files=files, data=data)
        assert r.status_code == 200, r.text
        assert r.headers["content-type"] == "image/png"
        assert r.headers["x-sc-cost"] == "5.0"
        assert r.headers["x-scale"] == "2"
        assert r.headers["x-out-width"] == "400"
        assert r.headers["x-out-height"] == "400"
        assert r.content[:8] == b"\x89PNG\r\n\x1a\n"
        w, h, _mode = _png_size(r.content)
        assert (w, h) == (400, 400)


def test_upscale_x4():
    with _client() as c:
        body = (FIX / "upscale_input.png").read_bytes()
        files = [("image", ("upscale_input.png", body, "image/png"))]
        data = {"scale": "4"}
        r = c.post("/api/v1/commons/image/upscale", files=files, data=data)
        assert r.status_code == 200, r.text
        assert r.headers["x-out-width"] == "800"
        assert r.headers["x-out-height"] == "800"
        w, h, _mode = _png_size(r.content)
        assert (w, h) == (800, 800)
        # x4 output should be larger than input by definition
        assert len(r.content) > len(body) * 2


def test_upscale_bad_scale_rejected():
    with _client() as c:
        body = (FIX / "upscale_input.png").read_bytes()
        files = [("image", ("upscale_input.png", body, "image/png"))]
        data = {"scale": "8"}
        r = c.post("/api/v1/commons/image/upscale", files=files, data=data)
        assert r.status_code == 400
        assert r.json()["detail"]["error"] == "bad_scale"


def test_enhance_returns_same_dimensions():
    with _client() as c:
        body = (FIX / "upscale_input.png").read_bytes()
        files = [("image", ("upscale_input.png", body, "image/png"))]
        r = c.post("/api/v1/commons/image/enhance", files=files)
        assert r.status_code == 200, r.text
        assert r.headers["content-type"] == "image/png"
        assert r.headers["x-sc-cost"] == "2.0"
        assert r.headers["x-width"] == "200"
        assert r.headers["x-height"] == "200"
        w, h, _mode = _png_size(r.content)
        assert (w, h) == (200, 200)
        # Bytes should differ from input - enhancement must actually happen.
        assert r.content != body


def test_auth_rejected_without_key():
    r = httpx.post(f"{BASE}/api/v1/commons/image/upscale", timeout=10)
    assert r.status_code in (401, 422)


def test_bad_extension_rejected():
    with _client() as c:
        files = [("image", ("evil.exe", b"MZ\x90\x00fake", "application/octet-stream"))]
        data = {"scale": "2"}
        r = c.post("/api/v1/commons/image/upscale", files=files, data=data)
        assert r.status_code == 400


def test_oversized_upload_rejected():
    """5 MB + 1 byte must 413."""
    with _client() as c:
        big = b"\x00" * (5 * 1024 * 1024 + 1)
        files = [("image", ("big.png", big, "image/png"))]
        data = {"scale": "2"}
        r = c.post("/api/v1/commons/image/upscale", files=files, data=data)
        assert r.status_code == 413
