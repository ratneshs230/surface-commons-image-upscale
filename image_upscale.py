"""Image Upscaler — Commons guild's image-tooling wave (continuation of bg_remove).

Open-source replacement for Topaz Gigapixel ($99 one-time, monthly upgrades) and
Let's Enhance ($9-29/mo plans). Two endpoints:

- ``POST /commons/image/upscale`` — body = multipart {image, scale: 2|3|4} -->
  returns upscaled PNG. Uses Real-ESRGAN (NCNN/Vulkan with CPU fallback,
  no PyTorch). Model selected by scale: x2 = animevideov3-x2, x3 =
  animevideov3-x3, x4 = realesrgan-x4plus (general photo).
- ``POST /commons/image/enhance`` — body = multipart {image} --> contrast +
  sharpen + light denoise via PIL only (no AI model). Cheap fast pass.

A separate ``GET /commons/image/upscale-health`` endpoint reports which
backend is live (``realesrgan_ncnn`` if NCNN/Vulkan loaded,
``pil_bicubic`` if we had to fall back to plain bicubic + unsharp).

Auth + audit follow the same pattern as ``commons.image.bg_remove``. The
public health endpoint is unauthenticated so the Commons landing page can
probe availability cheaply.
"""
from __future__ import annotations

import io
import json
import os
import time
import uuid
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Response,
    UploadFile,
)
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.service_auth import require_service_key_and_agent
from app.models.agent import Agent
from app.services.ledger import (
    InsufficientFundsError,
    record_transaction,
)


router = APIRouter(prefix="/api/v1/commons/image", tags=["commons-image-upscale"])


SC_COST = {
    "upscale": float(os.environ.get("IMAGE_UPSCALE_SC", "5")),
    "enhance": float(os.environ.get("IMAGE_ENHANCE_SC", "2")),
}

MAX_UPLOAD_BYTES = int(os.environ.get("IMAGE_UPSCALE_MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))
# Reject inputs whose long edge would explode beyond this after upscale.
MAX_OUTPUT_PIXELS = int(os.environ.get("IMAGE_UPSCALE_MAX_OUTPUT_PIXELS", str(8000 * 8000)))
ALLOWED_SCALES = (2, 3, 4)
ALLOWED_MIMES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
ALLOWED_EXTS = (".jpg", ".jpeg", ".png", ".webp")

# Model index map for realesrgan-ncnn-py:
#   0: realesr-animevideov3-x2  (scale 2)
#   1: realesr-animevideov3-x3  (scale 3)
#   2: realesr-animevideov3-x4  (scale 4 - anime/video)
#   3: realesrgan-x4plus-anime  (scale 4 - anime stills)
#   4: realesrgan-x4plus        (scale 4 - general photos, default for x4)
SCALE_TO_MODEL_IDX = {2: 0, 3: 1, 4: 4}

_engine_cache: dict = {}


def _get_engine(model_idx: int):
    """Lazy-load a realesrgan engine for a given model index. Cached per model.

    Uses CPU (gpuid=-1) - Surface VM has no discrete GPU. NCNN/Vulkan still
    runs on CPU via SwiftShader-style software backend; small inputs (<= 5 MB)
    upscale in seconds.
    """
    eng = _engine_cache.get(model_idx)
    if eng is not None:
        return eng
    from realesrgan_ncnn_py import Realesrgan
    eng = Realesrgan(gpuid=-1, model=model_idx)
    _engine_cache[model_idx] = eng
    return eng


def _capabilities() -> dict:
    caps = {
        "realesrgan_ncnn": False,
        "pil_enhance": False,
        "engines_loaded": [],
    }
    try:
        import realesrgan_ncnn_py  # noqa: F401
        caps["realesrgan_ncnn"] = True
    except Exception:
        pass
    try:
        import PIL  # noqa: F401
        caps["pil_enhance"] = True
    except Exception:
        pass
    caps["engines_loaded"] = sorted(_engine_cache.keys())
    return caps


def _backend_name() -> str:
    try:
        import realesrgan_ncnn_py  # noqa: F401
        return "realesrgan_ncnn"
    except Exception:
        return "pil_bicubic"


async def _read_upload(f: UploadFile, *, label: str) -> bytes:
    name = (f.filename or "").lower()
    if not name.endswith(ALLOWED_EXTS):
        raise HTTPException(
            status_code=400,
            detail={"error": "bad_extension", "field": label, "allowed": list(ALLOWED_EXTS)},
        )
    if f.content_type and f.content_type not in ALLOWED_MIMES:
        raise HTTPException(
            status_code=400,
            detail={"error": "bad_mime", "field": label, "got": f.content_type, "allowed": list(ALLOWED_MIMES)},
        )
    data = await f.read()
    if not data:
        raise HTTPException(status_code=400, detail=f"{label}_empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "upload_too_large",
                "limit_bytes": MAX_UPLOAD_BYTES,
                "got_bytes": len(data),
                "field": label,
            },
        )
    return data


async def _log_tool_call(
    db: AsyncSession,
    *,
    agent_id: str,
    tool_name: str,
    input_summary: dict,
    output_len: int,
    sc_cost: float,
    latency_ms: int,
    error: str | None = None,
) -> None:
    await db.execute(
        text(
            """
            INSERT INTO tool_calls
                (id, agent_id, tool_name, input_json, output_len,
                 sc_cost, latency_ms, error, created_at)
            VALUES
                (:id, :agent_id, :tool_name, CAST(:input_json AS JSONB),
                 :output_len, :sc_cost, :latency_ms, :error, NOW())
            """
        ),
        {
            "id": str(uuid.uuid4()),
            "agent_id": agent_id,
            "tool_name": tool_name,
            "input_json": json.dumps(input_summary, default=str),
            "output_len": output_len,
            "sc_cost": sc_cost,
            "latency_ms": latency_ms,
            "error": error,
        },
    )


async def _charge(
    db: AsyncSession,
    agent: Agent,
    *,
    tool: str,
    cost: float,
    ref: str,
) -> float:
    if cost <= 0:
        return 0.0
    try:
        await record_transaction(
            db,
            agent_id=agent.id,
            delta=-cost,
            reason=f"commons.image.{tool}",
            ref_id=ref,
            note=f"Image Upscaler: {tool}",
        )
    except InsufficientFundsError as e:
        raise HTTPException(
            status_code=402,
            detail={
                "error": "insufficient_sc",
                "balance": e.balance,
                "needed": cost,
                "tool": f"commons.image.{tool}",
            },
        )
    return cost


def _open_image(data: bytes):
    """Open + decode upload, normalize mode. Returns PIL.Image."""
    from PIL import Image
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={"error": "bad_image", "msg": str(e)[:200]},
        )
    # Realesrgan needs RGB or RGBA. Convert palette / CMYK / etc.
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGBA" if "A" in im.mode else "RGB")
    return im


def _upscale(data: bytes, scale: int) -> tuple[bytes, dict]:
    """Run real-ESRGAN upscale at the requested integer scale.

    Falls back to PIL bicubic + unsharp if the NCNN backend is not available
    (mostly relevant in CI / offline test envs).
    """
    from PIL import Image, ImageFilter

    im = _open_image(data)
    in_w, in_h = im.size
    out_w, out_h = in_w * scale, in_h * scale

    if out_w * out_h > MAX_OUTPUT_PIXELS:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "output_too_large",
                "in_size": [in_w, in_h],
                "scale": scale,
                "out_pixels": out_w * out_h,
                "limit_pixels": MAX_OUTPUT_PIXELS,
            },
        )

    backend = _backend_name()
    t0 = time.monotonic()
    if backend == "realesrgan_ncnn":
        model_idx = SCALE_TO_MODEL_IDX[scale]
        eng = _get_engine(model_idx)
        # ncnn doesn't accept RGBA; flatten alpha (rare for upscale inputs).
        rgb = im.convert("RGB") if im.mode != "RGB" else im
        out_im = eng.process_pil(rgb)
        # Engine model fixed at native scale (2/3/4) - output already correct.
    else:
        # Cheap fallback: bicubic resize + unsharp mask.
        rgb = im.convert("RGB") if im.mode != "RGB" else im
        out_im = rgb.resize((out_w, out_h), Image.Resampling.BICUBIC)
        out_im = out_im.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=2))

    elapsed_ms = int((time.monotonic() - t0) * 1000)

    buf = io.BytesIO()
    out_im.save(buf, format="PNG", optimize=True)
    body = buf.getvalue()
    return body, {
        "in_width": in_w,
        "in_height": in_h,
        "out_width": out_im.size[0],
        "out_height": out_im.size[1],
        "scale": scale,
        "backend": backend,
        "engine_ms": elapsed_ms,
        "in_bytes": len(data),
        "out_bytes": len(body),
    }


def _enhance(data: bytes) -> tuple[bytes, dict]:
    """PIL-only enhance pass: contrast + colour + sharpness + light denoise.

    No AI model - deterministic, cheap. Useful for "make it pop a little"
    type asks where the input is already at the desired resolution.
    """
    from PIL import ImageEnhance, ImageFilter

    im = _open_image(data)
    in_w, in_h = im.size

    # Light denoise first (smooths JPEG artefacts before we sharpen).
    work = im.filter(ImageFilter.MedianFilter(size=3))
    work = ImageEnhance.Contrast(work).enhance(1.15)
    work = ImageEnhance.Color(work).enhance(1.10)
    work = ImageEnhance.Sharpness(work).enhance(1.40)
    # Final unsharp pass to recover edge detail eaten by the median filter.
    work = work.filter(ImageFilter.UnsharpMask(radius=1.5, percent=80, threshold=3))

    buf = io.BytesIO()
    work.save(buf, format="PNG", optimize=True)
    body = buf.getvalue()
    return body, {
        "in_width": in_w,
        "in_height": in_h,
        "out_width": work.size[0],
        "out_height": work.size[1],
        "in_bytes": len(data),
        "out_bytes": len(body),
        "ops": ["median3", "contrast=1.15", "color=1.10", "sharpness=1.40", "unsharp"],
    }


@router.get("/upscale-health")
async def upscale_health() -> dict:
    """Public health probe - separate from /commons/image/health (bg-remove)
    so the Commons landing page can verify each tool independently."""
    caps = _capabilities()
    backend = _backend_name()
    return {
        "service": "commons.image.upscale",
        "version": "1.0.0",
        "available": caps,
        "backend": backend,
        "all_available": caps["realesrgan_ncnn"] or caps["pil_enhance"],
        "sc_cost": SC_COST,
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "max_output_pixels": MAX_OUTPUT_PIXELS,
        "allowed_scales": list(ALLOWED_SCALES),
        "allowed_mimes": sorted(ALLOWED_MIMES),
    }


@router.post("/upscale")
async def upscale(
    image: Annotated[UploadFile, File(description="Image (jpg/png/webp, <=5MB)")],
    scale: Annotated[int, Form(description="Integer upscale factor: 2, 3, or 4")] = 2,
    db: AsyncSession = Depends(get_db),
    agent: Agent = Depends(require_service_key_and_agent),
) -> Response:
    if scale not in ALLOWED_SCALES:
        raise HTTPException(
            status_code=400,
            detail={"error": "bad_scale", "got": scale, "allowed": list(ALLOWED_SCALES)},
        )
    t0 = time.monotonic()
    ref = str(uuid.uuid4())
    data = await _read_upload(image, label="image")
    in_size = len(data)

    cost = SC_COST["upscale"]
    await _charge(db, agent, tool="upscale", cost=cost, ref=ref)

    try:
        try:
            body, stats = _upscale(data, scale)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={"error": "upscale_failed", "msg": str(e)[:300]},
            )

        await _log_tool_call(
            db,
            agent_id=agent.id,
            tool_name="commons.image.upscale",
            input_summary={"in_bytes": in_size, "scale": scale, **stats},
            output_len=len(body),
            sc_cost=cost,
            latency_ms=int((time.monotonic() - t0) * 1000),
        )
        await db.commit()

        return Response(
            content=body,
            media_type="image/png",
            headers={
                "Content-Disposition": f'attachment; filename="upscaled-x{scale}.png"',
                "X-SC-Cost": str(cost),
                "X-Backend": stats["backend"],
                "X-Scale": str(scale),
                "X-In-Width": str(stats["in_width"]),
                "X-In-Height": str(stats["in_height"]),
                "X-Out-Width": str(stats["out_width"]),
                "X-Out-Height": str(stats["out_height"]),
                "X-In-Bytes": str(in_size),
                "X-Out-Bytes": str(len(body)),
                "X-Engine-Ms": str(stats["engine_ms"]),
            },
        )
    except HTTPException:
        await db.rollback()
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail={"error": "upscale_failed", "msg": str(e)[:200]})


@router.post("/enhance")
async def enhance(
    image: Annotated[UploadFile, File(description="Image (jpg/png/webp, <=5MB)")],
    db: AsyncSession = Depends(get_db),
    agent: Agent = Depends(require_service_key_and_agent),
) -> Response:
    t0 = time.monotonic()
    ref = str(uuid.uuid4())
    data = await _read_upload(image, label="image")
    in_size = len(data)

    cost = SC_COST["enhance"]
    await _charge(db, agent, tool="enhance", cost=cost, ref=ref)

    try:
        try:
            body, stats = _enhance(data)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={"error": "enhance_failed", "msg": str(e)[:300]},
            )

        await _log_tool_call(
            db,
            agent_id=agent.id,
            tool_name="commons.image.enhance",
            input_summary={"in_bytes": in_size, **stats},
            output_len=len(body),
            sc_cost=cost,
            latency_ms=int((time.monotonic() - t0) * 1000),
        )
        await db.commit()

        return Response(
            content=body,
            media_type="image/png",
            headers={
                "Content-Disposition": 'attachment; filename="enhanced.png"',
                "X-SC-Cost": str(cost),
                "X-Width": str(stats["in_width"]),
                "X-Height": str(stats["in_height"]),
                "X-In-Bytes": str(in_size),
                "X-Out-Bytes": str(len(body)),
                "X-Ops": ",".join(stats["ops"]),
            },
        )
    except HTTPException:
        await db.rollback()
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail={"error": "enhance_failed", "msg": str(e)[:200]})
