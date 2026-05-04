# surface-commons-image-upscale

Open-source image upscaler + enhancer — Commons guild's replacement for [Topaz Gigapixel](https://www.topazlabs.com/topaz-photo-ai) and [Let's Enhance](https://letsenhance.io).

Two endpoints. Real-ESRGAN (via NCNN/Vulkan, CPU-friendly, no PyTorch) handles the heavy upscale. PIL handles the cheap "make it pop" enhance pass. SC-billed per call, audited to the Surface ledger.

## Endpoints

| Method | Path | Cost | Body | Returns |
|---|---|---|---|---|
| `POST` | `/commons/image/upscale` | 5 SC | multipart `{image, scale: 2 \| 3 \| 4}` | `image/png` |
| `POST` | `/commons/image/enhance` | 2 SC | multipart `{image}` | `image/png` |
| `GET`  | `/commons/image/upscale-health` | free | — | `application/json` capability probe |

The health endpoint is intentionally separate from `bg-remove`'s `/commons/image/health` so the Commons landing page can probe upscale availability independently.

## Stack

- [realesrgan-ncnn-py](https://pypi.org/project/realesrgan-ncnn-py/) 2.0.0 — Vulkan/NCNN binding, CPU-runnable, ~46 MB wheel
  - Models bundled with the wheel, no separate download:
    - scale 2 → `realesr-animevideov3-x2` (fast, generic)
    - scale 3 → `realesr-animevideov3-x3`
    - scale 4 → `realesrgan-x4plus` (best general photo)
- [Pillow](https://python-pillow.org/) for the PIL-only `/enhance` pass (median3 → contrast 1.15 → colour 1.10 → sharpness 1.40 → unsharp)
- FastAPI router, attaches under any Surface app via `include_router`
- Engines lazy-loaded per scale, cached across calls

## Fallback

If `realesrgan_ncnn_py` import fails (CI / sandboxed environments), the upscale path drops to **PIL bicubic + UnsharpMask**. Lower quality but ships the bytes. The health endpoint reports the active backend (`realesrgan_ncnn` vs `pil_bicubic`).

## Limits

- Input upload ≤ 5 MB (`IMAGE_UPSCALE_MAX_UPLOAD_BYTES`)
- Output capped at 64 MP / 8000×8000 (`IMAGE_UPSCALE_MAX_OUTPUT_PIXELS`)
- Allowed scales: 2, 3, 4 (integer)
- Allowed mimes: `image/jpeg`, `image/png`, `image/webp`

## System dependency

`libomp5` (OpenMP runtime, used by NCNN). On Ubuntu / Debian:

```bash
sudo apt-get install -y libomp5
```

## SC overrides

| Env | Default | Effect |
|---|---|---|
| `IMAGE_UPSCALE_SC` | `5` | per-call cost of `/upscale` |
| `IMAGE_ENHANCE_SC` | `2` | per-call cost of `/enhance` |
| `IMAGE_UPSCALE_MAX_UPLOAD_BYTES` | `5242880` | upload byte limit |
| `IMAGE_UPSCALE_MAX_OUTPUT_PIXELS` | `64000000` | output area limit |

## Source of truth

This repo is a public mirror. Canonical home is `app/api/commons/image_upscale.py` in [Surface-V2](https://github.com/ratneshs230/Surface-V2). Pull-requests open against Surface-V2.

## License

MIT — see Surface Commons charter at [github.com/surface-commons](https://github.com/surface-commons).
