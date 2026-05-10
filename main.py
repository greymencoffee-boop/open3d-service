"""
open3d-service/main.py  v6
FastAPI service: LiDAR depth maps + Neural Depth (Depth Anything v2) → volumetric estimates.
Endpoints: /health  /process  /process-photo  /warmup
Pipeline aligned with LiDARCalorieCam (Fujita & Yanai, 2025).
"""

import base64
import logging
import os
import tempfile
import threading
import traceback
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("open3d-service")

# ---------------------------------------------------------------------------
# Optional heavy imports — each is tried once at startup so per-request paths
# know which libraries are available without repeated try/except overhead.
# ---------------------------------------------------------------------------

try:
    import open3d as o3d

    OPEN3D_AVAILABLE = True
    logger.info("open3d loaded successfully")
except Exception as _e:
    o3d = None  # type: ignore[assignment]
    OPEN3D_AVAILABLE = False
    logger.warning("open3d not available: %s", _e)

try:
    import trimesh

    TRIMESH_AVAILABLE = True
    logger.info("trimesh loaded successfully")
except Exception as _e:
    trimesh = None  # type: ignore[assignment]
    TRIMESH_AVAILABLE = False
    logger.warning("trimesh not available: %s", _e)

try:
    from scipy.spatial import ConvexHull, Delaunay

    SCIPY_AVAILABLE = True
    logger.info("scipy loaded successfully")
except Exception as _e:
    ConvexHull = None  # type: ignore[assignment]
    Delaunay = None    # type: ignore[assignment]
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available: %s", _e)

try:
    from sklearn.cluster import DBSCAN

    SKLEARN_AVAILABLE = True
    logger.info("scikit-learn loaded successfully")
except Exception as _e:
    DBSCAN = None  # type: ignore[assignment]
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available: %s", _e)

try:
    import io as _io
    import torch
    from PIL import Image as PILImage
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    DEPTH_ANYTHING_AVAILABLE = True
    logger.info("Depth Anything v2 imports ready")
except Exception as _e:
    _io = None         # type: ignore[assignment]
    torch = None       # type: ignore[assignment]
    PILImage = None    # type: ignore[assignment]
    AutoImageProcessor = None           # type: ignore[assignment]
    AutoModelForDepthEstimation = None  # type: ignore[assignment]
    DEPTH_ANYTHING_AVAILABLE = False
    logger.warning("Depth Anything v2 not available: %s", _e)

# Singletons — loaded once, reused across all requests.
_DA_PROCESSOR = None
_DA_MODEL = None
_DA_LOCK = threading.Lock()   # prevents double-load under concurrent requests


def _get_da_model():
    """Load Depth Anything v2 Metric Indoor Small (thread-safe, cached after first call)."""
    global _DA_PROCESSOR, _DA_MODEL
    if _DA_PROCESSOR is not None:
        return _DA_PROCESSOR, _DA_MODEL
    with _DA_LOCK:
        if _DA_PROCESSOR is None:  # double-checked locking
            logger.info("Loading Depth Anything v2 model...")
            _DA_PROCESSOR = AutoImageProcessor.from_pretrained(
                "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
            )
            _DA_MODEL = AutoModelForDepthEstimation.from_pretrained(
                "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
            )
            _DA_MODEL.eval()
            logger.info("Depth Anything v2 model ready.")
    return _DA_PROCESSOR, _DA_MODEL


# Background preload — starts downloading the model the moment the container
# starts, so by the time the first user request arrives the model is ready.
if DEPTH_ANYTHING_AVAILABLE:
    _preload = threading.Thread(target=_get_da_model, daemon=True, name="da-preload")
    _preload.start()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEPTH_MAP_WIDTH = 64
DEPTH_MAP_HEIGHT = 48
DEPTH_MIN_M = 0.05   # metres (byte = 1)
DEPTH_MAX_M = 1.50   # metres (byte = 255); range = 1.45 m
DEPTH_RANGE_M = 1.45
DEFAULT_INTRINSICS = [55.0, 55.0, 32.0, 24.0]  # fx, fy, cx, cy

MIN_VALID_POINTS = 50
MIN_FOOD_POINTS = 20
TABLE_MARGIN_M = 0.005   # 5 mm above table plane

# ---------------------------------------------------------------------------
# Per-category linear regression: W_grams = a * vol_ml + b
# Coefficients from LiDARCalorieCam paper Table 1 (Japanese foods) +
# density-derived approximations for common Western foods.
# ---------------------------------------------------------------------------

CATEGORY_REGRESSION: dict[str, tuple[float, float]] = {
    # Paper's 10 Japanese categories
    "karaage":            (0.45, 11.4),
    "croquette":          (0.52,  8.7),
    "yakitori":           (0.48,  6.2),
    "hot dog":            (0.61,  5.1),
    "toast":              (0.38,  4.9),
    "yakisoba":           (0.44, 12.1),
    "potato salad":       (0.57,  9.3),
    "onigiri":            (0.71,  3.8),
    "tamagoyaki":         (0.82,  2.6),
    "sauteed vegetables": (0.35, 15.2),
    # Common Western foods (density-derived approximations)
    "chicken":            (0.45, 11.0),
    "rice":               (0.70,  4.0),
    "salad":              (0.30, 15.0),
    "bread":              (0.38,  5.0),
    "egg":                (0.82,  2.5),
    "pasta":              (0.44, 12.0),
    "noodle":             (0.44, 12.0),
    "potato":             (0.57,  9.0),
    "steak":              (0.55,  8.0),
    "fish":               (0.50,  7.0),
}


def _estimate_weight(
    vol_ml: float, category: Optional[str]
) -> tuple[Optional[float], Optional[str]]:
    """Return (weight_g, matched_category_key) via fuzzy substring match."""
    if not category:
        return None, None
    key = category.lower().strip()
    for cat_key, (a, b) in CATEGORY_REGRESSION.items():
        if cat_key in key or key in cat_key:
            return round(a * vol_ml + b, 1), cat_key
    return None, None


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ProcessRequest(BaseModel):
    depthMapBase64: str
    intrinsics: Optional[list[float]] = Field(default=None)
    scanId: str = ""
    category: Optional[str] = None   # food name for per-category regression


class ProcessResponse(BaseModel):
    volumeMl: Optional[float] = None
    methodBreakdown: dict[str, Any] = {}
    confidence: Optional[float] = None
    plyBase64: Optional[str] = None
    estimatedWeightG: Optional[float] = None    # from W = aV + b regression
    regressionCategory: Optional[str] = None   # matched regression key
    error: Optional[str] = None
    timedOut: bool = False
    depthRawBase64: Optional[str] = None        # compact 64×48 uint8 — same format as LiDAR
    depthHeatmapBase64: Optional[str] = None    # colorized JPEG heatmap for Claude visual context
    # Iteration 14 Layer 6 — fractional spread (max - min) / max across the valid
    # ensemble methods. Same semantic as LiDAR's divergence field. Drives the
    # client-side ENSEMBLE AGREEMENT / DIVERGENCE language in the Claude prompt.
    # Low (< 0.20) = methods agree → trust volume. High (> 0.50) = methods
    # disagree → AI prefers visual estimate (Universal Accuracy Rules Step 6).
    divergence: Optional[float] = None


class ProcessPhotoRequest(BaseModel):
    imageBase64: str            # Base64-encoded JPEG from the phone camera
    scanId: str = ""
    category: Optional[str] = None   # food name for weight regression


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="LiDAR Depth-Map Processor", version="6.0.0")
logger.info("open3d-service v6 starting — endpoints: /health /process /process-photo /warmup")

# Read at startup — Railway injects this as a service variable.
_SERVICE_SECRET_KEY: str | None = os.environ.get("SERVICE_SECRET_KEY")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/warmup")
def warmup() -> dict[str, object]:
    """Wake the container and report whether the Depth Anything model is ready.
    Called by the client when the user enters Neural Depth Scan mode, giving the
    model ~10-20 s to finish loading before the photo is taken."""
    ready = _DA_PROCESSOR is not None
    if DEPTH_ANYTHING_AVAILABLE and not ready:
        logger.info("/warmup called — model still loading in background thread")
    elif DEPTH_ANYTHING_AVAILABLE and ready:
        logger.info("/warmup called — model already ready")
    return {
        "status": "ok",
        "model_ready": ready,
        "depth_anything_available": DEPTH_ANYTHING_AVAILABLE,
    }


@app.post("/process", response_model=ProcessResponse)
def process(req: ProcessRequest, request: Request) -> ProcessResponse:
    # Validate shared secret when configured (always set in production).
    if _SERVICE_SECRET_KEY:
        incoming = request.headers.get("x-service-key", "")
        if incoming != _SERVICE_SECRET_KEY:
            return JSONResponse(status_code=401, content={"error": "Unauthorized"})

    scan_id = req.scanId or "<no-id>"
    logger.info("Processing LiDAR scan %s", scan_id)

    try:
        return _run_pipeline(req)
    except Exception:
        msg = traceback.format_exc()
        logger.error("Unhandled exception in scan %s:\n%s", scan_id, msg)
        return ProcessResponse(
            error=f"Internal error: {traceback.format_exc(limit=3)}",
            timedOut=False,
        )


def _height_field_volume_ml(
    depth_small: np.ndarray,
    fx: float = DEFAULT_INTRINSICS[0],
    fy: float = DEFAULT_INTRINSICS[1],
) -> tuple:
    """Method 6 — height-field integration of a metric depth map.

    Works for any camera angle (overhead, angled, etc.).  The background plane
    is estimated as the 85th-percentile depth (far = table/plate) and food
    height at each pixel is max(0, bg_z − pixel_z).  Most reliable when some
    plate/table is visible; still returns a useful approximation otherwise.

    Returns (volume_ml | None, confidence ∈ [0, 1]).
    """
    d = depth_small.ravel()
    bg_z       = float(np.percentile(d, 85))
    food_z_min = float(np.percentile(d, 5))
    z_range    = bg_z - food_z_min          # depth relief of the food

    if z_range < 0.003:                     # < 3 mm — scene too flat
        logger.warning(
            "Height-field: z_range=%.1f mm — scene too flat, skipping",
            z_range * 1000,
        )
        return None, 0.0

    height     = np.maximum(0.0, bg_z - depth_small)           # H×W, metres
    threshold  = max(0.003, z_range * 0.05)
    significant = height > threshold

    n_sig = int(significant.sum())
    if n_sig < 30:
        logger.warning("Height-field: only %d significant pixels — skipping", n_sig)
        return None, 0.0

    # Pixel footprint in world space at each pixel's depth: (z/fx) × (z/fy) m²
    pixel_area = (depth_small / fx) * (depth_small / fy)
    vol_m3     = float(np.sum(height * pixel_area * significant))
    vol_ml     = vol_m3 * 1_000_000.0

    food_frac  = float(n_sig) / float(significant.size)
    # Confidence: larger depth relief & food coverage → more confident (cap at 0.75)
    confidence = float(min(0.75, z_range * 5.0 * (0.5 + food_frac)))

    logger.info(
        "Height-field (method 6): z_range=%.1f mm  food_frac=%.0f%%  → %.1f ml  conf=%.2f",
        z_range * 1000, food_frac * 100, vol_ml, confidence,
    )
    return vol_ml, confidence


def _colorize_depth_heatmap(depth_clamped: np.ndarray) -> str:
    """Render a metric depth map as a colorized JPEG heatmap for Claude visual input.

    Warm colours (red/orange) = near camera = food surface.
    Cool colours (blue/cyan) = far from camera = plate/table background.

    Uses a jet-like colormap implemented in pure numpy/PIL (no matplotlib).
    Upsamples 64×48 → 320×240 (5× nearest-neighbor) and encodes as JPEG.
    Returns base64-encoded JPEG string.
    """
    d_min = float(depth_clamped.min())
    d_max = float(depth_clamped.max())
    if d_max - d_min < 1e-4:
        norm = np.zeros_like(depth_clamped, dtype=np.float32)
    else:
        norm = (depth_clamped - d_min) / (d_max - d_min)

    # Invert: low depth (near/food) → 1.0 (warm), high depth (far/background) → 0.0 (cool)
    d = (1.0 - norm).astype(np.float32)

    # Jet-like colormap: d=1 (near/food) → red, d=0.5 → green, d=0 (far/bg) → blue
    r = np.clip(4.0 * d - 2.0, 0.0, 1.0)
    g = np.clip(np.minimum(4.0 * d, 4.0 - 4.0 * d), 0.0, 1.0)
    b = np.clip(2.0 - 4.0 * d, 0.0, 1.0)

    rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    pil_heatmap = PILImage.fromarray(rgb, "RGB")
    # If input is already at target size (320×240) skip the resize.
    # If input is the 64×48 pipeline map, upscale 5× with nearest-neighbor.
    if (pil_heatmap.width, pil_heatmap.height) != (320, 240):
        pil_heatmap = pil_heatmap.resize((320, 240), PILImage.NEAREST)

    buf = _io.BytesIO()
    pil_heatmap.save(buf, format="JPEG", quality=75)
    heatmap_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    logger.info(
        "Depth heatmap generated: %dx%d → 320×240 JPEG, %d bytes",
        rgb.shape[1], rgb.shape[0], len(buf.getvalue()),
    )
    return heatmap_b64


@app.post("/process-photo", response_model=ProcessResponse)
def process_photo(req: ProcessPhotoRequest, request: Request) -> ProcessResponse:
    """Neural Depth Scan — runs Depth Anything v2 Metric Indoor on a JPEG photo,
    produces a 64×48 metric depth map, then feeds it to the same 5-method
    Open3D ensemble used by the LiDAR pipeline."""
    if _SERVICE_SECRET_KEY:
        incoming = request.headers.get("x-service-key", "")
        if incoming != _SERVICE_SECRET_KEY:
            return JSONResponse(status_code=401, content={"error": "Unauthorized"})

    scan_id = req.scanId or "<no-id>"
    logger.info("Processing neural depth scan %s", scan_id)

    try:
        return _process_photo_depth(req)
    except Exception:
        msg = traceback.format_exc()
        logger.error("Unhandled exception in neural depth scan %s:\n%s", scan_id, msg)
        return ProcessResponse(
            error=f"Internal error: {traceback.format_exc(limit=3)}",
            timedOut=False,
        )


def _process_photo_depth(req: ProcessPhotoRequest) -> ProcessResponse:
    """Convert a JPEG photo → metric depth map → 5-method volume ensemble."""
    if not DEPTH_ANYTHING_AVAILABLE:
        return ProcessResponse(
            error="Depth Anything v2 model not available on this server.",
            timedOut=False,
        )

    # 1. Decode JPEG from base64
    try:
        img_bytes = base64.b64decode(req.imageBase64)
        pil_img = PILImage.open(_io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        logger.error("Photo decode failed in scan %s: %s", req.scanId, e)
        return ProcessResponse(error=f"Photo decode error: {e}", timedOut=False)

    # 2. Run Depth Anything v2 Metric Indoor → float32 metres (absolute metric depth)
    # Heatmap resolution: 320×240 — the target size for Claude's second image.
    # We interpolate the model output directly to this resolution rather than
    # going via the full original image size (e.g. 1366×1024) and then shrinking.
    # Both paths give the same 64×48 pipeline input but this avoids allocating
    # a large intermediate tensor, saving memory on the Railway container.
    HEATMAP_W, HEATMAP_H = 320, 240

    try:
        processor, model = _get_da_model()
        inputs = processor(images=pil_img, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            # Upsample model output directly to heatmap resolution (320×240).
            # The processor already resized pil_img to 518×518 for the forward
            # pass, so the original image dimensions don't affect inference time.
            # Skipping the original-size intermediate tensor saves ~5-15 MB RAM.
            depth_tensor = torch.nn.functional.interpolate(
                outputs.predicted_depth.unsqueeze(1),
                size=(HEATMAP_H, HEATMAP_W),   # (H=240, W=320)
                mode="bicubic",
                align_corners=False,
            ).squeeze()  # shape: (240, 320), float32, metres
        depth_heatmap_np = depth_tensor.cpu().numpy().astype(np.float32)
        logger.info(
            "Depth Anything v2 inference done. depth range: [%.4fm, %.4fm]  "
            "mean=%.4fm  heatmap_shape=%s",
            float(depth_heatmap_np.min()), float(depth_heatmap_np.max()),
            float(depth_heatmap_np.mean()), depth_heatmap_np.shape,
        )
    except Exception as e:
        logger.error("Depth Anything inference failed: %s", e)
        return ProcessResponse(error=f"Depth estimation failed: {e}", timedOut=False)

    # 3. Downsample 320×240 → 64×48 for the 5-method pipeline.
    try:
        depth_small = np.array(
            PILImage.fromarray(depth_heatmap_np).resize(
                (DEPTH_MAP_WIDTH, DEPTH_MAP_HEIGHT), PILImage.LANCZOS
            ),
            dtype=np.float32,
        )
    except Exception as e:
        logger.error("Depth downsample failed: %s", e)
        return ProcessResponse(error=f"Downsample error: {e}", timedOut=False)

    # 4. Clamp pipeline map to the valid depth range (0.05–1.50 m).
    depth_clamped = np.clip(depth_small, DEPTH_MIN_M, DEPTH_MAX_M)

    # 4b. Generate colorized JPEG heatmap from the 320×240 map (better quality
    #     than 5× upsampling the 64×48 pipeline map with nearest-neighbor).
    #     Always produced — returned even when volume estimation fails so Claude
    #     can use depth information for 3D visual reasoning about the food.
    try:
        depth_heatmap_clamped = np.clip(depth_heatmap_np, DEPTH_MIN_M, DEPTH_MAX_M)
        heatmap_b64 = _colorize_depth_heatmap(depth_heatmap_clamped)
    except Exception as e:
        logger.warning("Heatmap generation failed (non-fatal): %s", e)
        heatmap_b64 = None

    # 5. Encode as uint8 [1–255] — the same byte format as the ARKit compact depth map.
    #    byte = 0 means "invalid"; all monocular pixels are valid so we map to [1, 255].
    byte_array = np.clip(
        np.round(1.0 + (depth_clamped - DEPTH_MIN_M) / DEPTH_RANGE_M * 254.0)
        .astype(np.uint8),
        1,
        255,
    )

    # 6. Height-field volume (method 6) — works for any camera angle.
    #    Computed here before encoding so we have the float32 map available.
    hf_vol_ml, hf_conf = _height_field_volume_ml(depth_clamped)

    # 7. Encode as uint8 and hand off to the 5-method pipeline.
    depth_b64 = base64.b64encode(byte_array.tobytes()).decode("utf-8")
    synthetic_req = ProcessRequest(
        depthMapBase64=depth_b64,
        intrinsics=None,   # use DEFAULT_INTRINSICS — correct for a phone at 20–60 cm
        scanId=req.scanId,
        category=req.category,
    )
    logger.info(
        "Neural depth scan %s → depth clamped [%.3fm, %.3fm], forwarding to pipeline",
        req.scanId or "<no-id>",
        float(depth_clamped.min()),
        float(depth_clamped.max()),
    )
    result = _run_pipeline(synthetic_req)

    # Augment the method breakdown with the height-field result regardless.
    breakdown = dict(result.methodBreakdown)
    if hf_vol_ml is not None:
        breakdown["height_field"] = round(hf_vol_ml, 2)

    # If the 5-method pipeline failed but height-field succeeded, use it as
    # the final volume with a transparency note in the breakdown.
    if result.error and hf_vol_ml is not None and 10.0 < hf_vol_ml < 8000.0:
        logger.info(
            "5-method pipeline failed (%s) — using height-field fallback: %.1f ml",
            result.error, hf_vol_ml,
        )
        wt_g, wt_cat = _estimate_weight(hf_vol_ml, req.category)
        return ProcessResponse(
            volumeMl=hf_vol_ml,
            methodBreakdown=breakdown,
            confidence=hf_conf,
            estimatedWeightG=wt_g,
            regressionCategory=wt_cat,
            depthRawBase64=depth_b64,
            depthHeatmapBase64=heatmap_b64,
            timedOut=False,
            # Iteration 14 Layer 6 — height-field fallback path: divergence is
            # n/a (single-method estimate). Leave None so the client's prompt
            # treats the volume as un-cross-validated.
        )

    # 8. Return pipeline result (success or hard failure).
    # Include compact depth bytes and the colorized heatmap.
    # depthHeatmapBase64 is always included so the client can pass it to Claude
    # as a second image even when volumeMl is None (volume estimation failed).
    return ProcessResponse(
        volumeMl=result.volumeMl,
        methodBreakdown=breakdown,
        confidence=result.confidence,
        plyBase64=result.plyBase64,
        estimatedWeightG=result.estimatedWeightG,
        regressionCategory=result.regressionCategory,
        error=result.error,
        timedOut=result.timedOut,
        depthRawBase64=depth_b64,
        depthHeatmapBase64=heatmap_b64,
        # Iteration 14 Layer 6 — propagate divergence from the 5-method pipeline.
        divergence=result.divergence,
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _run_pipeline(req: ProcessRequest) -> ProcessResponse:
    # 1. Decode base64 → uint8 depth map → float32 depths
    try:
        raw = base64.b64decode(req.depthMapBase64)
        byte_array = np.frombuffer(raw, dtype=np.uint8).reshape(
            DEPTH_MAP_HEIGHT, DEPTH_MAP_WIDTH
        )
    except Exception as e:
        logger.error("Depth map decode failed: %s", e)
        return ProcessResponse(error=f"Depth map decode error: {e}", timedOut=False)

    depth = np.where(
        byte_array == 0,
        np.nan,
        DEPTH_MIN_M + ((byte_array.astype(np.float32) - 1.0) / 254.0) * DEPTH_RANGE_M,
    )

    # 2. Backproject to 3-D
    fx, fy, cx, cy = (
        req.intrinsics
        if req.intrinsics and len(req.intrinsics) == 4
        else DEFAULT_INTRINSICS
    )
    points = _backproject(depth, fx, fy, cx, cy)

    if points.shape[0] < MIN_VALID_POINTS:
        return ProcessResponse(
            error=f"Too few valid depth pixels ({points.shape[0]} < {MIN_VALID_POINTS})",
            timedOut=False,
        )

    # 3. DBSCAN noise removal (paper: ε=5 mm, minPts=5)
    points = _denoise_dbscan(points)

    if points.shape[0] < MIN_VALID_POINTS:
        return ProcessResponse(
            error=f"Too few points after denoising ({points.shape[0]} < {MIN_VALID_POINTS})",
            timedOut=False,
        )

    # 4. Table plane + food points
    table_y = float(np.percentile(points[:, 1], 10))
    food_mask = points[:, 1] > (table_y + TABLE_MARGIN_M)
    food_pts = points[food_mask]

    low_pt_flag = food_pts.shape[0] < MIN_FOOD_POINTS
    if low_pt_flag:
        logger.warning(
            "Only %d food points found (< %d); proceeding with all points",
            food_pts.shape[0],
            MIN_FOOD_POINTS,
        )
        food_pts = points

    # Build Open3D point cloud once (used by methods 3 and 5)
    pcd = _make_pcd(food_pts)

    # 5. Run the paper's 5 volume methods
    breakdown: dict[str, Any] = {}

    m_convex   = _method_convex(food_pts)
    m_delaunay = _method_delaunay(food_pts)
    m_alpha    = _method_alpha_shape(pcd)
    m_spline   = _method_spline(food_pts)
    m_poisson  = _method_poisson(pcd)

    breakdown["convex"]   = round(m_convex,   2) if m_convex   is not None else None
    breakdown["delaunay"] = round(m_delaunay, 2) if m_delaunay is not None else None
    breakdown["alpha"]    = round(m_alpha,    2) if m_alpha    is not None else None
    breakdown["spline"]   = round(m_spline,   2) if m_spline   is not None else None
    breakdown["poisson"]  = round(m_poisson,  2) if m_poisson  is not None else None

    # 6. Paper ensemble: confidence = exp(−σ/μ)
    # Range widened from 40–1600 ml: LiDAR large-plate and neural overhead shots
    # can legitimately produce volumes > 1600 ml (e.g. whole pizza, large bowl).
    valid = {k: v for k, v in breakdown.items() if v is not None and 10.0 < v < 8000.0}

    if not valid:
        return ProcessResponse(
            error="All volume methods failed or produced out-of-range estimates",
            methodBreakdown=breakdown,
            timedOut=False,
        )

    vals      = list(valid.values())
    mu        = float(np.mean(vals))
    sigma     = float(np.std(vals))
    cv        = sigma / mu if mu > 0 else 1.0
    confidence = float(np.exp(-cv))   # paper formula

    # Iteration 14 Layer 6 — analog to LiDAR build 134's hull-drop. Convex
    # hull tends to over-estimate when food sits in a bowl-shaped plate
    # (wraps the rim's depth shadow). Move convex out of the primary
    # selection path: prefer poisson at high confidence, otherwise trimmed
    # mean of the non-convex methods. Convex stays as last-resort fallback
    # if no other method produced a valid estimate.
    non_convex_valid = {k: v for k, v in valid.items() if k != "convex"}
    if confidence >= 0.8 and valid.get("poisson") is not None:
        final_vol = float(valid["poisson"])
    elif len(non_convex_valid) >= 3:
        # Trimmed mean — drop highest and lowest of the non-convex methods.
        # Same rationale as LiDAR's trimmed-mean ensemble: stable to a single
        # method blowing up.
        nc_vals = sorted(non_convex_valid.values())
        final_vol = float(np.mean(nc_vals[1:-1]))
    elif non_convex_valid:
        final_vol = float(np.mean(list(non_convex_valid.values())))
    elif valid.get("convex") is not None:
        # Last resort — only convex produced a valid estimate
        final_vol = float(valid["convex"])
    else:
        final_vol = mu

    # Iteration 14 Layer 6 — divergence signal across all valid methods. Same
    # semantic as LiDAR's divergence (max - min) / max. Returned to the client
    # so the prompt-builder fires the existing ENSEMBLE AGREEMENT / DIVERGENCE
    # language (built for LiDAR; now applies to Neural Depth too).
    divergence_val: Optional[float] = None
    if vals and max(vals) > 0:
        divergence_val = float((max(vals) - min(vals)) / max(vals))

    # Clamp confidence if we had very few food points
    if low_pt_flag:
        confidence = max(0.30, confidence - 0.15)

    # 7. Per-category weight regression
    weight_g, matched_cat = _estimate_weight(final_vol, req.category)

    # 8. PLY export
    ply_b64 = _export_ply_base64(food_pts)

    logger.info(
        "Scan %s → vol=%.1f ml conf=%.2f methods=%d/%d divergence=%s%s",
        req.scanId or "<no-id>",
        final_vol,
        confidence,
        len(valid),
        5,
        f"{divergence_val:.2f}" if divergence_val is not None else "n/a",
        f" weight={weight_g}g ({matched_cat})" if weight_g else "",
    )

    return ProcessResponse(
        volumeMl=round(final_vol, 2),
        methodBreakdown=breakdown,
        confidence=round(confidence, 3),
        plyBase64=ply_b64,
        estimatedWeightG=weight_g,
        regressionCategory=matched_cat,
        divergence=round(divergence_val, 4) if divergence_val is not None else None,
    )


# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------


def _backproject(
    depth: np.ndarray, fx: float, fy: float, cx: float, cy: float
) -> np.ndarray:
    """Pinhole back-projection. Returns Nx3 float32 world points (y-up)."""
    rows, cols = np.indices(depth.shape)
    valid = ~np.isnan(depth)

    d  = depth[valid].astype(np.float32)
    ox = cols[valid].astype(np.float32)
    oy = rows[valid].astype(np.float32)

    x_world = (ox - cx) / fx * d
    y_world = (oy - cy) / fy * d   # double-negation → upward positive
    z_world = -d

    return np.stack([x_world, y_world, z_world], axis=1)


def _make_pcd(food_pts: np.ndarray):
    """Build an Open3D PointCloud with estimated normals (needed by alpha + poisson)."""
    if not OPEN3D_AVAILABLE:
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(food_pts.astype(np.float64))
    pcd.estimate_normals()
    return pcd


def _denoise_dbscan(points: np.ndarray) -> np.ndarray:
    """DBSCAN noise removal — paper params: ε=5 mm, minPts=5. Keep largest cluster."""
    if not SKLEARN_AVAILABLE or points.shape[0] < MIN_VALID_POINTS:
        return points

    try:
        labels = DBSCAN(eps=0.005, min_samples=5).fit_predict(points)
        unique, counts = np.unique(labels[labels >= 0], return_counts=True)
        if len(unique) == 0:
            logger.warning("DBSCAN found no clusters — using all points")
            return points
        # Paper: if >50% removed, fall back to raw cloud
        largest = unique[np.argmax(counts)]
        filtered = points[labels == largest]
        removal_frac = 1.0 - filtered.shape[0] / points.shape[0]
        if removal_frac > 0.5:
            logger.warning(
                "DBSCAN removed %.0f%% of points — using raw cloud (paper fallback)",
                removal_frac * 100,
            )
            return points
        logger.info(
            "DBSCAN: %d → %d points (largest cluster)", points.shape[0], filtered.shape[0]
        )
        return filtered
    except Exception:
        logger.warning("DBSCAN failed:\n%s", traceback.format_exc())
        return points


# ---------------------------------------------------------------------------
# Volume methods — paper's exact 5 (each returns ml or None)
# ---------------------------------------------------------------------------


def _method_convex(food_pts: np.ndarray) -> Optional[float]:
    """Method 1 — scipy 3-D convex hull volume (no fill-ratio correction)."""
    if not SCIPY_AVAILABLE:
        logger.warning("Method 1 (convex) skipped: scipy not available")
        return None
    try:
        if food_pts.shape[0] < 5:
            return None
        hull = ConvexHull(food_pts)
        vol_ml = hull.volume * 1e6   # m³ → ml
        logger.info("Method 1 (convex): %.2f ml", vol_ml)
        return float(vol_ml)
    except Exception:
        logger.warning("Method 1 (convex) failed:\n%s", traceback.format_exc())
        return None


def _method_delaunay(food_pts: np.ndarray) -> Optional[float]:
    """Method 2 — Delaunay tetrahedral decomposition volume."""
    if not SCIPY_AVAILABLE:
        logger.warning("Method 2 (delaunay) skipped: scipy not available")
        return None
    try:
        if food_pts.shape[0] < 5:
            return None
        tri = Delaunay(food_pts)
        tetra = food_pts[tri.simplices]   # (N_tetra, 4, 3)
        a = tetra[:, 1] - tetra[:, 0]
        b = tetra[:, 2] - tetra[:, 0]
        c = tetra[:, 3] - tetra[:, 0]
        vols = np.abs(np.einsum("ni,ni->n", a, np.cross(b, c))) / 6.0
        vol_ml = float(np.sum(vols)) * 1e6
        logger.info("Method 2 (delaunay): %.2f ml", vol_ml)
        return vol_ml
    except Exception:
        logger.warning("Method 2 (delaunay) failed:\n%s", traceback.format_exc())
        return None


def _method_alpha_shape(pcd) -> Optional[float]:
    """Method 3 — Open3D alpha shape (α=0.5) surface reconstruction."""
    if not OPEN3D_AVAILABLE or pcd is None:
        logger.warning("Method 3 (alpha_shape) skipped: open3d not available")
        return None
    try:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd, alpha=0.5
        )
        if not mesh.is_watertight():
            logger.warning("Method 3 (alpha_shape): mesh not watertight — skipping")
            return None
        vol_ml = float(mesh.get_volume()) * 1e6
        logger.info("Method 3 (alpha_shape): %.2f ml", vol_ml)
        return vol_ml
    except Exception:
        logger.warning("Method 3 (alpha_shape) failed:\n%s", traceback.format_exc())
        return None


def _method_spline(food_pts: np.ndarray, n_slices: int = 30) -> Optional[float]:
    """Method 4 — slice integration along Y axis (spline/trapz).

    Slices the point cloud into n_slices horizontal slabs, computes the 2-D
    convex hull area of each slab's XZ cross-section, then integrates via
    np.trapz to get volume.
    """
    if not SCIPY_AVAILABLE:
        logger.warning("Method 4 (spline) skipped: scipy not available")
        return None
    try:
        y = food_pts[:, 1]
        y_min, y_max = float(y.min()), float(y.max())
        if y_max - y_min < 0.001:   # < 1 mm height — degenerate
            return None

        edges = np.linspace(y_min, y_max, n_slices + 1)
        areas: list[float] = []
        ys: list[float] = []

        for i in range(n_slices):
            mask = (y >= edges[i]) & (y < edges[i + 1])
            if mask.sum() < 3:
                continue
            xz = food_pts[mask][:, [0, 2]]
            try:
                areas.append(float(ConvexHull(xz).volume))   # 2-D hull → area m²
                ys.append((edges[i] + edges[i + 1]) / 2)
            except Exception:
                pass   # skip degenerate slices

        if len(areas) < 2:
            return None

        vol_ml = float(np.trapz(areas, ys)) * 1e6
        logger.info("Method 4 (spline): %.2f ml (%d slices)", vol_ml, len(areas))
        return vol_ml
    except Exception:
        logger.warning("Method 4 (spline) failed:\n%s", traceback.format_exc())
        return None


def _method_poisson(pcd) -> Optional[float]:
    """Method 5 — Poisson surface reconstruction (depth=9, paper preferred method)."""
    if not OPEN3D_AVAILABLE or pcd is None:
        logger.warning("Method 5 (poisson) skipped: open3d not available")
        return None
    try:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9
        )
        densities_np = np.asarray(densities)
        threshold = np.quantile(densities_np, 0.1)
        # remove_vertices_by_mask: True = remove (i.e. low-density vertices)
        remove_mask = (densities_np <= threshold).tolist()
        mesh.remove_vertices_by_mask(remove_mask)

        if not mesh.is_watertight():
            logger.warning("Method 5 (poisson): mesh not watertight after cleanup")
            return None

        vol_ml = float(mesh.get_volume()) * 1e6
        logger.info("Method 5 (poisson): %.2f ml", vol_ml)
        return vol_ml
    except Exception:
        logger.warning("Method 5 (poisson) failed:\n%s", traceback.format_exc())
        return None


# ---------------------------------------------------------------------------
# PLY export
# ---------------------------------------------------------------------------


def _export_ply_base64(food_pts: np.ndarray) -> Optional[str]:
    """Export food_pts as a compressed binary PLY and return base64."""
    if not OPEN3D_AVAILABLE:
        try:
            return _export_ply_ascii_base64(food_pts)
        except Exception:
            logger.warning("PLY ASCII fallback failed:\n%s", traceback.format_exc())
            return None

    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(food_pts.astype(np.float64))

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            o3d.io.write_point_cloud(tmp_path, pcd, write_ascii=False, compressed=True)
            with open(tmp_path, "rb") as f:
                ply_bytes = f.read()
        finally:
            os.unlink(tmp_path)

        return base64.b64encode(ply_bytes).decode("utf-8")
    except Exception:
        logger.warning("PLY export (open3d) failed:\n%s", traceback.format_exc())
        try:
            return _export_ply_ascii_base64(food_pts)
        except Exception:
            return None


def _export_ply_ascii_base64(food_pts: np.ndarray) -> str:
    """Minimal ASCII PLY fallback."""
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {food_pts.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    for pt in food_pts:
        lines.append(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}")
    ply_bytes = "\n".join(lines).encode("utf-8")
    return base64.b64encode(ply_bytes).decode("utf-8")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
