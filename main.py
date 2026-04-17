"""
open3d-service/main.py
FastAPI service for processing LiDAR depth maps into volumetric estimates.
Pipeline aligned with LiDARCalorieCam (Fujita & Yanai, 2025).
"""

import base64
import logging
import os
import tempfile
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


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="LiDAR Depth-Map Processor", version="2.0.0")

# Read at startup — Railway injects this as a service variable.
_SERVICE_SECRET_KEY: str | None = os.environ.get("SERVICE_SECRET_KEY")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/process", response_model=ProcessResponse)
def process(req: ProcessRequest, request: Request) -> ProcessResponse:
    # Validate shared secret when configured (always set in production).
    if _SERVICE_SECRET_KEY:
        incoming = request.headers.get("x-service-key", "")
        if incoming != _SERVICE_SECRET_KEY:
            return JSONResponse(status_code=401, content={"error": "Unauthorized"})

    scan_id = req.scanId or "<no-id>"
    logger.info("Processing scan %s", scan_id)

    try:
        return _run_pipeline(req)
    except Exception:
        msg = traceback.format_exc()
        logger.error("Unhandled exception in scan %s:\n%s", scan_id, msg)
        return ProcessResponse(
            error=f"Internal error: {traceback.format_exc(limit=3)}",
            timedOut=False,
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
    valid = {k: v for k, v in breakdown.items() if v is not None and 40.0 < v < 1600.0}

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

    # Paper's selection rule
    if confidence >= 0.8 and valid.get("poisson") is not None:
        final_vol = float(valid["poisson"])
    elif valid.get("convex") is not None:
        final_vol = float(valid["convex"])
    else:
        final_vol = mu

    # Clamp confidence if we had very few food points
    if low_pt_flag:
        confidence = max(0.30, confidence - 0.15)

    # 7. Per-category weight regression
    weight_g, matched_cat = _estimate_weight(final_vol, req.category)

    # 8. PLY export
    ply_b64 = _export_ply_base64(food_pts)

    logger.info(
        "Scan %s → vol=%.1f ml conf=%.2f methods=%d/%d%s",
        req.scanId or "<no-id>",
        final_vol,
        confidence,
        len(valid),
        5,
        f" weight={weight_g}g ({matched_cat})" if weight_g else "",
    )

    return ProcessResponse(
        volumeMl=round(final_vol, 2),
        methodBreakdown=breakdown,
        confidence=round(confidence, 3),
        plyBase64=ply_b64,
        estimatedWeightG=weight_g,
        regressionCategory=matched_cat,
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
