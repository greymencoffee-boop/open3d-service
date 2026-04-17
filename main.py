"""
open3d-service/main.py
FastAPI service for processing LiDAR depth maps into volumetric estimates.
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
    from scipy.spatial import ConvexHull

    SCIPY_AVAILABLE = True
    logger.info("scipy loaded successfully")
except Exception as _e:
    ConvexHull = None  # type: ignore[assignment]
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
TABLE_MARGIN_M = 0.005  # 5 mm above table plane
VOXEL_SIZE = 0.004      # 4 mm
FOOD_FILL_RATIO = 0.55  # food is ~55 % of its convex hull

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ProcessRequest(BaseModel):
    depthMapBase64: str
    intrinsics: Optional[list[float]] = Field(default=None)
    scanId: str = ""


class ProcessResponse(BaseModel):
    volumeMl: Optional[float] = None
    methodBreakdown: dict[str, Any] = {}
    confidence: Optional[float] = None
    plyBase64: Optional[str] = None
    error: Optional[str] = None
    timedOut: bool = False


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="LiDAR Depth-Map Processor", version="1.0.0")

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

    # 3. DBSCAN noise removal
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

    low_confidence_base = food_pts.shape[0] < MIN_FOOD_POINTS
    if low_confidence_base:
        logger.warning(
            "Only %d food points found (< %d); proceeding with low confidence",
            food_pts.shape[0],
            MIN_FOOD_POINTS,
        )
        # Use all points as a fallback so we still get a result
        food_pts = points

    # 5. Volume methods
    breakdown: dict[str, Any] = {}

    m1 = _method_depth_projection(food_pts, table_y)
    breakdown["depth_projection"] = round(m1, 2) if m1 is not None else None

    m2 = _method_voxel_grid(food_pts)
    breakdown["voxel_grid"] = round(m2, 2) if m2 is not None else None

    m3 = _method_convex_hull_3d(food_pts)
    breakdown["convex_hull_3d"] = round(m3, 2) if m3 is not None else None

    m4 = _method_alpha_shape(food_pts)
    breakdown["alpha_shape"] = round(m4, 2) if m4 is not None else None

    m5 = _method_poisson(food_pts)
    breakdown["poisson"] = round(m5, 2) if m5 is not None else None

    # 6. Ensemble
    candidates = [
        v for v in [m1, m2, m3, m4, m5] if v is not None and 40.0 < v < 1600.0
    ]

    if not candidates:
        return ProcessResponse(
            error="All volume methods failed or produced out-of-range estimates",
            methodBreakdown=breakdown,
            timedOut=False,
        )

    if len(candidates) >= 3:
        s = sorted(candidates)
        trimmed = s[1:-1]
        final_vol = sum(trimmed) / len(trimmed)
    else:
        final_vol = sum(candidates) / len(candidates)

    # 7. Confidence
    confidence = _compute_confidence(candidates, final_vol, low_confidence_base)

    # 8. PLY export
    ply_b64 = _export_ply_base64(food_pts)

    logger.info(
        "Scan %s complete — vol=%.1f ml conf=%.2f methods=%d",
        req.scanId or "<no-id>",
        final_vol,
        confidence,
        len(candidates),
    )

    return ProcessResponse(
        volumeMl=round(final_vol, 2),
        methodBreakdown=breakdown,
        confidence=round(confidence, 3),
        plyBase64=ply_b64,
    )


# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------


def _backproject(
    depth: np.ndarray, fx: float, fy: float, cx: float, cy: float
) -> np.ndarray:
    """Pinhole back-projection. Returns Nx3 float32 world points (y-up).

    Spec math (step-by-step):
      zCam = -d
      xCam = (ox - cx) / fx * d
      yCam = -(oy - cy) / fy * d          <- image row 0 is top; negate so up = -row
      y_world = -yCam                     <- negate again so world y is upward-positive
               = (oy - cy) / fy * d       <- the two negations cancel
    """
    rows, cols = np.indices(depth.shape)  # each (H, W)
    valid = ~np.isnan(depth)

    d = depth[valid].astype(np.float32)
    ox = cols[valid].astype(np.float32)
    oy = rows[valid].astype(np.float32)

    x_world = (ox - cx) / fx * d
    y_world = (oy - cy) / fy * d  # double-negation → upward positive
    z_world = -d

    pts = np.stack([x_world, y_world, z_world], axis=1)
    return pts


def _denoise_dbscan(points: np.ndarray) -> np.ndarray:
    """DBSCAN noise removal — keep largest cluster. Falls back to all points."""
    if not SKLEARN_AVAILABLE or points.shape[0] < MIN_VALID_POINTS:
        return points

    try:
        labels = DBSCAN(eps=0.012, min_samples=8).fit_predict(points)
        unique, counts = np.unique(labels[labels >= 0], return_counts=True)
        if len(unique) == 0:
            logger.warning("DBSCAN found no clusters — using all points")
            return points
        largest = unique[np.argmax(counts)]
        filtered = points[labels == largest]
        logger.info(
            "DBSCAN: %d → %d points (largest cluster)", points.shape[0], filtered.shape[0]
        )
        return filtered
    except Exception:
        logger.warning("DBSCAN failed:\n%s", traceback.format_exc())
        return points


# ---------------------------------------------------------------------------
# Volume methods (each returns ml or None)
# ---------------------------------------------------------------------------


def _method_depth_projection(food_pts: np.ndarray, table_y: float) -> Optional[float]:
    """Method 1 — depth projection via 2-D convex hull."""
    if not SCIPY_AVAILABLE:
        logger.warning("Method 1 skipped: scipy not available")
        return None
    try:
        xz = food_pts[:, [0, 2]]
        if xz.shape[0] < 4:
            return None
        hull = ConvexHull(xz)
        area_m2 = hull.volume  # ConvexHull.volume = area in 2-D
        height_m = float(np.mean(food_pts[:, 1]) - table_y)
        if height_m <= 0:
            height_m = float(np.abs(np.max(food_pts[:, 1]) - np.min(food_pts[:, 1])))
        vol_ml = area_m2 * height_m * 1e6
        logger.info("Method 1 (depth_projection): %.2f ml", vol_ml)
        return float(vol_ml)
    except Exception:
        logger.warning("Method 1 (depth_projection) failed:\n%s", traceback.format_exc())
        return None


def _method_voxel_grid(food_pts: np.ndarray) -> Optional[float]:
    """Method 2 — voxel grid occupancy (open3d)."""
    if not OPEN3D_AVAILABLE:
        logger.warning("Method 2 skipped: open3d not available")
        return None
    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(food_pts.astype(np.float64))
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
            pcd, voxel_size=VOXEL_SIZE
        )
        voxels = voxel_grid.get_voxels()
        n_voxels = len(voxels)
        vol_ml = n_voxels * (VOXEL_SIZE**3) * 1e6
        logger.info("Method 2 (voxel_grid): %d voxels → %.2f ml", n_voxels, vol_ml)
        return float(vol_ml)
    except Exception:
        logger.warning("Method 2 (voxel_grid) failed:\n%s", traceback.format_exc())
        return None


def _method_convex_hull_3d(food_pts: np.ndarray) -> Optional[float]:
    """Method 3 — scipy 3-D convex hull × 0.55."""
    if not SCIPY_AVAILABLE:
        logger.warning("Method 3 skipped: scipy not available")
        return None
    try:
        if food_pts.shape[0] < 5:
            return None
        hull = ConvexHull(food_pts)
        vol_ml = hull.volume * FOOD_FILL_RATIO * 1e6
        logger.info("Method 3 (convex_hull_3d): %.2f ml", vol_ml)
        return float(vol_ml)
    except Exception:
        logger.warning("Method 3 (convex_hull_3d) failed:\n%s", traceback.format_exc())
        return None


def _method_alpha_shape(food_pts: np.ndarray) -> Optional[float]:
    """Method 4 — BPA mesh (open3d) or trimesh convex hull × 0.55 fallback."""
    # Attempt open3d BPA first
    if OPEN3D_AVAILABLE:
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(food_pts.astype(np.float64))
            pcd.estimate_normals()
            radii = [0.01, 0.02, 0.04]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii)
            )
            if mesh.is_watertight():
                vol_ml = float(mesh.get_volume()) * 1e6
                logger.info("Method 4 (BPA mesh): %.2f ml", vol_ml)
                return vol_ml
            else:
                logger.warning("Method 4: BPA mesh not watertight — falling back")
        except Exception:
            logger.warning("Method 4 (BPA) failed:\n%s", traceback.format_exc())

    # Fallback: trimesh convex hull × 0.55
    if TRIMESH_AVAILABLE:
        try:
            pc = trimesh.PointCloud(food_pts.astype(np.float64))
            hull = pc.convex_hull
            vol_ml = float(hull.volume) * FOOD_FILL_RATIO * 1e6
            logger.info("Method 4 (trimesh convex hull proxy): %.2f ml", vol_ml)
            return vol_ml
        except Exception:
            logger.warning(
                "Method 4 (trimesh fallback) failed:\n%s", traceback.format_exc()
            )

    logger.warning("Method 4 skipped: no suitable library available")
    return None


def _method_poisson(food_pts: np.ndarray) -> Optional[float]:
    """Method 5 — Poisson surface reconstruction (open3d)."""
    if not OPEN3D_AVAILABLE:
        logger.warning("Method 5 skipped: open3d not available")
        return None
    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(food_pts.astype(np.float64))
        pcd.estimate_normals()

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=6
        )
        densities_np = np.asarray(densities)
        threshold = np.quantile(densities_np, 0.1)
        # remove_vertices_by_mask removes vertices where mask[i] is True,
        # so pass True for low-density (unwanted) vertices.
        remove_mask = (densities_np <= threshold).tolist()
        mesh.remove_vertices_by_mask(remove_mask)

        if not mesh.is_watertight():
            logger.warning("Method 5: Poisson mesh not watertight after cleanup")
            return None

        vol_ml = float(mesh.get_volume()) * 1e6
        logger.info("Method 5 (poisson): %.2f ml", vol_ml)
        return vol_ml
    except Exception:
        logger.warning("Method 5 (poisson) failed:\n%s", traceback.format_exc())
        return None


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


def _compute_confidence(
    candidates: list[float], final_vol: float, low_pt_flag: bool
) -> float:
    base_conf = 0.70

    if len(candidates) >= 4:
        base_conf += 0.10

    spread = max(candidates) - min(candidates)
    rel_spread = spread / final_vol if final_vol > 0 else 1.0

    if rel_spread < 0.15:
        base_conf += 0.10
    elif rel_spread > 0.40:
        base_conf -= 0.10

    if low_pt_flag:
        base_conf -= 0.15

    return float(max(0.30, min(0.95, base_conf)))


# ---------------------------------------------------------------------------
# PLY export
# ---------------------------------------------------------------------------


def _export_ply_base64(food_pts: np.ndarray) -> Optional[str]:
    """Export food_pts as a compressed binary little-endian PLY and return base64."""
    if not OPEN3D_AVAILABLE:
        # Fallback: write a minimal ASCII PLY manually
        try:
            return _export_ply_ascii_base64(food_pts)
        except Exception:
            logger.warning("PLY ASCII fallback failed:\n%s", traceback.format_exc())
            return None

    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(food_pts.astype(np.float64))

        # Write to a temp file, then read back bytes
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            o3d.io.write_point_cloud(
                tmp_path, pcd, write_ascii=False, compressed=True
            )
            with open(tmp_path, "rb") as f:
                ply_bytes = f.read()
        finally:
            os.unlink(tmp_path)

        return base64.b64encode(ply_bytes).decode("utf-8")
    except Exception:
        logger.warning("PLY export (open3d) failed:\n%s", traceback.format_exc())
        # Fallback
        try:
            return _export_ply_ascii_base64(food_pts)
        except Exception:
            logger.warning("PLY ASCII fallback also failed")
            return None


def _export_ply_ascii_base64(food_pts: np.ndarray) -> str:
    """Minimal ASCII PLY as a fallback when open3d is unavailable."""
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
