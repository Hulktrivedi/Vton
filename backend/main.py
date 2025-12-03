import os
import io
import json
from typing import Tuple, Optional, Dict

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import torch
from ultralytics import YOLO

# Kornia TPS
import kornia
import kornia.geometry.transform as KGT
import torch.nn.functional as F

# -------------------------
# CONFIG
# -------------------------

POSE_MODEL = os.getenv("POSE_MODEL", "yolo11n-pose.pt")
SEG_MODEL = os.getenv("SEG_MODEL", "yolo11n-seg.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[engine] Using device: {DEVICE}")

print("[engine] Loading models...")
try:
    pose_model = YOLO(POSE_MODEL)
    seg_model = YOLO(SEG_MODEL)
    print("[engine] ✅ Models loaded successfully.")
except Exception as e:
    print(f"[engine] ❌ Error loading models: {e}")
    # We don't raise here so the server can still start,
    # but actual endpoints might fail if models are missing.
    pose_model = None
    seg_model = None

# -------------------------
# ANCHOR MAPPING
# -------------------------
# Maps garment anchor names to body anchor names
ANCHOR_MAPPING = {
    "top_left": "left_shoulder",
    "top_right": "right_shoulder",
    "waist_left": "left_waist",
    "waist_right": "right_waist",
    "bottom_left": "left_leg",
    "bottom_right": "right_leg",
}


# -------------------------
# Utility Functions
# -------------------------

def read_imagefile_to_bgr(file: UploadFile) -> np.ndarray:
    """Read uploaded file and convert to BGR numpy array."""
    try:
        data = file.file.read()
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")
        return img
    except Exception as e:
        raise ValueError(f"Invalid image file: {str(e)}")


def bgr_to_jpeg_bytes(img_bgr: np.ndarray, quality: int = 90) -> bytes:
    """Encode BGR image to JPEG bytes."""
    ok, enc = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return enc.tobytes()


def extract_mask(seg_results) -> Optional[np.ndarray]:
    """Extract largest segmentation mask from YOLO results."""
    if not seg_results or len(seg_results) == 0:
        return None

    res = seg_results[0]
    if (not hasattr(res, "masks")) or (res.masks is None):
        return None

    masks = res.masks.data.cpu().numpy()
    if len(masks) == 0:
        return None

    # Choose largest mask by area
    best = None
    best_area = -1
    for m in masks:
        area = np.count_nonzero(m > 0)
        if area > best_area:
            best_area = area
            best = m

    if best is None:
        return None

    return (best * 255).astype("uint8")


# -------------------------
# TPS Warp
# -------------------------

def warp_tps(
        person_bgr: np.ndarray,
        garment_bgr: np.ndarray,
        garment_mask: Optional[np.ndarray],
        garment_anchors: Dict[str, Tuple[int, int]],
        body_anchors: Dict[str, Tuple[int, int]],
        anchor_mapping: Dict[str, str] = ANCHOR_MAPPING
) -> np.ndarray:
    """
    Apply TPS (Thin Plate Spline) warping to fit garment onto person.
    """

    device = torch.device(DEVICE)

    # Convert to RGB and normalize to [0, 1]
    g_rgb = cv2.cvtColor(garment_bgr, cv2.COLOR_BGR2RGB)
    g = torch.from_numpy(g_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    g = g.to(device)

    p_rgb = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2RGB)
    p = torch.from_numpy(p_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    p = p.to(device)

    h_g, w_g = garment_bgr.shape[:2]
    h_p, w_p = person_bgr.shape[:2]

    # Prepare source (garment) and destination (body) control points
    src = []  # Garment points
    dst = []  # Body points

    print(f"[DEBUG] Garment anchors: {garment_anchors}")
    print(f"[DEBUG] Body anchors: {body_anchors}")

    # Use the mapping to match garment points to body points
    for garment_key, body_key in anchor_mapping.items():
        if garment_key in garment_anchors and body_key in body_anchors:
            sx, sy = garment_anchors[garment_key]
            dx, dy = body_anchors[body_key]
            src.append([sx, sy])
            dst.append([dx, dy])
            print(f"[DEBUG] Matched: {garment_key}({sx},{sy}) -> {body_key}({dx},{dy})")
        else:
            # Optional warning logging
            pass

    if len(src) < 3:
        raise ValueError(
            f"Need at least 3 matching anchor points, got {len(src)}. "
            f"Garment keys: {list(garment_anchors.keys())}, "
            f"Body keys: {list(body_anchors.keys())}"
        )

    print(f"[DEBUG] Found {len(src)} matching anchor pairs")

    src = torch.tensor(src, dtype=torch.float32, device=device).unsqueeze(0)
    dst = torch.tensor(dst, dtype=torch.float32, device=device).unsqueeze(0)

    # Normalize control points to [-1, 1]
    def norm(points, h, w):
        x = (points[..., 0] / w) * 2 - 1
        y = (points[..., 1] / h) * 2 - 1
        return torch.stack([x, y], dim=-1)

    src_norm = norm(src, h_g, w_g)
    dst_norm = norm(dst, h_p, w_p)

    # Compute TPS transformation weights (DST -> SRC)
    try:
        kernel_weights, affine_weights = KGT.get_tps_transform(dst_norm, src_norm)
        print(f"[DEBUG] TPS weights computed - kernel: {kernel_weights.shape}, affine: {affine_weights.shape}")
    except Exception as e:
        print(f"[ERROR] Failed to compute TPS weights: {e}")
        raise

    # Create output mesh grid for the person image
    grid = kornia.create_meshgrid(h_p, w_p, normalized_coordinates=True, device=device)

    # ---------------------------------------------------------
    # FIX: Flatten grid for Kornia TPS (BxNx2)
    # ---------------------------------------------------------
    grid_flat = grid.reshape(1, -1, 2)
    print(f"[DEBUG] Created grid (flat): {grid_flat.shape}")

    # Apply TPS warping to the flattened grid
    try:
        # We use dst_norm here because we are mapping FROM the destination grid TO the source
        warp_grid_flat = KGT.warp_points_tps(grid_flat, dst_norm, kernel_weights, affine_weights)

        # Reshape back to image dimensions (BxHxWx2)
        warp_grid = warp_grid_flat.reshape(1, h_p, w_p, 2)
        print(f"[DEBUG] Warped grid reshaped: {warp_grid.shape}")
    except Exception as e:
        print(f"[ERROR] Failed to warp grid: {e}")
        raise
    # ---------------------------------------------------------

    # Warp garment image using the warped grid
    warped_g = F.grid_sample(g, warp_grid, mode="bilinear", padding_mode="zeros", align_corners=True)

    # Warp mask
    if garment_mask is None:
        print("[DEBUG] No mask provided, creating full mask")
        garment_mask = np.ones((h_g, w_g), dtype=np.uint8) * 255

    mask_t = torch.from_numpy(garment_mask).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
    warped_m = F.grid_sample(mask_t, warp_grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    warped_m = warped_m.clamp(0, 1)

    # Composite: warped garment over person using mask
    result = warped_g * warped_m + p * (1 - warped_m)

    # Convert back to numpy BGR
    result = (result.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    print("[DEBUG] TPS warping completed successfully")
    return result


# -------------------------
# API
# -------------------------

app = FastAPI(title="TPS VTON Engine", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "kornia_version": kornia.__version__,
        "torch_version": torch.__version__,
        "anchor_mapping": ANCHOR_MAPPING
    }


@app.post("/tryon")
async def tryon(
        person: UploadFile = File(..., description="Person image"),
        garment: UploadFile = File(..., description="Garment image"),
        body_anchors_json: str = Form(..., description="Body anchor points as JSON"),
        garment_anchors_json: str = Form(..., description="Garment anchor points as JSON"),
):
    try:
        # Read images
        try:
            person_bgr = read_imagefile_to_bgr(person)
            garment_bgr = read_imagefile_to_bgr(garment)
            print(f"[DEBUG] Images loaded - Person: {person_bgr.shape}, Garment: {garment_bgr.shape}")
        except ValueError as e:
            return JSONResponse(status_code=400, content={"error": str(e)})

        # Parse anchors
        try:
            body_anchors = json.loads(body_anchors_json)
            body_anchors = {k: tuple(v) for k, v in body_anchors.items()}
        except json.JSONDecodeError as e:
            return JSONResponse(status_code=400, content={"error": f"Invalid body_anchors_json format: {e}"})

        try:
            garment_anchors = json.loads(garment_anchors_json)
            garment_anchors = {k: tuple(v) for k, v in garment_anchors.items()}
        except json.JSONDecodeError as e:
            return JSONResponse(status_code=400, content={"error": f"Invalid garment_anchors_json format: {e}"})

        # Validate we have enough anchors
        if len(body_anchors) < 3:
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"Need at least 3 body anchor points, got {len(body_anchors)}",
                    "received_keys": list(body_anchors.keys())
                }
            )
        if len(garment_anchors) < 3:
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"Need at least 3 garment anchor points, got {len(garment_anchors)}",
                    "received_keys": list(garment_anchors.keys())
                }
            )

        # Segment garment
        mask = None
        if seg_model is not None:
            try:
                print("[DEBUG] Running segmentation...")
                seg = seg_model(garment_bgr)
                mask = extract_mask(seg)
                if mask is not None:
                    if mask.shape[:2] != garment_bgr.shape[:2]:
                        mask = cv2.resize(mask, (garment_bgr.shape[1], garment_bgr.shape[0]))
                    print(f"[DEBUG] Mask extracted: {mask.shape}")
                else:
                    print("[DEBUG] No mask extracted from segmentation")
            except Exception as e:
                print(f"[WARNING] Segmentation error: {e}")
        else:
            print("[WARNING] Segmentation model not loaded, skipping mask generation.")

        # Apply TPS warp
        try:
            print("[DEBUG] Starting TPS warping...")
            composed = warp_tps(
                person_bgr,
                garment_bgr,
                mask,
                garment_anchors,
                body_anchors,
                ANCHOR_MAPPING
            )
            print("[DEBUG] TPS warping completed")
        except ValueError as e:
            return JSONResponse(status_code=400, content={"error": str(e)})
        except Exception as e:
            print(f"[ERROR] TPS warping error: {e}")
            import traceback
            traceback.print_exc()
            return JSONResponse(status_code=500, content={"error": f"TPS warping failed: {str(e)}"})

        # Encode result
        jpeg_bytes = bgr_to_jpeg_bytes(composed)
        print("[DEBUG] Result encoded, sending response")
        return StreamingResponse(io.BytesIO(jpeg_bytes), media_type="image/jpeg")

    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"Internal server error: {str(e)}"})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8502"))
    print(f"[engine] Starting server on port {port}...")
    print(f"[engine] Kornia version: {kornia.__version__}")
    print(f"[engine] PyTorch version: {torch.__version__}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
