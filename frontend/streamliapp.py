import os
import json
import requests
from typing import Tuple, Dict, List, Optional

import cv2
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image

from ultralytics import YOLO

# ------------------------------------------
# BACKEND URL (TPS Engine)
# ------------------------------------------
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8502/tryon")


# ------------------------------------------
# Helper Functions
# ------------------------------------------

def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def parse_canvas_points(json_data):
    if json_data is None:
        return []
    objs = json_data.get("objects", [])
    pts = []
    for o in objs:
        if o.get("top") is not None and o.get("left") is not None:
            pts.append((int(o["left"]), int(o["top"])))
    return pts


def extract_yolo_keypoints(results) -> Optional[Dict[str, Tuple[int, int]]]:
    """
    Extracts specific COCO keypoints (Shoulders, Hips, Knees)
    to match the backend's expected body anchors.
    """
    try:
        # Get keypoints from the first detected person
        # Shape is usually (1, 17, 2) or (1, 17, 3)
        if not hasattr(results[0], 'keypoints') or results[0].keypoints is None:
            return None

        kpts = results[0].keypoints.xy[0].cpu().numpy()

        # Check if we have enough points (COCO has 17 points)
        if len(kpts) < 17:
            return None

        # COCO Keypoint Indices:
        # 5: Left Shoulder, 6: Right Shoulder
        # 11: Left Hip, 12: Right Hip
        # 13: Left Knee, 14: Right Knee

        anchors = {
            "left_shoulder": (int(kpts[5][0]), int(kpts[5][1])),
            "right_shoulder": (int(kpts[6][0]), int(kpts[6][1])),
            "left_waist": (int(kpts[11][0]), int(kpts[11][1])),  # Mapping Hip to Waist
            "right_waist": (int(kpts[12][0]), int(kpts[12][1])),
            "left_leg": (int(kpts[13][0]), int(kpts[13][1])),  # Mapping Knee to Leg
            "right_leg": (int(kpts[14][0]), int(kpts[14][1])),
        }

        # Filter out (0,0) points which indicate low confidence/not visible
        if any(v == (0, 0) for v in anchors.values()):
            return None

        return anchors
    except Exception as e:
        print(f"Auto-detection failed: {e}")
        return None


def call_engine(person_path, garment_path, body_anchors, garment_anchors):
    try:
        files = {
            "person": open(person_path, "rb"),
            "garment": open(garment_path, "rb"),
        }
        data = {
            "body_anchors_json": json.dumps(body_anchors),
            "garment_anchors_json": json.dumps(garment_anchors),
        }

        resp = requests.post(BACKEND_URL, files=files, data=data, timeout=300)

        if resp.status_code != 200:
            raise RuntimeError(f"Engine error: {resp.status_code} {resp.text}")

        out_path = "hq_tryon.jpg"
        with open(out_path, "wb") as f:
            f.write(resp.content)
        return out_path
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"Cannot connect to backend at {BACKEND_URL}. Make sure the backend is running.")
    except Exception as e:
        raise RuntimeError(f"Error calling engine: {str(e)}")


# ------------------------------------------
# STREAMLIT UI
# ------------------------------------------

st.set_page_config(page_title="VTON TPS Demo", layout="wide")
st.title("👗 Virtual Try-On (TPS Core Engine + Auto Pose)")

# Initialize session state
if "pose_model" not in st.session_state:
    st.session_state["pose_model"] = None
if "seg_model" not in st.session_state:
    st.session_state["seg_model"] = None
if "use_auto_anchors" not in st.session_state:
    st.session_state["use_auto_anchors"] = False

# Sidebar
with st.sidebar:
    st.header("Configuration")

    st.info(f"Backend URL: {BACKEND_URL}")

    st.subheader("Input Images")
    person_path = st.text_input("Person Image Path", "data/person.jpg")
    garment_path = st.text_input("Garment Image Path", "data/garment.jpg")

    st.subheader("Model Paths")
    pose_model_path = st.text_input("Pose Model", "yolo11n-pose.pt")
    seg_model_path = st.text_input("Seg Model", "yolo11n-seg.pt")

    if st.button("Load Models"):
        try:
            with st.spinner("Loading models..."):
                st.session_state["pose_model"] = YOLO(pose_model_path)
                st.session_state["seg_model"] = YOLO(seg_model_path)
            st.success("✅ Models loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading models: {e}")

# -------------------------
# Step 1: Pose Detection (UPDATED WITH AUTOMATION)
# -------------------------

st.subheader("Step 1: Detect Pose")

if not os.path.exists(person_path):
    st.warning(f"⚠️ Person image not found at: {person_path}")
else:
    if st.button("Run Pose Detection"):
        if st.session_state["pose_model"] is None:
            st.error("❌ Please load models first (use sidebar)")
        else:
            try:
                model = st.session_state["pose_model"]
                person_img = cv2.imread(person_path)
                if person_img is None:
                    st.error("❌ Invalid person image")
                else:
                    with st.spinner("Detecting pose..."):
                        results = model(person_img)
                        overlay = results[0].plot()

                        # --- AUTO-DETECTION ---
                        auto_anchors = extract_yolo_keypoints(results)

                        if auto_anchors:
                            st.session_state["body_anchors"] = auto_anchors
                            st.session_state["use_auto_anchors"] = True
                            st.success("✅ Pose detected & Anchors AUTO-MAPPED! (No clicking needed)")
                        else:
                            st.session_state["use_auto_anchors"] = False
                            st.warning("⚠️ Could not auto-detect all joints. Please switch to manual clicking.")

                    st.session_state["person_img"] = person_img
                    st.session_state["person_overlay"] = overlay
            except Exception as e:
                st.error(f"❌ Error during pose detection: {e}")

if "person_overlay" in st.session_state:
    # Logic: If auto anchors exist, show them. Else, show manual canvas.
    if st.session_state.get("use_auto_anchors"):
        st.image(st.session_state["person_overlay"], caption="Auto-Detected Pose", use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            with st.expander("View Auto Coordinates"):
                st.json(st.session_state["body_anchors"])
        with col_b:
            if st.button("Reset to Manual Mode"):
                st.session_state["use_auto_anchors"] = False
                st.rerun()
    else:
        # MANUAL FALLBACK
        pose_img = st.session_state["person_overlay"]
        h, w = pose_img.shape[:2]
        scale = min(600 / w, 600 / h)
        cw, ch = int(w * scale), int(h * scale)
        resized = cv2.resize(pose_img, (cw, ch))

        st.markdown("#### Click body anchors (in order):")
        st.markdown("**Order:** left_shoulder → right_shoulder → left_waist → right_waist → left_leg → right_leg")

        canvas_person = st_canvas(
            stroke_width=5,
            stroke_color="#00FF00",
            background_image=Image.fromarray(bgr_to_rgb(resized)),
            update_streamlit=True,
            height=ch,
            width=cw,
            drawing_mode="point",
            key="person_canvas",
        )

        pts = parse_canvas_points(canvas_person.json_data)
        body_anchor_names = ["left_shoulder", "right_shoulder", "left_waist", "right_waist", "left_leg", "right_leg"]
        true_pts = [(int(x / scale), int(y / scale)) for x, y in pts]

        if len(true_pts) >= len(body_anchor_names):
            st.session_state["body_anchors"] = {n: true_pts[i] for i, n in enumerate(body_anchor_names)}
            st.success(f"✅ Body anchors recorded: {len(true_pts)} points")
        else:
            st.info(f"📍 Mark {len(body_anchor_names)} points (currently: {len(true_pts)})")

# -------------------------
# Step 2: Garment Segmentation (MANUAL)
# -------------------------

st.subheader("Step 2: Garment Segmentation")

if not os.path.exists(garment_path):
    st.warning(f"⚠️ Garment image not found at: {garment_path}")
else:
    if st.button("Run Segmentation"):
        if st.session_state["seg_model"] is None:
            st.error("❌ Please load models first (use sidebar)")
        else:
            try:
                seg_model = st.session_state["seg_model"]
                garment_img = cv2.imread(garment_path)
                if garment_img is None:
                    st.error("❌ Invalid garment image")
                else:
                    with st.spinner("Segmenting garment..."):
                        seg = seg_model(garment_img)
                        overlay = seg[0].plot()
                    st.session_state["garment_img"] = garment_img
                    st.session_state["garment_overlay"] = overlay
                    st.success("✅ Segmentation done! Mark garment anchors below.")
            except Exception as e:
                st.error(f"❌ Error during segmentation: {e}")

if "garment_overlay" in st.session_state:
    ov = st.session_state["garment_overlay"]
    h, w = ov.shape[:2]
    scale = min(600 / w, 600 / h)
    cw, ch = int(w * scale), int(h * scale)
    resized = cv2.resize(ov, (cw, ch))

    st.markdown("#### Click garment anchors (in order):")
    st.markdown("**Order:** top_left → top_right → waist_left → waist_right → bottom_left → bottom_right")

    canvas_garment = st_canvas(
        stroke_width=5,
        stroke_color="#FF0000",
        background_image=Image.fromarray(bgr_to_rgb(resized)),
        update_streamlit=True,
        height=ch,
        width=cw,
        drawing_mode="point",
        key="garment_canvas",
    )

    pts = parse_canvas_points(canvas_garment.json_data)
    names = ["top_left", "top_right", "waist_left", "waist_right", "bottom_left", "bottom_right"]
    true_pts = [(int(x / scale), int(y / scale)) for x, y in pts]

    if len(true_pts) >= len(names):
        st.session_state["garment_anchors"] = {n: true_pts[i] for i, n in enumerate(names)}
        st.success(f"✅ Garment anchors recorded: {len(true_pts)} points")

        with st.expander("View Garment Anchors"):
            for name, coord in st.session_state["garment_anchors"].items():
                st.text(f"{name}: {coord}")
    else:
        st.info(f"📍 Mark {len(names)} points (currently: {len(true_pts)})")

# -------------------------
# STEP 3 — HQ Try-On (TPS ENGINE)
# -------------------------

st.subheader("⭐ Step 3: Run HQ Try-On (TPS Engine)")

if st.button("🚀 Run HQ Try-On (TPS Engine)", type="primary"):
    # Validate all required data
    missing = []
    if "body_anchors" not in st.session_state:
        missing.append("body anchors")
    if "garment_anchors" not in st.session_state:
        missing.append("garment anchors")

    if missing:
        st.error(f"❌ Missing: {', '.join(missing)}. Complete steps 1 and 2 first.")
    else:
        try:
            body = st.session_state["body_anchors"]
            garment = st.session_state["garment_anchors"]

            with st.spinner("🔄 Processing with TPS Engine... This may take a while."):
                out_path = call_engine(
                    person_path,
                    garment_path,
                    body_anchors=body,
                    garment_anchors=garment
                )

            st.success("✅ Try-on completed successfully!")

            col1, col2 = st.columns(2)
            with col1:
                st.image(person_path, caption="Original Person", use_container_width=True)
            with col2:
                st.image(out_path, caption="Try-On Result", use_container_width=True)

            with open(out_path, "rb") as f:
                st.download_button(
                    "⬇️ Download HQ Image",
                    data=f,
                    file_name="hq_tryon.jpg",
                    mime="image/jpeg",
                )

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.info("💡 Make sure the backend is running at: " + BACKEND_URL)
