"""
Cross-platform real-time emotion detection with OpenCV + Keras.

- Works on Linux, Windows, macOS (chooses best OpenCV camera backend per OS)
- Auto-detects model input shape (HxWxC) and matches training-time preprocessing:
    grayscale -> resize -> optional preprocess_fn(img, blur=False, threshold=False)
    -> float32, NO normalization (/255). If model expects C=3, gray is repeated to 3 channels.
- Uses Haar cascade for face detection with padding
- Draws label and FPS; press 'q' or ESC to quit

@Usage:
    python detect_realtime.py --model src/models/convnet.keras \
        --cascade src/haarcascade_frontalface_default.xml \
        --pref_width 640 --pref_height 480 --pad 0.18

@Parameters (CLI):
    --model:   Path to Keras model (.keras or .h5)
    --cascade: Path to Haar cascade (defaults to OpenCV's built-in if your path fails)
    --pad:     Face padding ratio (default 0.18)
    --scan:    Max camera indices to scan (default 5)
    --pref_width / --pref_height: Preferred capture size (default 640x480)
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
from keras.models import load_model

PREPROCESS_FN = None

EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


def parse_args():
    p = argparse.ArgumentParser(description="Cross-platform real-time emotion detection")
    p.add_argument("--model", type=str, default="src/models/convnet.keras")
    p.add_argument("--cascade", type=str, default="src/haarcascade_frontalface_default.xml")
    p.add_argument("--pad", type=float, default=0.18)
    p.add_argument("--scale_factor", type=float, default=1.1)
    p.add_argument("--min_neighbors", type=int, default=5)
    p.add_argument("--min_face_px", type=int, default=60)
    p.add_argument("--pref_width", type=int, default=640)
    p.add_argument("--pref_height", type=int, default=480)
    p.add_argument("--scan", type=int, default=5, help="Number of camera indices to try (0..scan-1)")
    return p.parse_args()


def resolve_cascade(path_hint: str) -> cv2.CascadeClassifier:
    """
    Try user path first; if it fails, fall back to OpenCV's built-in haarcascades dir.
    """
    paths_to_try = []
    if path_hint and os.path.isfile(path_hint):
        paths_to_try.append(path_hint)

    cv_default_dir = getattr(cv2, "data", None)
    if cv_default_dir and hasattr(cv2.data, "haarcascades"):
        paths_to_try.append(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))

    paths_to_try.extend([
        "haarcascade_frontalface_default.xml",
        os.path.join("src", "haarcascade_frontalface_default.xml")
    ])

    tried = []
    for p in paths_to_try:
        if p and os.path.isfile(p):
            clf = cv2.CascadeClassifier(p)
            if not clf.empty():
                print(f"[INFO] Using Haar cascade: {p}")
                return clf
            tried.append(p)

    raise RuntimeError(f"Failed to load Haar cascade. Tried: {tried or ['<none>']}")


def load_keras_model(model_path: str):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    model = load_model(model_path)
    return model


def detect_input_shape(model):
    """
    Returns (H, W, C) from model.input_shape which is usually (None, H, W, C).
    Falls back to (64, 64, 1) if unknown.
    """
    try:
        _, H, W, C = model.input_shape
        return int(H), int(W), int(C)
    except Exception:
        return 64, 64, 1


def crop_with_pad(img, x, y, w, h, pad_ratio=0.18):
    H_img, W_img = img.shape[:2]
    pw, ph = int(w * pad_ratio), int(h * pad_ratio)
    x1, y1 = max(0, x - pw), max(0, y - ph)
    x2, y2 = min(W_img, x + w + pw), min(H_img, y + h + ph)
    return img[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)


def preprocess_face(face_bgr, target_w, target_h, out_channels):
    """
    Match loader preprocessing:
      - convert to GRAYSCALE
      - resize to (target_w, target_h)
      - optional PREPROCESS_FN(img, blur=False, threshold=False)
      - convert to float32
      - NO normalization (/255)
      - if model expects 3 channels, repeat gray to 3
    Returns: (1, H, W, C)
    """
    face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    face_gray = cv2.resize(face_gray, (target_w, target_h), interpolation=cv2.INTER_AREA)

    if PREPROCESS_FN is not None:
        try:
            face_gray = PREPROCESS_FN(face_gray, blur=False, threshold=False)
        except TypeError:
            face_gray = PREPROCESS_FN(face_gray)

    face_gray = face_gray.astype("float32")
    if out_channels == 1:
        face = face_gray[..., None]  # (H, W, 1)
    else:
        face = np.repeat(face_gray[..., None], out_channels, axis=-1)  # (H, W, C)

    return face[None, ...]  # batch dim


def softmax_safe(x):
    x = np.asarray(x, dtype=np.float32)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def backend_candidates():
    """
    Return a list of camera backend constants in a sensible order for the host OS.
    """
    sysplat = sys.platform
    if sysplat.startswith("win"):
        return [cv2.CAP_DSHOW, cv2.CAP_ANY]
    elif sysplat == "darwin":
        return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    else:
        return [cv2.CAP_V4L2, cv2.CAP_ANY]


def open_camera(scan_count=5, pref_w=640, pref_h=480):
    """
    Try indices 0..scan_count-1 across backend candidates. Return an opened capture.
    """
    backends = backend_candidates()
    for be in backends:
        for idx in range(scan_count):
            cap = cv2.VideoCapture(idx, be)
            if not cap or not cap.isOpened():
                if cap:
                    cap.release()
                continue
            # Set preferred size
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(pref_w))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(pref_h))
            time.sleep(0.05)
            ok, frame = cap.read()
            if ok and frame is not None:
                print(f"[INFO] Opened camera index {idx} with backend {be}")
                return cap
            cap.release()
    raise RuntimeError("Could not open any camera with available backends.")


def main():
    args = parse_args()

    # Haar cascade + model
    face_cascade = resolve_cascade(args.cascade)
    model = load_keras_model(args.model)
    H, W, C = detect_input_shape(model)
    print(f"[INFO] Model expects input: (H={H}, W={W}, C={C})")

    # Camera
    cap = open_camera(scan_count=args.scan, pref_w=args.pref_width, pref_h=args.pref_height)

    last_t = time.time()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Warning: failed to read frame from camera.")
                break

            now = time.time()
            dt = now - last_t
            last_t = now
            if dt > 0:
                fps = (0.9 * fps) + (0.1 * (1.0 / dt)) if fps > 0 else 1.0 / dt

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=args.scale_factor,
                minNeighbors=args.min_neighbors,
                minSize=(args.min_face_px, args.min_face_px),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            if len(faces) == 0:
                cv2.putText(
                    frame, 'No Face', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2, cv2.LINE_AA
                )

            for (x, y, w, h) in faces:
                roi_bgr, (px, py, pw, ph) = crop_with_pad(frame, x, y, w, h, args.pad)
                if roi_bgr.size == 0:
                    continue

                face_in = preprocess_face(roi_bgr, W, H, C)  # (1, H, W, C)

                preds = model.predict(face_in, verbose=0)
                if preds.ndim == 2 and not np.allclose(preds.sum(axis=1), 1.0, atol=1e-3):
                    probs = softmax_safe(preds)[0]
                else:
                    probs = preds[0]

                cls_idx = int(np.argmax(probs))
                prob = float(probs[cls_idx])

                # Draw box + label
                cv2.rectangle(frame, (px, py), (px + pw, py + ph), (0, 255, 0), 2)
                label = f"{EMOTION_LABELS[cls_idx]} {prob*100:.1f}%"
                cv2.putText(
                    frame, label, (px, max(0, py - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA
                )

            # Draw FPS
            cv2.putText(
                frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
            )

            cv2.imshow("Emotion Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):  # ESC or 'q'
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

