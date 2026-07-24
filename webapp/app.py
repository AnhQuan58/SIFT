"""
app.py
Minimal Flask web app for the live demo described in the proposal:
  - left panel:  raw webcam feed
  - right panel: SIFT output (keypoints drawn on the frame)
  - FPS counter, overlaid on the output panel
  - a dropdown to switch the Step 1 backend at runtime (currently only
    "lib" is implemented; cpu/gpu_v1/gpu_v2/gpu_v3 will appear automatically
    once their modules are added under src/sift_pipeline/steps/ and
    registered -- see steps/__init__.py)

Run with:
    python webapp/app.py
then open http://127.0.0.1:5000 in a browser.

Design note: a single background thread owns the camera and runs the
pipeline once per frame; the two /input_feed and /output_feed MJPEG
streams just re-serve the most recently produced JPEG bytes. This avoids
opening the webcam twice and keeps both panels in sync.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, render_template, request

# Make `import sift_pipeline` work without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sift_pipeline import SiftPipeline  # noqa: E402
from sift_pipeline.visualize import draw_hud, draw_keypoints  # noqa: E402

app = Flask(__name__)

CAMERA_INDEX = 0
JPEG_QUALITY = 80
FPS_SMOOTHING = 0.9  # exponential moving average factor


class SharedState:
    """Everything the camera thread produces and the HTTP handlers read.
    Guarded by `lock` since Flask serves requests on multiple threads."""

    def __init__(self):
        self.lock = threading.Lock()
        self.input_jpeg: bytes | None = None
        self.output_jpeg: bytes | None = None
        self.fps: float = 0.0
        self.n_keypoints: int = 0
        self.backend: str = "lib"
        self.pending_backend: str | None = None
        self.running = True


state = SharedState()
pipeline = SiftPipeline(step1_backend=state.backend)


def camera_loop():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(
            f"[sift-demo] ERROR: could not open camera index {CAMERA_INDEX}. "
            "Check that a webcam is connected and not in use by another app.",
            file=sys.stderr,
        )
        return

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    last_time = time.perf_counter()
    ema_fps = 0.0

    while state.running:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        # Apply a pending backend switch requested from the UI.
        with state.lock:
            if state.pending_backend is not None:
                try:
                    pipeline.set_step1_backend(state.pending_backend)
                    state.backend = state.pending_backend
                except KeyError as exc:
                    print(f"[sift-demo] {exc}", file=sys.stderr)
                state.pending_backend = None

        result = pipeline.process_frame(frame)
        output_frame = draw_keypoints(frame, result.keypoints)

        now = time.perf_counter()
        instant_fps = 1.0 / max(now - last_time, 1e-6)
        last_time = now
        ema_fps = (
            instant_fps
            if ema_fps == 0.0
            else FPS_SMOOTHING * ema_fps + (1 - FPS_SMOOTHING) * instant_fps
        )

        output_frame = draw_hud(
            output_frame,
            fps=ema_fps,
            backend_name=result.step1_backend,
            n_keypoints=len(result.keypoints),
            total_ms=result.total_ms,
        )

        ok_in, input_buf = cv2.imencode(".jpg", frame, encode_params)
        ok_out, output_buf = cv2.imencode(".jpg", output_frame, encode_params)

        if ok_in and ok_out:
            with state.lock:
                state.input_jpeg = input_buf.tobytes()
                state.output_jpeg = output_buf.tobytes()
                state.fps = ema_fps
                state.n_keypoints = len(result.keypoints)

    cap.release()


def _mjpeg_stream(get_frame):
    """Shared MJPEG generator; get_frame() must return the latest JPEG
    bytes (or None) from SharedState."""
    boundary = b"--frame"
    while True:
        with state.lock:
            frame = get_frame()
        if frame is not None:
            yield (
                boundary
                + b"\r\nContent-Type: image/jpeg\r\n\r\n"
                + frame
                + b"\r\n"
            )
        time.sleep(1 / 60)  # cap generator loop rate; actual FPS is camera-bound


@app.route("/")
def index():
    return render_template(
        "index.html", backends=SiftPipeline.available_backends(), active=state.backend
    )


@app.route("/input_feed")
def input_feed():
    return Response(
        _mjpeg_stream(lambda: state.input_jpeg),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/output_feed")
def output_feed():
    return Response(
        _mjpeg_stream(lambda: state.output_jpeg),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/status")
def status():
    with state.lock:
        return jsonify(
            fps=round(state.fps, 1),
            n_keypoints=state.n_keypoints,
            backend=state.backend,
        )


@app.route("/set_backend", methods=["POST"])
def set_backend():
    name = (request.json or {}).get("backend", "")
    if name not in SiftPipeline.available_backends():
        return jsonify(ok=False, error=f"unknown backend '{name}'"), 400
    with state.lock:
        state.pending_backend = name
    return jsonify(ok=True, backend=name)


if __name__ == "__main__":
    thread = threading.Thread(target=camera_loop, daemon=True)
    thread.start()
    try:
        app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
    finally:
        state.running = False
