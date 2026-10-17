import os
import sys
from flask import Flask, render_template, Response, jsonify, request, send_from_directory

# Ensure web_app directory is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from detector import ThreatDetector

app = Flask(
    __name__,
    template_folder=os.path.join(current_dir, "templates"),
    static_folder=os.path.join(current_dir, "static")
)

# Global Detector Instance
detector = ThreatDetector(camera_index=0, conf_threshold=0.35)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    """Video streaming route for CCTV feed."""
    return Response(
        detector.process_next_frame(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/api/status")
def get_status():
    """Returns real-time telemetry, active threats, and FPS."""
    with detector.lock:
        return jsonify({
            "fps": detector.current_fps,
            "inference_fps": detector.inference_fps,
            "active_threats": detector.active_threats,
            "conf_threshold": detector.conf_threshold,
            "camera_index": detector.camera_index,
            "selected_engine": detector.selected_engine,
            "model_status": detector.model_status,
            "latest_snapshot": detector.latest_snapshot
        })

@app.route("/api/config", methods=["POST"])
def update_config():
    """Update confidence threshold, camera input, or model engine."""
    data = request.get_json() or {}
    conf = data.get("confidence_threshold")
    cam = data.get("camera_index")
    engine = data.get("engine")
    detector.set_config(camera_index=cam, conf_threshold=conf, engine=engine)
    return jsonify({
        "status": "success",
        "conf": detector.conf_threshold,
        "camera": detector.camera_index,
        "engine": detector.selected_engine
    })

@app.route("/api/snapshot", methods=["POST"])
def take_snapshot():
    """Triggers immediate evidence capture."""
    snap = detector.capture_snapshot(custom_label="OPERATOR_CAPTURE")
    if snap:
        return jsonify({"status": "success", "snapshot": snap})
    return jsonify({"status": "error", "message": "Frame not available"}), 400

@app.route("/static/snapshots/<path:filename>")
def serve_snapshot(filename):
    return send_from_directory(detector.snapshot_dir, filename)

if __name__ == "__main__":
    PORT = 8000
    print(f"\n=======================================================")
    print(f" [!] THREAT DETECTION SYSTEM - CONSOLE ONLINE")
    print(f" [!] OPEN IN BROWSER: http://localhost:{PORT}")
    print(f"=======================================================\n")
    try:
        app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
    finally:
        detector.close()
