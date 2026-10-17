# 🚨 Real-Time Threat Detection Console (Web Edition)

A high-performance, minimalist, monochrome real-time surveillance dashboard for automated threat detection (firearms, handguns, rifles, knives, explosives, grenades, bombs).

## ⚡ Overview
- **Zero-Shot Object Detection**: Powered by Ultralytics YOLO-World, detecting specified armament and explosive classes on the fly with **zero training required**.
- **Live Optical Surveillance**: Connects directly to webcams, external USB cameras, or IP RTSP video feeds at high frame rates.
- **Minimalist Monochrome UI**: Clean, distraction-free surveillance console with high-contrast HUD bounding boxes, telemetry overlays, and subtle warning alerts.
- **Evidence Archiving**: Captures time-stamped visual evidence snapshots whenever a threat is identified.
- **Audible & Visual Sirens**: Real-time status indicators and optional acoustic alerts.

## 🚀 Quickstart

1. **Install Dependencies** (if not already installed):
   ```bash
   pip install opencv-python ultralytics flask
   ```

2. **Launch the Surveillance Server**:
   ```bash
   python web_app/app.py
   ```

3. **Access the Console**:
   Open your browser at:
   ```
   http://localhost:8000
   ```
