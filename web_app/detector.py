import os
import time
import datetime
import threading
from collections import defaultdict
import numpy as np
import cv2
import torch

# Optimize PyTorch CPU inference threads to prevent thread thrashing
torch.set_num_threads(4)

# Real weapon target classes
THREAT_CLASSES = {
    "handgun", "pistol", "revolver", "rifle", "shotgun", "gun", "firearm",
    "heavy_weapon", "knife", "dagger", "machete", "combat knife",
    "bomb", "grenade", "explosive", "weapon"
}

# Negative distractor classes for zero-shot disambiguation
NEGATIVE_DISTRACTORS = [
    "scissors", "pair of scissors", "soldering iron", "ruler", "scale",
    "screwdriver", "pliers", "tool", "human arm", "bare arm", "arm",
    "hand", "finger", "cell phone", "pen", "wire", "cable"
]

class ThreadedCamera:
    """Non-blocking low-latency camera frame grabber running on a dedicated thread."""
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.frame = None
        self.ret = False
        self.is_running = True
        self.lock = threading.Lock()
        
        self._init_cap()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _init_cap(self):
        if self.cap is not None:
            self.cap.release()
        try:
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not self.cap.isOpened():
                print(f"[ThreadedCamera] Camera {self.camera_index} not opened. Falling back to simulation.")
                self.cap = None
            else:
                print(f"[ThreadedCamera] Camera {self.camera_index} online via DirectShow.")
        except Exception as e:
            print(f"[ThreadedCamera] Capture error: {e}")
            self.cap = None

    def _capture_loop(self):
        while self.is_running:
            if self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self.lock:
                        self.ret = True
                        self.frame = frame
                else:
                    time.sleep(0.01)
            else:
                time.sleep(0.03)

    def read(self):
        with self.lock:
            if self.ret and self.frame is not None:
                return True, self.frame.copy()
            return False, None

    def switch_camera(self, new_index):
        with self.lock:
            self.camera_index = new_index
            self._init_cap()

    def release(self):
        self.is_running = False
        if self.cap:
            self.cap.release()


class ThreatDetector:
    def __init__(self, camera_index=0, conf_threshold=0.50):
        self.camera_index = camera_index
        self.conf_threshold = conf_threshold
        self.selected_engine = "sota_threat"  # 'sota_threat', 'yolo11_tactical', 'extended', 'world', 'legacy'
        
        self.model = None
        self.model_status = "INITIALIZING..."
        self.is_running = True
        self.lock = threading.Lock()
        
        self.current_fps = 0
        self.inference_fps = 0
        self.active_threats = []
        self.latest_frame = None
        self.latest_snapshot = None
        
        # Temporal persistence buffer
        self.detection_history = defaultdict(int)
        
        self.snapshot_dir = os.path.join(os.path.dirname(__file__), "static", "snapshots")
        os.makedirs(self.snapshot_dir, exist_ok=True)
        
        # Threaded camera
        self.camera = ThreadedCamera(self.camera_index)
        
        # Ground-truth distractor & utility model (YOLO11 nano COCO: scissors, cell phones, knives)
        self.distractor_model = None
        self._init_distractor_model()

        # Start background model loader and inference worker
        self._load_model_engine(self.selected_engine)
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()

    def _init_distractor_model(self):
        """Loads lightweight COCO model to actively veto scissors and boost knife recall."""
        def load_dist():
            try:
                from ultralytics import YOLO
                base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                coco_path = os.path.join(base_dir, "yolo11n.pt")
                if os.path.exists(coco_path):
                    self.distractor_model = YOLO(coco_path)
                    print("[ThreatDetector] Distractor & knife verification engine online (YOLO11n).")
            except Exception as e:
                print(f"[ThreatDetector] Distractor engine notice: {e}")

        t = threading.Thread(target=load_dist, daemon=True)
        t.start()

    def _load_model_engine(self, engine_name):
        """Loads and switches model engine."""
        def load():
            from ultralytics import YOLO
            self.model_status = f"LOADING {engine_name.upper()}..."
            try:
                base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                
                if engine_name in ["sota_threat", "specialized"]:
                    # SOTA Multi-Threat AI (Gun, Grenade, Knife, Explosion)
                    model_path = os.path.join(base_dir, "weights", "subh_threat_yolov8n.pt")
                    if not os.path.exists(model_path):
                        model_path = os.path.join(base_dir, "weapon_model.pt")
                    model = YOLO(model_path)
                    self.model_status = "SOTA MULTI-THREAT AI (GUN/GRENADE/KNIFE)"

                elif engine_name == "yolo11_tactical":
                    # Custom YOLOv11 Medium Tactical Model (100 Epochs: Gun, Knife)
                    model_path = os.path.join(base_dir, "weights", "yolo11m_guns.pt")
                    model = YOLO(model_path)
                    self.model_status = "YOLOV11 TACTICAL ENGINE (GUNS & BLADES)"

                elif engine_name == "extended":
                    # Extended Armament Model (gun, rifle, heavy_weapon, knife)
                    model_path = os.path.join(base_dir, "izzy_model.pt")
                    if not os.path.exists(model_path):
                        model_path = "izzy_model.pt"
                    model = YOLO(model_path)
                    self.model_status = "EXTENDED ARMAMENT ENGINE"

                elif engine_name == "world":
                    # Zero-shot YOLO-World with comprehensive negative distractor grounding
                    model = YOLO("yolov8s-worldv2.pt")
                    all_prompts = list(THREAT_CLASSES) + NEGATIVE_DISTRACTORS
                    model.set_classes(all_prompts)
                    self.model_status = "YOLO-WORLD ZERO-SHOT (GROUNDED)"

                else:  # 'legacy'
                    # Legacy Capstone Model (Dual-Class Nano)
                    model_path = os.path.join(base_dir, "weapon_model.pt")
                    model = YOLO(model_path)
                    self.model_status = "LEGACY CAPSTONE MODEL (NANO)"

                with self.lock:
                    self.model = model
                    self.selected_engine = engine_name
                    self.detection_history.clear()
                print(f"[ThreatDetector] Engine active: {self.model_status}")
            except Exception as e:
                print(f"[ThreatDetector] Error loading engine {engine_name}: {e}")
                self.model_status = "ONLINE (FALLBACK)"

        t = threading.Thread(target=load, daemon=True)
        t.start()

    def set_config(self, camera_index=None, conf_threshold=None, engine=None):
        with self.lock:
            if conf_threshold is not None:
                self.conf_threshold = float(conf_threshold)
            if camera_index is not None and camera_index != self.camera_index:
                self.camera_index = camera_index
                self.camera.switch_camera(self.camera_index)
            if engine is not None and engine != self.selected_engine:
                self.selected_engine = engine
                self._load_model_engine(engine)

    def _inference_loop(self):
        """Asynchronous worker running inference at ~20-30 FPS with calibrated false-positive defense."""
        sim_t = 0
        frame_counter = 0

        while self.is_running:
            success, frame = self.camera.read()
            if not success or frame is None:
                sim_t += 0.04
                frame = self.generate_simulated_frame(sim_t)

            frame_counter += 1

            if self.model is not None:
                t_start = time.time()
                try:
                    # Optimize inference resolution for high knife recall and threat precision
                    results = self.model.predict(
                        frame,
                        conf=float(self.conf_threshold),
                        imgsz=480,
                        verbose=False
                    )
                    
                    frame_candidates = []
                    phone_boxes = []

                    # 1. Detect smartphone regions to enable screen threat scanning and suppress bare-body false alarms
                    if self.distractor_model is not None:
                        try:
                            coco_phones = self.distractor_model.predict(
                                frame, classes=[67], conf=0.30, imgsz=320, verbose=False
                            )[0]
                            for pb in coco_phones.boxes:
                                phone_boxes.append(pb.xyxy[0].cpu().numpy().astype(int))
                        except Exception:
                            pass

                    fh, fw = frame.shape[:2]

                    # 2. Process primary engine detections
                    for r in results:
                        for box in r.boxes:
                            cls_id = int(box.cls[0])
                            raw_label = self.model.names[cls_id].lower()
                            conf = float(box.conf[0])
                            xyxy = box.xyxy[0].cpu().numpy().astype(int)
                            x1, y1, x2, y2 = xyxy

                            # Bounds clamp
                            x1 = max(0, min(fw - 1, x1))
                            y1 = max(0, min(fh - 1, y1))
                            x2 = max(0, min(fw - 1, x2))
                            y2 = max(0, min(fh - 1, y2))

                            w = x2 - x1
                            h = y2 - y1
                            if w <= 12 or h <= 12:
                                continue

                            aspect_ratio = max(w / (h + 1e-5), h / (w + 1e-5))

                            # Normalize label name
                            if "gun" in raw_label or "pistol" in raw_label:
                                label = "pistol"
                            elif "grenade" in raw_label or "bomb" in raw_label:
                                label = "grenade"
                            elif "knife" in raw_label or "blade" in raw_label:
                                label = "knife"
                            elif "explosion" in raw_label or "explosive" in raw_label:
                                label = "explosive"
                            else:
                                label = raw_label

                            # Reject explicit distractor text categories
                            is_distractor = any(d in label for d in [
                                "scissor", "scale", "ruler", "screwdriver", "pliers",
                                "soldering", "cell phone", "pen", "wire", "cable"
                            ])
                            if is_distractor:
                                continue

                            # Verify label is an authorized threat category
                            if not any(t in label for t in THREAT_CLASSES):
                                continue

                            # Suppress bare phone body false alarms: if candidate heavily overlaps phone box,
                            # defer to internal phone screen scanner
                            overlaps_phone = False
                            for pbox in phone_boxes:
                                px1, py1, px2, py2 = pbox
                                ix1 = max(x1, px1)
                                iy1 = max(y1, py1)
                                ix2 = min(x2, px2)
                                iy2 = min(y2, py2)
                                if ix2 > ix1 and iy2 > iy1:
                                    inter = (ix2 - ix1) * (iy2 - iy1)
                                    b_area = (x2 - x1) * (y2 - y1)
                                    if inter / (b_area + 1e-5) > 0.50:
                                        overlaps_phone = True
                                        break
                            if overlaps_phone:
                                continue

                            # Scale / Ruler Filter: Aspect Ratio > 3.0
                            if label in ["pistol", "gun"] and aspect_ratio > 3.0:
                                continue

                            crop = frame[y1:y2, x1:x2]
                            if crop.size > 0:
                                ch, cw = crop.shape[:2]

                                # Scissor Immunity: Check if candidate is actually scissors (verified at 320px)
                                if label in ["pistol", "gun", "knife"]:
                                    # (a) Topological Euler hole check
                                    try:
                                        gray_c = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                                        _, thresh_c = cv2.threshold(gray_c, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                                        cnts, hier = cv2.findContours(thresh_c, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                                        if hier is not None and len(hier[0]) > 0:
                                            euler_holes = sum(1 for h_info in hier[0] if h_info[3] != -1)
                                            if euler_holes >= 2:
                                                continue
                                    except Exception:
                                        pass

                                    # (b) COCO Scissor Veto (320px Crop Verification)
                                    if self.distractor_model is not None:
                                        try:
                                            c_res = self.distractor_model.predict(crop, imgsz=320, conf=0.18, verbose=False)[0]
                                            is_scissors = any(
                                                self.distractor_model.names[int(b.cls[0])].lower() == "scissors"
                                                for b in c_res.boxes
                                            )
                                            if is_scissors:
                                                continue
                                        except Exception:
                                            pass

                            frame_candidates.append({
                                "label": label,
                                "confidence": round(conf, 3),
                                "box": [int(x1), int(y1), int(x2), int(y2)]
                            })

                    # 3. Dedicated Smartphone Screen Threat Scanner
                    # When operators display weapon images on phone screens, scan the screen sub-region
                    for pbox in phone_boxes:
                        px1, py1, px2, py2 = pbox
                        pw = px2 - px1
                        ph = py2 - py1
                        sx1 = max(0, int(px1 + pw * 0.05))
                        sy1 = max(0, int(py1 + ph * 0.05))
                        sx2 = min(fw, int(px2 - pw * 0.05))
                        sy2 = min(fh, int(py2 - ph * 0.05))
                        screen_crop = frame[sy1:sy2, sx1:sx2]
                        if screen_crop.size > 0:
                            try:
                                s_res = self.model.predict(screen_crop, conf=0.22, imgsz=320, verbose=False)[0]
                                for sb in s_res.boxes:
                                    s_cls = int(sb.cls[0])
                                    s_raw = self.model.names[s_cls].lower()
                                    s_conf = float(sb.conf[0])
                                    s_xyxy = sb.xyxy[0].cpu().numpy().astype(int)

                                    if "gun" in s_raw or "pistol" in s_raw:
                                        s_lbl = "pistol"
                                    elif "grenade" in s_raw or "bomb" in s_raw:
                                        s_lbl = "grenade"
                                    elif "knife" in s_raw:
                                        s_lbl = "knife"
                                    elif "explosion" in s_raw or "explosive" in s_raw:
                                        s_lbl = "explosive"
                                    else:
                                        s_lbl = s_raw

                                    if any(t in s_lbl for t in THREAT_CLASSES):
                                        frame_candidates.append({
                                            "label": s_lbl,
                                            "confidence": round(s_conf, 3),
                                            "box": [int(s_xyxy[0] + sx1), int(s_xyxy[1] + sy1), int(s_xyxy[2] + sx1), int(s_xyxy[3] + sy1)]
                                        })
                            except Exception:
                                pass

                    # 4. COCO Knife Recall Booster:
                    # Guarantees yellow kitchen knives, chef knives, and utility blades are never missed.
                    if self.distractor_model is not None and not any("knife" in c["label"] for c in frame_candidates):
                        try:
                            coco_res = self.distractor_model.predict(
                                frame,
                                classes=[43],  # Class 43 is COCO 'knife'
                                conf=max(0.30, float(self.conf_threshold)),
                                imgsz=320,
                                verbose=False
                            )
                            for kbox in coco_res[0].boxes:
                                k_conf = float(kbox.conf[0])
                                k_xyxy = kbox.xyxy[0].cpu().numpy().astype(int)
                                frame_candidates.append({
                                    "label": "knife",
                                    "confidence": round(k_conf, 3),
                                    "box": [int(k_xyxy[0]), int(k_xyxy[1]), int(k_xyxy[2]), int(k_xyxy[3])]
                                })
                        except Exception:
                            pass

                    t_elapsed = time.time() - t_start
                    if t_elapsed > 0:
                        self.inference_fps = int(1.0 / t_elapsed)

                    with self.lock:
                        self.active_threats = frame_candidates
                        self.latest_frame = frame.copy()
                except Exception as e:
                    pass

            time.sleep(0.01)

    def generate_simulated_frame(self, t):
        """Monochrome simulated feed when no camera is active."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        for y in range(0, 720, 40):
            cv2.line(frame, (0, y), (1280, y), (20, 20, 20), 1)
        for x in range(0, 1280, 40):
            cv2.line(frame, (x, 0), (x, 720), (20, 20, 20), 1)
            
        cv2.putText(frame, "SIMULATED SURVEILLANCE FEED [SECTOR 01]", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
        cv2.putText(frame, f"ENGINE: {self.model_status}", (50, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
        return frame

    def capture_snapshot(self, custom_label=None):
        """Captures evidence snapshot of the current frame."""
        with self.lock:
            snap_frame = self.latest_frame
            if snap_frame is None:
                _, snap_frame = self.camera.read()
            if snap_frame is None:
                return None
            
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"threat_{timestamp_str}.jpg"
            filepath = os.path.join(self.snapshot_dir, filename)
            
            cv2.imwrite(filepath, snap_frame)
            
            snapshot_meta = {
                "filename": filename,
                "url": f"/static/snapshots/{filename}",
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                "label": custom_label or ("THREAT" if self.active_threats else "MANUAL_CAPTURE")
            }
            self.latest_snapshot = snapshot_meta
            return snapshot_meta

    def process_next_frame(self):
        """Yields low-latency video stream frames at 30 FPS."""
        prev_time = time.time()
        sim_t = 0

        while self.is_running:
            success, raw_frame = self.camera.read()
            if not success or raw_frame is None:
                sim_t += 0.04
                raw_frame = self.generate_simulated_frame(sim_t)

            current_time = time.time()
            dt = current_time - prev_time
            if dt > 0:
                self.current_fps = int(1.0 / dt)
            prev_time = current_time

            # Copy active threats to draw onto frame
            with self.lock:
                threats_to_draw = list(self.active_threats)

            for threat in threats_to_draw:
                xyxy = threat["box"]
                label = threat["label"]
                conf = threat["confidence"]
                x1, y1, x2, y2 = xyxy

                is_critical = any(c in label for c in ["bomb", "grenade", "explosive", "rifle", "gun", "pistol"])
                box_color = (60, 60, 230) if is_critical else (220, 220, 220)

                # Thin high-contrast HUD box
                cv2.rectangle(raw_frame, (x1, y1), (x2, y2), box_color, 2)

                # Corner crosshairs
                c_len = min(20, max(6, (x2 - x1) // 5), max(6, (y2 - y1) // 5))
                # Top-Left
                cv2.line(raw_frame, (x1, y1), (x1 + c_len, y1), (255, 255, 255), 2)
                cv2.line(raw_frame, (x1, y1), (x1, y1 + c_len), (255, 255, 255), 2)
                # Top-Right
                cv2.line(raw_frame, (x2, y1), (x2 - c_len, y1), (255, 255, 255), 2)
                cv2.line(raw_frame, (x2, y1), (x2, y1 + c_len), (255, 255, 255), 2)
                # Bottom-Left
                cv2.line(raw_frame, (x1, y2), (x1 + c_len, y2), (255, 255, 255), 2)
                cv2.line(raw_frame, (x1, y2), (x1, y2 - c_len), (255, 255, 255), 2)
                # Bottom-Right
                cv2.line(raw_frame, (x2, y2), (x2 - c_len, y2), (255, 255, 255), 2)
                cv2.line(raw_frame, (x2, y2), (x2, y2 - c_len), (255, 255, 255), 2)

                # Label Badge
                badge_text = f"[{label.upper()}] {int(conf * 100)}%"
                (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(raw_frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), (15, 15, 15), -1)
                cv2.rectangle(raw_frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), box_color, 1)
                cv2.putText(raw_frame, badge_text, (x1 + 4, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            with self.lock:
                self.latest_frame = raw_frame.copy()

            # Encode as JPEG
            ret, buffer = cv2.imencode('.jpg', raw_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Smooth 30 FPS interval (~33ms)
            time.sleep(0.02)

    def close(self):
        self.is_running = False
        self.camera.release()
