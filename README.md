# Real-Time Multi-Modal Threat Detection System (v2.0)

A high-performance automated computer vision surveillance system designed for real-time weapon and threat detection across optical feeds and security streams. The system couples multi-scale convolutional and vision transformer architectures with a minimal, low-latency monochrome operator console.

Originally researched and developed by Aryan Hardik Dani, Swayamprakash Patro, Sobaan M. Jagirdaar, and Prakhar Jaiswal at MIT World Peace University, and enhanced by Parth Doshi with real-time edge streaming, multi-threat vision backbones, smartphone screen threat inspection, and invariant false-positive suppression.

---

## Overview

Traditional manual security monitoring is vulnerable to cognitive fatigue and elevated false alarm rates. This system provides an automated detection and verification pipeline engineered for optical CCTV feeds and edge video streams:

- Real-Time Optical Threat Detection: Multi-class detection covering handguns, firearms, combat/kitchen knives, and explosive munitions (grenades, bombs).
- False-Positive Immunity: Topological contour analysis and crop-level secondary verification eliminate false alarms triggered by everyday distractors (scissors, rulers, tools, bare smartphones).
- Smartphone Screen Threat Inspection: Dedicated sub-region scanning for weapons displayed on mobile device displays during operator screening.
- Minimal Monochrome Dashboard: High-contrast, dark-mode operator console with real-time telemetry, live incident logging, and evidence snapshot archival.
- High-Throughput Edge Pipeline: Multi-threaded DirectShow frame acquisition delivering 30+ FPS video streaming with low-latency CPU inference.

---

## Detection Engines

The operator console allows seamless runtime switching across specialized model backbones:

| Engine Key | Architecture | Classes Supported | Primary Use Case |
| :--- | :--- | :--- | :--- |
| `sota_threat` (Default) | YOLOv8n Multi-Threat | Gun, Grenade, Knife, Explosion | Primary real-time multi-threat surveillance with native grenade detection |
| `yolo11_tactical` | YOLOv11m Tactical (100 Epochs) | Gun, Knife | High-capacity tactical armament detection |
| `extended` | YOLOv8x Extended | Gun, Rifle, Heavy Weapon, Knife | Extended long-range armament surveillance |
| `world` | YOLO-World v2 | Open-Vocabulary Prompts | Zero-shot open-vocabulary threat scanning with negative prompt grounding |
| `legacy` | YOLOv8n Dual-Class Nano | Pistol, Knife | Baseline lightweight model |

---

## False Positive Mitigation Pipeline

Standard object detection models frequently misclassify non-threatening geometric lookalikes. Version 2.0 incorporates an invariant multi-stage rejection pipeline:

1. Topological Euler Loop Filter: Evaluates object contour hierarchy using `cv2.RETR_CCOMP`. Scissor handles physically require two enclosed finger loops, whereas solid firearm grips have at most one trigger guard opening. Candidates exhibiting dual enclosed holes are suppressed in under 0.05 ms.
2. Crop-Level Verification: Candidates flagged as potential weapons undergo focused verification at 320px against ground-truth distractor classes (scissors, tools).
3. Smartphone Screen Discriminator: Separates bare mobile phone bodies from weapons displayed on screen. When a phone is identified, the screen sub-region is isolated and scanned for displayed munitions without generating false firearm alarms on the phone chassis.
4. Aspect Ratio Filtering: Reject long, narrow items (such as rulers and scales) exceeding an aspect ratio threshold of 3.0.
5. Knife Recall Booster: Incorporates dual-model recall boosting to guarantee reliable detection of yellow-handled kitchen knives, chef cutlery, and utility blades.

---

## System Architecture

```
Camera Feed (DirectShow Threaded Grabber @ 30+ FPS)
    |
    v
Primary Detection Backbone (imgsz=480 @ SOTA Multi-Threat AI)
    |
    +---> Handgun / Firearm Candidate
    |         |
    |         +---> Aspect Ratio > 3.0? -----------> Suppress (Ruler / Scale)
    |         +---> Euler Holes >= 2? -------------> Suppress (Scissors Handles)
    |         +---> 320px Crop Verification? ------> Suppress (Verified Distractor)
    |         +---> Overlaps Phone Body? ----------> Defer to Screen Scanner
    |         +---> Invariant Passes --------------> Trigger CRITICAL Threat Alert
    |
    +---> Knife / Blade Candidate
    |         |
    |         +---> 320px Scissor Check? ----------> Suppress if Shears
    |         +---> Ground-Truth Boost ------------> Trigger Confirmed Knife Alert
    |
    +---> Grenade / Explosive Candidate -----------> Trigger CRITICAL Threat Alert
    |
    +---> Smartphone Screen Scanner
              |
              +---> Crop Screen Sub-Region
              +---> Detect Displayed Munition -----> Trigger Screen Threat Alert
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Webcam or DirectShow-compatible video capture device
- Windows, Linux, or macOS

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AlcesOzdst/Threat_Detection_System.git
   cd Threat_Detection_System
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Web Console

Start the Flask surveillance application:

```bash
python web_app/app.py
```

Open your browser and navigate to:
```
http://localhost:8000
```

To access from other devices on the same local network, use the network IP displayed in the terminal output (e.g., `http://<LAN_IP>:8000`).

---

## Project Structure

```
Threat_Detection_System/
|-- Documentation/
|   |-- PROJECT_LOGBOOK.md           # Master chronological project logbook
|   |-- TECH_STACK_SPECIFICATION.md  # Detailed technical specification
|   `-- Log Book.pdf                 # Historical academic logbook
|-- web_app/
|   |-- app.py                       # Flask application backend
|   |-- detector.py                  # Threaded video capture & detection pipeline
|   |-- templates/
|   |   `-- index.html               # Minimalist monochrome web UI
|   `-- static/
|       |-- css/styles.css           # High-contrast industrial design system
|       |-- js/main.js               # Frontend telemetry & alert logic
|       `-- snapshots/               # Threat evidence snapshot archive
|-- weights/                         # Pre-trained model checkpoints (git-ignored)
|-- requirements.txt                 # Python project dependencies
`-- README.md                        # Project documentation
```

---

## Authors & Contributors

- **Parth Doshi**
  - Email: parthdoshi404@gmail.com
  - Website: [parthdoshi.me](https://www.parthdoshi.me)

- **Aryan Hardik Dani**
  - Email: 1032220083@mitwpu.edu.in
  - LinkedIn: [Aryan Dani](https://linkedin.com/in/aryandani/)

- **Swayamprakash Patro**
- **Sobaan M. Jagirdaar**
- **Prakhar Jaiswal**

### Institution & Acknowledgments

- **Institution**: [MIT World Peace University](https://mitwpu.edu.in)
- **Acknowledgments**: We thank Dr. Vishwanath Karad MIT World Peace University for supporting the baseline research, and the open-source community for computer vision architectures and tooling.

---

## Documentation Links

- [Technical Stack Specification](Documentation/TECH_STACK_SPECIFICATION.md): Detailed hardware, backend, ML model, and frontend specifications.
- [Project Logbook](Documentation/PROJECT_LOGBOOK.md): Chronological record of project development from baseline models through Phase 6 real-time optimization.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
