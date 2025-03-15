# 🚨 Real-Time Multi-Modal Threat Detection System

Welcome to the **Real-Time Multi-Modal Threat Detection System**! 🔒💻 This project integrates **YOLOv11** and **EfficientNetV2** through an adaptive Angular frontend to deliver a cutting-edge solution for automated threat detection in CCTV footage and X-ray scans. Developed by Aryan Hardik Dani, Swayamprakash Patro, Sobaan M. Jagirdaar, and Prakhar Jaiswal at MIT World Peace University, this system addresses the limitations of traditional surveillance by achieving high accuracy and low latency in real-world security scenarios.

## 🌟 Project Motivation

Rising violent attacks and the limitations of human monitoring (e.g., cognitive overload, high false alarm rates) demand advanced surveillance systems. Our goal was to create an automated, reliable threat detection framework that:
- Detects weapons in real-time CCTV feeds with **YOLOv11**.
- Classifies X-ray scans with **EfficientNetV2**.
- Reduces human error and enhances public safety in crowded, high-stake environments like airports and public spaces.

## 🔥 Features

- 🕵️ **Real-Time Weapon Detection**: Identifies knives, pistols, and guns in CCTV footage at 30 FPS using YOLOv11 (mAP 0.83 @ IoU 0.5).
- 📡 **X-Ray Threat Classification**: Classifies scans as "threat" or "safe" with 98.99% accuracy using EfficientNetV2.
- 🎛️ **Adaptive Frontend**: Angular-based dashboard with real-time bounding boxes, confidence thresholding, and reporting (PDF/CSV).
- 📊 **Synthetic Data Enhancement**: GAN-generated images improve generalization across diverse scenarios.

## 🛠️ Technologies Used

- **Python** – Core logic for YOLOv11 and EfficientNetV2 pipelines.
- **YOLOv11** – Object detection on CCTV feeds (trained on OD-Weapon + synthetic data).
- **EfficientNetV2** – X-ray classification (trained on GDXray dataset).
- **Angular** – Frontend framework for real-time visualization and reporting.
- **GANs (DCGAN)** – Synthetic data generation for robust training.
- **Git & GitHub** – Version control and project hosting.

## 🚀 Getting Started

Follow these steps to set up and run the system locally:

1. **Clone this repository:**
   ```sh
   git clone https://github.com/aryan-dani/Threat-Detection-System.git
2. **Navigate to the project directory:**
   ```sh
   cd Threat-Detection-System
3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
4. **Set up the Angular frontend:**
   ```sh
   cd frontend
   npm install
   ng serve

## 📌Future Plans

- Expand dataset with more weapon classes and 3D X-ray scans.
- Integrate federated learning for cross-platform robustness.
- Optimize for edge devices with reduced computational overhead.

## 🎯 What We Learned

This project enhanced our skills in:

- Real-time computer vision with YOLOv11.
- Efficient deep learning classification with EfficientNetV2.
- Frontend-backend integration using Angular.
- Synthetic data generation with GANs for model generalization.

## 📊 Results

- **YOLOv11**: mAP 0.83 (IoU 0.5), Precision 0.90, Recall 0.90, F1 0.87.
- **EfficientNetV2**: Accuracy 98.99%, AUC 0.999, F1 0.9934.
- Outperforms baselines like YOLOv8 (mAP 0.78) and HVGG19 (AUC 0.95).

See the [paper](#) for detailed metrics and comparisons. *(Link to paper TBD.)*

## 🤝 Contributing

Want to enhance this system? We welcome contributions! Here’s how:

1. Fork the Repository
2. Create a New Branch (`git checkout -b feature-name`)
3. Commit Your Changes (`git commit -m "Add some feature"`)
4. Push to the Branch (`git push origin feature-name`)
5. Open a Pull Request 🚀

Suggestions, bug reports, and feedback are appreciated! 😊

## 📬 Contact

Reach out to the team for questions or collaboration:

- 📧 **Email:** 1032220083@mitwpu.edu.in
- 🔗 **LinkedIn:** [Aryan Dani](https://linkedin.com/in/aryandani/) *(Update with actual profiles.)*
- 🌐 **Institution:** [MIT World Peace University](https://mitwpu.edu.in)

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

We thank Dr. Vishwanath Karad MIT World Peace University for supporting this research, and the open-source community for tools like YOLOv11 and EfficientNetV2.
