# 3D Geometry-Based Computer Vision Pipeline for Heavy Vehicle Classification

## 📌 Overview
Standard 2D object detectors (like YOLO) are excellent at finding bounding boxes, but they lack physical scale—a truck 100 meters away looks identical to a car 20 meters away. 

This repository contains a monocular, physics-based computer vision pipeline that measures the real-world width and height of vehicle rear faces to accurately classify heavy vehicles (vans, buses, trucks, semi-trailers). By combining deep learning segmentation with classical homography, we achieve 
real-world metric accuracy without the need for LiDAR or stereo cameras.

### 📸 Visual Demo
![Semi-Trailer Classification](docs/demo_semi_trailer.jpg?v=2)
![Multi-Vehicle Tracking](docs/demo_multi_tracking.jpg?v=2)

> **Note:** This repository is a sanitized version of my Master's Thesis project, completed in collaboration with South Engineering srl and the University of Naples Federico II. Proprietary video data and custom-trained weights have been excluded.

## 🏗️ Architecture Pipeline
1. **Detection & Tracking:** YOLO11s + ByteTrack for robust temporal tracking.
2. **Segmentation:** SAM2 (Segment Anything Model 2) extracts precise vehicle silhouettes.
3. **Geometric Measurement:** A closed-form homography back-projects 2D pixels onto a 3D ground plane, calculating metric width and height.

## 📊 Key Results
*   **Mean Absolute Error (MAE):** **0.128m** (Width) and **0.121m** (Height) across 49 validated passenger vehicles.
*   **Operational Envelope:** We discovered that while custom-trained YOLO models underperform at long ranges (30+ meters) due to feature sparsity, SAM2 with per-transit propagation maintains the semi-trailer class accurately.

## 🚀 Quick Start

### Prerequisites
*   Python 3.10+
*   PyTorch (CPU or CUDA)
*   Ultralytics (YOLO)
*   SAM2 (Meta)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/aminfatemihaghighi/heavy-vehicle-geometry-cv.git
   cd heavy-vehicle-geometry-cv