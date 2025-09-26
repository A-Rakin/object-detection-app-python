# YOLOv5 Object Detection (Image & Webcam)

This project provides a straightforward implementation of object detection using the **YOLOv5** (You Only Look Once, version 5) model by Ultralytics. It allows for object detection in both static images and real-time video streams from a webcam.

The small and fast version of the model, **YOLOv5s**, is loaded directly from PyTorch Hub, pre-trained on the comprehensive **COCO dataset** (80 common object classes).

---

## 🚀 Features

* **Image Detection:** Analyze any single image file and display the results with labeled bounding boxes and confidence scores in a new window.
* **Real-Time Webcam Detection:** Stream live video from your default camera, process frames with YOLOv5, and render the results instantly.
* **Minimal Setup:** Uses `torch.hub.load` for quick model retrieval, minimizing setup steps.

---

## 💻 Prerequisites

You need **Python 3.x** and the following libraries installed:

1.  **PyTorch**
2.  **OpenCV**
3.  **NumPy**
4.  **Ultralytics** (for YOLOv5 dependencies)

### Installation

Use `pip` to install the necessary packages.

```bash
# Install PyTorch (refer to the official PyTorch website for CUDA/GPU support if needed)
pip install torch torchvision torchaudio

# Install the other required libraries
pip install opencv-python numpy ultralytics
