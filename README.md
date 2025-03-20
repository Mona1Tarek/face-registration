# Face Registration

## Overview
This project implements a **Face Registration System** that captures and stores face images for recognition. It allows users to register their face, which can later be used for face detection and recognition.

## Features
- **Face registration and dataset creation**
- **Real-time face detection using OpenCV**
- **Automatic image capture and storage**
- **Prepares dataset for face recognition models**

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Mona1Tarek/face-registration.git
   cd face-registration
   ```
2. Install required dependencies:
   ```bash
   pip install opencv-python numpy
   ```

## Usage
### Run Face Registration
```bash
python face_registration.py
```

1. The script will open the webcam and detect faces in real time.
2. Once a face is detected, it will capture and store images in the `dataset/` folder.
3. After collecting enough images, press **'q'** to exit.

