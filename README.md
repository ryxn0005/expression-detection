# Expression Detection

Real-time facial expression classification using a ConvNet, Keras, and OpenCV.

## Features
- Live webcam inference with face detection (Haar cascade)
- Grayscale 48×48 preprocessing with normalization (÷255)
- Model loaded from `models/convnet.keras`
- On-screen label, confidence, and FPS

---

## Quick start

### 1) Clone
```bash
git clone https://github.com/ryxn0005/expression-detection.git
cd expression-detection
``` 


### 2) (Recommended) create a virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3) Install requirement libraries
```bash
pip install -r requirements.txt
```

### 4) Run the application
```bash
python app.py
```


The window will open, faces will be boxed, and the predicted emotion plus confidence will be drawn above the face.
Press q or Esc to quit.
