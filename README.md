# CervixScan — AI-Powered Cervical Type Classifier

> A deep learning-based web application that classifies cervical colposcopy images into Type 1, Type 2, or Type 3 transformation zones, built to support clinical screening workflows.

<br>

## Overview

CervixScan is an AI powered web application that allows clinicians and researchers to upload a cervical colposcopy image and receive an instant classification result with per-class confidence scores. It is powered by a transfer-learned VGG16 convolutional neural network, served via a Flask REST API, and accessed through a responsive web frontend.

The three cervical types the model distinguishes are:

| Type | Description |
|------|-------------|
| **Type 1** | Transformation zone fully visible on the ectocervix |
| **Type 2** | Transformation zone partially endocervical, fully visible |
| **Type 3** | Transformation zone partially or fully in the endocervical canal, not fully visible |

---

## Project Structure

```
cervixscan/
│
├── backend/
│   ├── artifact/
│   │   └── model/
│   │       └── cancer_screen_model.tflite   # Converted TFLite model
│   └── app.py                               # Flask API server
│
├── frontend/
│   └── index.html                           # Single-page web application
│
├── notebook/
│   └── cervical_cancer_classification.ipynb # Model training notebook
│
├── requirements.txt                         # Python dependencies
└── README.md
```

---

## ML Model

### Dataset

The model was trained on the **Intel MobileODT Cervical Cancer Screening** dataset, sourced from Kaggle. The dataset contains colposcopy images across three cervical type classes, supplemented with additional annotated images.

| Class  | Training Images |
|--------|----------------|
| Type 1 | 1,440          |
| Type 2 | 4,346          |
| Type 3 | 2,426          |
| **Total** | **8,212**   |

> **Note:** The dataset is significantly imbalanced — Type 2 accounts for over 50% of total samples. This is expected to produce a model with higher precision on Type 2 and lower recall on Type 1.

### Model Architecture

The classifier is built using **transfer learning** on top of a pretrained **VGG16** backbone (ImageNet weights):

```
Layer               Output Shape        Parameters
─────────────────────────────────────────────────────
VGG16 (conv base)   (None, 5, 5, 512)   14,714,688
Flatten             (None, 12800)        0
Dropout (0.5)       (None, 12800)        0
Dense (softmax, 3)  (None, 3)            38,403
─────────────────────────────────────────────────────
Total params:       14,753,091
Trainable params:   7,117,827 (last 5 VGG16 layers unfrozen)
Non-trainable:      7,635,264
```

- **Optimizer:** Adam (`lr=0.0001`)
- **Loss:** Sparse Categorical Cross-Entropy
- **Input size:** 180 × 180 × 3
- **Output:** Softmax probabilities over 3 classes

### Evaluation

| Metric         | Value   |
|----------------|---------|
| Test Loss      | 0.7693  |
| Test Accuracy  | **71.3%** |

### Known Limitations

- **Class imbalance** — Type 2 dominates the training data; the model may underperform on Type 1 cases.
- **Overfitting** — A significant gap exists between training and validation accuracy.
- **Image quality** — The dataset contains images of varying quality, lighting, and scale, which impacts generalization.

Recommended paths for improvement:
1. Acquiring more balanced, high-quality annotated data
2. Stronger regularization (L2, additional dropout layers)
3. Using a lighter backbone (e.g., MobileNetV2, EfficientNetB0)
4. Class-weighted loss to address imbalance
5. Test-time augmentation for more robust predictions

---

## Backend Service

The backend is deployed at **[CervixScan-Service](https://cervixscan-service.onrender.com)**. It utilizes a **Flask** REST API that loads the trained model as a **TFLite** file for efficient inference and exposes endpoints for image classification.

### Endpoints

#### `GET /`
Returns a welcome message and a summary of available endpoints.

**Response:**
```json
{
  "message": "Welcome to CervixScan API",
  "endpoints": {
    "/health": "GET - Health check",
    "/classify": "POST - Classify cervix type from image"
  }
}
```

---

#### `GET /health`
Health check endpoint. Useful for uptime monitoring and deployment health probes.

**Response:**
```json
{ "status": "live" }
```

---

#### `POST /classify`
Accepts a base64-encoded image and returns the predicted cervical type with confidence scores.

**Request Body:**
```json
{
  "image_data": "data:image/jpeg;base64,/9j/4AAQSkZJRgAB..."
}
```

**Success Response:**
```json
{
  "class": "Type 2",
  "confidence": 0.8912,
  "all_probabilities": [0.0521, 0.8912, 0.0567]
}
```

**Error Response:**
```json
{ "error": "No image data provided" }
```

**Image Preprocessing Pipeline:**
1. Base64 decode → NumPy array → OpenCV decode
2. Resize to **180 × 180** pixels
3. Normalize pixel values to `[0.0, 1.0]`
4. Expand dims to shape `(1, 180, 180, 3)`
5. Run TFLite interpreter inference
6. Return argmax prediction + softmax probabilities

---

### Setup & Running Locally

#### Prerequisites

- Python 3.8+
- pip

#### 1. Clone the repository

```bash
git clone https://github.com/your-username/cervixscan.git
cd cervixscan
```

#### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

#### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
flask
flask-cors
tensorflow
numpy
opencv-python
```

#### 4. Place the model file

Ensure the TFLite model is located at:
```
backend/artifact/model/cancer_screen_model.tflite
```

To convert an existing `.h5` model to TFLite:
```python
import tensorflow as tf

model = tf.keras.models.load_model('cancer_screen_model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('cancer_screen_model.tflite', 'wb') as f:
    f.write(tflite_model)
```

#### 5. Run the server

```bash
python backend/app.py
```

The API will be available at `http://localhost:5000`.

---

##  Web Application
The web application is available at **[CervixScan](htts://cervixscan.onrender.com)** 

## End-to-End Flow

```
User uploads image
        │
        ▼
Frontend reads file as base64
        │
        ▼
POST /classify  ──►  Flask API receives base64 image
                            │
                            ▼
                    Decode → Resize (180×180) → Normalize
                            │
                            ▼
                    TFLite Interpreter Inference
                            │
                            ▼
                    Return class + confidence + all_probabilities
        │
        ◄──────────────────────────────────────────────
        │
        ▼
Frontend renders result with animated probability bars
```

---

## Future Improvements

- [ ] **Grad-CAM visualisation** — Highlight the image regions that influenced the prediction
- [ ] **Model retraining** with class-weighted loss to address Type 1 underrepresentation
- [ ] **Batch classification** — Support uploading and classifying multiple images at once
- [ ] **PDF report generation** — Export classification results as a downloadable clinical report
- [ ] **User authentication** — Secure the API for clinical deployment
- [ ] **Logging & monitoring** — Track inference requests and model performance over time
- [ ] **Docker support** — Containerise the backend for portable deployment

---

## Authors

**Samuel Ozechi** · **Adebowale Akande**

Built as part of the Intel MobileODT Cervical Cancer Screening challenge.

---

## Disclaimer

> **CervixScan is intended for research and educational purposes only.**  
> It is not a certified medical device and has not been validated for clinical use. All classification outputs should be reviewed and confirmed by a licensed healthcare professional before any clinical decision is made. The authors accept no liability for misuse of this tool in a clinical context.
