# 🦴 Bone Age Prediction using Deep Learning

An **end-to-end AI-powered medical imaging system** that predicts a child's **bone age** from **hand X-ray images**.  
This helps doctors diagnose **growth disorders**, **endocrine issues**, and **track pediatric development** with speed and accuracy.

---

## 📋 Table of Contents
1. [Overview](#overview)
2. [Problem Definition](#problem-definition)
3. [Data Pipeline](#data-pipeline)
4. [Model Architecture](#model-architecture)
5. [Training Pipeline](#training-pipeline)
6. [Model Evaluation](#model-evaluation)
7. [Deployment](#deployment)
8. [Project Structure](#project-structure)
9. [Technology Stack](#technology-stack)
10. [Key Design Decisions](#key-design-decisions)
11. [End-to-End Flow](#end-to-end-flow)
12. [Real-World Impact](#real-world-impact)
13. [Contributors](#contributors)

---

## 🧠 Overview
**Goal:** Automate bone age estimation from X-ray images using deep learning.  
**Dataset:** RSNA Bone Age Dataset (~12,600 labeled images).  
**Model:** Transfer learning with **Xception** CNN pre-trained on ImageNet.  
**Output:** Predicted bone age (in months).

---

## 🩻 Problem Definition
| Component | Description |
|------------|-------------|
| **Input** | Hand X-ray image (PNG/JPG) |
| **Output** | Predicted bone age (in months) |
| **Challenge** | Manual assessment by radiologists takes 15–30 mins per image; accuracy is crucial for medical diagnosis |

---

## 📦 Data Pipeline

### 1. Data Collection
- **Dataset:** RSNA Bone Age Dataset (~12,600 training images)  
- Each image labeled with:
  - Ground truth bone age (in months)
  - Gender (M/F)
- Images: grayscale X-rays of **left hands**

### 2. Data Preprocessing (`src/data_preprocessing.py`)
- Loads raw CSV metadata → maps image paths  
- Encodes gender numerically  
- Normalizes bone age values  
- Produces ready-to-train DataFrame  

### 3. Data Augmentation
Used during training to prevent overfitting:
- Rotation, zoom, brightness adjustment  
- Horizontal flip, shifting  

### 4. Data Loading
- Uses **Keras `ImageDataGenerator`**  
- Loads images **on-the-fly** (memory efficient)  
- Automatic resizing to **256×256**, normalization to **[0–1]**

---

## 🧩 Model Architecture (`src/model.py`)

### 🏗 Transfer Learning Approach
We fine-tuned the **Xception** model (pre-trained on ImageNet) for regression.

Input (256×256×3)
↓
Xception Base Model (feature extraction)
↓
GlobalAveragePooling2D
↓
Dropout(0.3)
↓
Dense(128, ReLU)
↓
Dropout(0.2)
↓
Dense(64, ReLU)
↓
Dense(1, Linear) → Predicted Bone Age

### 💡 Why This Architecture?
- **Xception:** Highly efficient and accurate for medical images  
- **GlobalAveragePooling:** Reduces parameters and overfitting  
- **Dropout:** Regularization for better generalization  
- **Dense Layers:** Learn fine-grained bone patterns  

---

## ⚙️ Training Pipeline (`src/train.py`)

### Training Steps
1. Split dataset → 80% train / 20% validation  
2. Initialize augmented data generators  
3. Build Xception-based model  
4. Compile with:
   - Optimizer: **Adam (lr=0.0001)**
   - Loss: **Mean Squared Error (MSE)**
   - Metrics: **MAE, MSE**
5. Train for **10 epochs** with callbacks  
6. Save the **best model** automatically

### Callbacks Used
- **ReduceLROnPlateau:** Reduces learning rate when validation loss stops improving  
- **ModelCheckpoint:** Saves best weights  

### ⏱ Training Time
~1–2 hours on GPU (e.g., Kaggle T4 / Colab Pro)

---

## 📊 Model Evaluation (`src/evaluate.py`)

| Metric | Score | Interpretation |
|---------|--------|----------------|
| **MAE** | ~20 months | Avg. error of ±20 months (clinically acceptable) |
| **RMSE** | ~25 months | Root mean squared deviation |
| **R² Score** | ~0.70 | Explains ~70% variance |

➡️ Comparable to **human radiologist** performance!

---

## 🌐 Deployment (`webapp/`)

### 🧠 Flask Backend (`app.py`)
**Architecture:**  
Flask Server → Load Model → Define API Routes → Handle Predictions  

**Routes:**
| Route | Method | Description |
|--------|----------|-------------|
| `/` | GET | Serve HTML upload page |
| `/predict` | POST | Accept image → Predict bone age → Return JSON |
| `/health` | GET | Health check endpoint |

**Prediction Flow**
User uploads image → Flask receives file
→ Resize (256x256) & preprocess
→ model.predict() → returns bone age (months)
→ Convert to years → Send JSON response
→ Delete temporary file


### 🎨 Frontend (`templates/` + `static/`)
- **HTML/CSS/JS** interface for uploading images  
- Displays real-time prediction results  
- Simple, responsive, and user-friendly UI  

---

## 🗂 Project Structure

bone-age-detection/
│
├── src/
│ ├── model.py # Model definition
│ ├── data_preprocessing.py # Data preprocessing
│ ├── train.py # Training logic
│ ├── evaluate.py # Evaluation scripts
│ └── utils.py # Helper utilities
│
├── webapp/
│ ├── app.py # Flask backend
│ ├── templates/ # HTML pages
│ ├── static/ # CSS/JS files
│ └── uploads/ # Temporary image storage
│
├── saved_models/ # Trained model files (.h5, .keras)
├── notebooks/ # Jupyter notebooks for experiments
├── requirements.txt # Dependencies
└── README.md # Documentation


---

## 🧰 Technology Stack

| Component | Technology | Purpose |
|------------|-------------|----------|
| Deep Learning | TensorFlow / Keras | Model building & training |
| Pre-trained Model | Xception (ImageNet) | Transfer learning backbone |
| Data Handling | Pandas, NumPy | Data manipulation |
| Image Processing | PIL, OpenCV | Image preprocessing |
| Web Framework | Flask | API and web deployment |
| Frontend | HTML, CSS, JS | User interface |
| Model Storage | HDF5 / Keras format | Model serialization |

---

## 🧱 Key Design Decisions

1. **Transfer Learning > Training from Scratch**  
   → Faster training, higher accuracy with limited data  
2. **Keras Functional API**  
   → More flexible & debuggable architecture  
3. **Data Generators**  
   → Memory-efficient and scalable  
4. **Flask Backend**  
   → Lightweight and easy to deploy  
5. **Modular Codebase**  
   → Reusable, maintainable, and production-ready  

---

## 🔁 End-to-End Flow

### **Training Phase (GPU/Kaggle)**
Dataset → Preprocessing → Augmentation → Model → Training → Validation → Save Best Model


### **Deployment Phase (Local/Server)**
User → Web UI → Upload Image → Flask API → Load Model → Preprocess → Predict → Display Result


---

## 🌍 Real-World Impact

**Before:** Radiologist manually compares X-ray to atlas (takes 15–30 minutes).  
**After:** AI predicts in **<2 seconds** with similar accuracy.

### 🏥 Use Cases
- Pediatric growth disorder diagnosis  
- Endocrine issue detection  
- Treatment monitoring  
- Forensic age estimation  

### 🧬 Analogy
| Stage | Analogy |
|--------|----------|
| Training | Medical school – learning from thousands of X-rays |
| Xception | Textbook knowledge – general image features |
| Fine-tuning | Residency – specialization in bone age |
| Web App | Clinic – providing instant second opinions |

---

## 🚀 Production-Ready Highlights
✅ Modular, maintainable code structure  
✅ GPU-accelerated training  
✅ Efficient data pipeline  
✅ Flask-based REST API  
✅ Clean UI  
✅ Model versioning support  
✅ Error handling and validation  

---

## 👨‍💻 Contributors
| Name | Role | GitHub |
|------|------|--------|
| **Aryan Kanchan** | Lead Developer / ML Engineer | [@aryankanchan018](https://github.com/aryankanchan018) |
| **[Add teammate name here]** | Collaborator / Model Optimization | *(GitHub link)* |

---

## 🏁 License
This project is released under the **MIT License**.  
You are free to use, modify, and distribute with attribution.

---

> *“AI that learns from thousands of X-rays to assist doctors — not replace them.”* 🧠
