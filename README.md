# User Guide: Federated MBTI Personality Classifier

This document provides a detailed user-level overview of the MBTI Personality Classifier system based on my Master's thesis, *"A Comparative Study of Federated and Centralized Learning in Neural Networks"*.

## 🧠 System Overview

This project explores the use of Federated Learning (FL) for MBTI personality classification using text input. The system is implemented with transformer-based models (DistilBERT and TinyBERT) and supports training across multiple simulated clients.

---

## 🌐 Web Application

A publicly accessible web version of the system is available here:

👉 [Launch the Demo App](https://deluxe-crumble-ec9c23.netlify.app/)

### How to Use the Web App:
1. **Input**: Enter up to 500 characters of English text.
2. **Submit**: Click the "Analyze" button.
3. **Output**: Receive an MBTI personality prediction and brief explanations of its traits.

### Known Limitations:
- Input is **limited to 500 characters** due to model cost and response time.
- Only **English** is supported for accurate results.
- The app uses the **DistilBERT model trained on a balanced dataset across 16 clients**, the best-performing configuration from the thesis.

---

## ⚙️ Deployment & Execution Environment

The model is deployed on a **remote server hosted by Stratus FI**, which provides:
- GPU resources for training and inference.
- Hosting for the web interface and backend API.

---

## 🖥 Program Structure

This project includes Python scripts for FL client/server communication and preprocessing pipelines, along with a minimal frontend.

### Key Folders:
- `gpu/`: Client/server scripts for federated training (DistilBERT, TinyBERT)
- `pth_data/`: Preprocessed training and validation datasets
- `site/`: HTML/CSS/JS files for the frontend
- `data/`: Jupyter notebooks for data preparation and augmentation
- `plots/`: Training results and evaluation graphs

---

## 🛠 Installation Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/LuckaHel/FederatedLeraningThesis
cd FederatedLeraningThesis
```

### 2. Set Up Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

---

## 📈 Running the Training Pipeline

Due to the modular structure, training consists of multiple steps:

### Step 1: Preprocess the Dataset
- Open `data_processing.ipynb` and follow the cells:
  - Load and clean MBTI dataset
  - Filter non-English content
  - Balance the dataset and split among 16 clients
  - Save datasets to `pth_data/`

### Step 2: Configure Clients
- Edit `copyClients.py` to match the number of clients
- Run:
```bash
python copyClients.py
```

### Step 3: Launch Federated Training
- Start server and clients using:
```bash
python runAllClients.py
```

The system uses Flower framework to simulate FL rounds, evaluate metrics, and aggregate models.

---

## 📌 Dataset Notes

- **Primary dataset** is based on MBTI-labeled Reddit comments (not publicly distributable).
- You can use [this public Kaggle dataset](https://www.kaggle.com/datasets/datasnaek/mbti-type) with manual adjustments.
- Augmentation (e.g., synonym replacement) is supported using the `augment_texts()` function in the notebook.

---

## 📋 Program Limitations

- No integrated GUI for training — console interaction only
- Requires GPU for efficient training (tested with NVIDIA A100)
- Not a single-step pipeline — user interaction is required in configuration stages

---

