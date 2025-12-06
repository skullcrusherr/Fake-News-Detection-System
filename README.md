# 📰 Fake News Detection System (Transformer-Based + Django Web App)

A complete end-to-end Fake News Detection system built using **DistilBERT Transformers** for NLP and **Django** for the web interface.  
The system classifies news as **REAL**, **FAKE**, or **UNSURE**, provides confidence scores, tracks user prediction history, and includes a responsive dark-themed dashboard.

This project demonstrates practical Machine Learning + Full Stack development for academic and real-world applications.

---

## 🚀 Features

### 🔹 Transformer-Based Fake News Classifier
- Fine-tuned **DistilBERT** model  
- Trained on multiple datasets:
  - Kaggle Fake/Real
  - Fake_Real.csv
  - GossipCop Fake/Real
  - Politifact Fake/Real
  - Bharat Fake News Kosh (Excel)  
- Outputs: **REAL / FAKE / UNSURE**  
- Provides probability scores

### 🔹 Full Django Web Application
- User Registration / Login / Logout  
- News input form to test model  
- History page showing all predictions  
- Admin-only list of all users  
- Responsive modern UI with Bootstrap 5  
- Chart.js Pie chart showing prediction stats  

### 🔹 Database Logging
Each prediction stores:
- News text  
- Prediction label  
- REAL probability  
- FAKE probability  
- Timestamp  
- User reference  

---

## 📊 System Architecture Diagram

A high-quality block diagram (PNG) explaining the system flow:

```
Datasets (Fake.csv, True.csv, Politifact, GossipCop, Bharat Fake News Kosh)
        ↓
Data Preprocessing & Cleaning
        ↓
DistilBERT Fine-Tuning (Transformer Training)
        ↓
Saved Transformer Model (HuggingFace format)
        ↓
Django Web App
    ├── Login / Register / Logout
    ├── Prediction Form
    ├── Transformer Inference Engine
    ├── Save Predictions to Database
    └── Dashboard Analytics (Pie Chart)
```

*(Place your PNG diagram here once uploaded into the repository.)*

---

## 📁 Project Structure

```
fake_news_detection/
│
├── dataset/                    # All CSV/XLSX files (tracked via Git LFS)
├── ml/                         # Training & inference scripts
│   ├── train_transformer.py
│   ├── train_transformer_multi.py
│   ├── transformer_inference.py
│   └── saved_models/           # Fine-tuned DistilBERT (Git LFS)
│
├── detector/                   # Django application
│   ├── models.py               # PredictionLog model
│   ├── views.py                # Web logic
│   ├── urls.py
│   ├── templates/              # HTML templates (Bootstrap Dark UI)
│   │   ├── base.html
│   │   ├── home.html
│   │   ├── history.html
│   │   ├── login.html
│   │   ├── register.html
│   │   └── users_list.html
│   └── templatetags/
│       └── form_tags.py        # Custom django tag for styling form fields
│
├── fake_news_site/             # Django site config
│   ├── settings.py
│   ├── urls.py
│   └── wsgi/asgi.py
│
└── manage.py
```

---

## 🧠 Model Training Overview

### 1️⃣ Datasets Used
- Fake.csv / True.csv  
- Fake_Real.csv  
- GossipCop Fake / Real  
- Politifact Fake / Real  
- Bharat Fake News Kosh (Excel)

### 2️⃣ ML Pipeline
- Clean & merge datasets  
- Tokenize using `distilbert-base-uncased`  
- Train with HuggingFace Trainer  
- Evaluate accuracy & F1  
- Save model to `saved_models/`

---

## 💻 Running the Project

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Create & Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Apply Migrations
```bash
python manage.py makemigrations
python manage.py migrate
```

### 5. Run the Server
```bash
python manage.py runserver
```

Open the app at:
👉 http://127.0.0.1:8000/

---

## 👨‍💻 Authentication

| Role | Permissions |
|------|-------------|
| **User** | Make predictions, view personal history |
| **Admin** | View ALL users and their prediction counts |

Passwords are securely hashed by Django.

---

## 📈 Dashboard & Visualizations

- Pie chart showing total REAL / FAKE / UNSURE predictions  
- Latest prediction history table  
- Time-stamped logs  
- User-wise statistics  

---
