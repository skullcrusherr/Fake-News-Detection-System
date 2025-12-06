# 📰 Fake News Detection System (Transformer-Based + Django Web App)

This project is an end-to-end **Fake News Classification System** built using:

- **DistilBERT Transformer (Fine-Tuned)**
- **Multi-dataset training (Kaggle, GossipCop, Politifact, Bharat Fake News Kosh, etc.)**
- **Django Web Application**
- **User Authentication (Register/Login/Logout)**
- **History Tracking & Dashboard Pie Charts**
- **Responsive UI with Bootstrap 5 (Dark Theme)**

The system allows users to submit a news headline or short article and returns:
✔ **REAL / FAKE / UNSURE**  
✔ Prediction probabilities  
✔ Logged user history  
✔ Admin-only user stats  

---

## 🚀 Features

### 🔹 **1. Transformer-Based Model (DistilBERT)**
- Fine-tuned on merged datasets from Kaggle, GossipCop, Politifact, and Bharat Fake News Kosh  
- Robust classification capability  
- Outputs prediction + confidence  

### 🔹 **2. Full Django Web App**
- User registration and login  
- Styled dark-mode UI with Bootstrap 5  
- Prediction form with real-time results  
- Pie chart visualization using Chart.js  
- Per-user prediction history  
- Admin-only users list  
- Logging of predictions into database  

### 🔹 **3. Clean Project Architecture**
datasets → preprocessing → DistilBERT fine-tuning → saved model → Django → web interface

yaml
Copy code

### 🔹 **4. Database Logging**
Every prediction is stored with:
- Text
- Label (REAL / FAKE / UNSURE)
- Probabilities
- Timestamp
- User account reference

---

## 📊 System Architecture (Block Diagram)

Below is the project workflow diagram used in the report:

![System Flow](A_flowchart-style_diagram_with_a_dark_gray_backgro.png)

---

## 📦 Project Structure

fake_news_detection/
│
├── dataset/ # All CSV and XLSX datasets
├── ml/ # Training & inference scripts
│ ├── train_transformer.py
│ ├── train_transformer_multi.py
│ ├── transformer_inference.py
│ └── saved_models/
│
├── detector/ # Django app
│ ├── models.py # PredictionLog model
│ ├── views.py # All views
│ ├── urls.py
│ ├── templates/
│ │ ├── base.html
│ │ ├── home.html
│ │ ├── history.html
│ │ ├── login.html
│ │ ├── register.html
│ │ └── users_list.html
│ └── templatetags/
│ └── form_tags.py # Custom Django filter
│
├── fake_news_site/ # Django project settings
│ ├── settings.py
│ ├── urls.py
│ └── asgi/wsgi.py
│
├── manage.py
└── README.md

yaml
Copy code

---

## 🧠 Model Training Overview

### Datasets Used:
- **Fake.csv / True.csv (Kaggle)**
- **Fake_Real.csv**
- **GossipCop Fake / Real**
- **Politifact Fake / Real**
- **Bharat Fake News Kosh**

### Training Pipeline:
1. Merge & clean datasets  
2. Tokenize using `distilbert-base-uncased` tokenizer  
3. Fine-tune for 4 epochs  
4. Achieved **96.2% validation accuracy**  
5. Exported as Transformers model directory  

---

## 🌐 Web Application Screenshots

### ✔ Prediction Interface  
-(Dark themed, Bootstrap 5)

### ✔ History Dashboard  
-With probabilities & timestamps

### ✔ Users List (Admin Only)

*(You can add screenshots here once uploaded)*

---

## 🛠 Installation & Setup

### 1️⃣ Clone the repository
```bash
-git clone https://github.com/skullcrusherr/Fake-News-Detection-System
-cd fake-news-detector
2️⃣ Create a virtual environment
-bash
-Copy code
-python3 -m venv .venv
-source .venv/bin/activate
3️⃣ Install dependencies
-bash
-Copy code
-pip install -r requirements.txt
4️⃣ Apply migrations
-bash
-Copy code
-python manage.py makemigrations
-python manage.py migrate
5️⃣ Run the development server
-bash
-Copy code
-python manage.py runserver
6️⃣ Access in browser
-👉 http://127.0.0.1:8000/

🔒 Authentication
-Users must log in to make predictions

-Each prediction is logged with user data

-Admins can view all registered users
