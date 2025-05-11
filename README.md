

---

# 🧠 Customer Data Deduplication and Processing App

This Streamlit-based web application enables the preprocessing, deduplication, and analysis of customer datasets using NLP, phonetic, and fuzzy matching techniques. It features embeddings with Sentence Transformers and FAISS indexing for fast similarity search, and includes Prometheus metrics for monitoring.

---

## 🚀 Features

* Upload and visualize multiple datasets (CSV, JSON, Excel, TXT)
* Auto-detect customer fields (name, email, phone, etc.)
* Text normalization, phonetic encoding, and fuzzy string matching
* Embedding generation using SentenceTransformers
* Fast nearest-neighbor search with FAISS
* Detect potential duplicates based on configurable similarity threshold
* Spellchecking and phonetic comparison for enhanced matching
* Save and load session data via SQLite
* Live Prometheus metrics (requests, processing time)

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **Data Processing:** pandas, numpy, scipy, scikit-learn
* **NLP & Matching:** sentence-transformers, fuzzywuzzy, phonetics, spellchecker, unidecode
* **Similarity Search:** FAISS
* **Monitoring:** prometheus-client
* **Database:** SQLite
* **Deployment:** Docker

---

## 📦 Installation

### 🔧 Requirements

* Python 3.9+
* pip

### 🐳 Run with Docker

```bash
docker build -t streamlit-app .
docker run -p 8501:8501 streamlit-app
```

### 💻 Run Locally

1. Clone the repo:

```bash
git clone https://github.com/your-username/customer-deduplication-app.git
cd customer-deduplication-app
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Start the app:

```bash
python your_app.py
```

4. Visit `http://localhost:8501` in your browser.

---

## 📂 File Structure

```
├── your_app.py              # Main Streamlit application
├── Dockerfile               # Docker configuration
├── requirements.txt         # Python dependencies
├── customer_data.db         # SQLite DB (auto-generated)
├── README.md                # Project documentation
```

---

## 📈 Prometheus Metrics

Once the app is running, Prometheus metrics are available at:

```
http://localhost:8000
```

Exposed metrics:

* `customer_data_requests`: Total number of user requests
* `data_processing_duration_seconds`: Time spent on each processing session

---

## 🧪 Example Use Cases

* Detect and clean duplicate entries across customer data files.
* Validate and normalize contact information.
* Integrate multiple datasets while avoiding redundancy.
* Monitor performance of data processing workflows.

---

## ✅ Supported File Types

* `.csv`
* `.json`
* `.xlsx`
* `.txt`

---

## 📝 License

MIT License – feel free to use and modify.

---


