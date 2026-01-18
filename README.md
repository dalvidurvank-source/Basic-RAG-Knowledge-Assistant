# Basic RAG Knowledge Assistant

A simple Retrieval-Augmented Generation (RAG) based knowledge assistant built using **LangChain** and Python.

This project demonstrates both **standard RetrievalQA** and **Conversational RAG** pipelines.

---

## Features
- Document ingestion and embedding
- Vector-based retrieval
- LLM-powered answer generation
- Conversational memory support

---

## Tech Stack
- Python
- LangChain
- Vector Store (FAISS / Chroma)
- OpenAI / compatible LLM

---

## 📂 Project Structure
Basic-RAG-Knowledge-Assistant/
├── src/
│ ├── retrieval_qa.py
│ └── conversational_rag.py
├── data/
├── requirements.txt
├── README.md
└── .gitignore

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository
```bash
git clone https://github.com/dalvidurvank-source/Basic-RAG-Knowledge-Assistant.git
cd Basic-RAG-Knowledge-Assistant
```

### 2️⃣ Create virtual environment
```bash
python -m venv venv
source venv/Scripts/activate   # Windows
```

### 3️⃣ Install dependencies
```
pip install -r requirements.txt
```

### 4️⃣ How to run 
```
python src/retrieval_qa.py
python src/conversational_rag.py
```


