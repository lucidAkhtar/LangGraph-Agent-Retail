# LangGraph Agent – Retail

**LangGraph-Agent-Retail** is an **AI-powered retail product recommendation and search agent** built using [LangGraph](https://github.com/langchain-ai/langgraph) and modern LLM tooling.  
It leverages **multi-agent reasoning**, **vector search**, and **natural language understanding** to deliver personalized product suggestions, comparison reasoning, and preference-based filtering for retail catalogs.

This project is designed for **retail AI prototyping** and can be deployed as an API or an interactive [Streamlit](https://streamlit.io) application.

---

## 🚀 Features

- **Conversational Product Search** – Understands natural language queries like _"Find me budget-friendly running shoes under ₹3000"_.
- **Multi-Agent Graph Workflow** – Modular agents for:
  - Preference extraction
  - Product retrieval
  - Filtering
  - Comparison reasoning
  - Recommendation generation
- **Vector Database Search** – Uses FAISS and embeddings (e.g., Cohere) for semantic product retrieval.
- **Retail CSV Dataset Support** – Works on preloaded multi-category retail datasets.
- **Multiple LLM Support** – Works with Ollama, Cerebras, or Gemini APIs.
- **Streamlit UI** – Easy-to-use frontend for live testing.
- **REST API** – FastAPI-style routes for integration into apps.

---
### 📊 Datasets
- The data is picked from [Kaggle](https://www.kaggle.com/datasets/lokeshparab/amazon-products-dataset) which is Amazon Product Sales dataset 2023.
- 10 selected categories with only 50 records have been taken for this exercise.
- This exercise is done using my MacBook Air with 8 GB RAM hence, had compute and infra constraints.

---
### LLM and Embedding Model
- **ollama** has been used as the open source large language model (LLM) with **Mistral** on top of it. 
```bash
from langchain_ollama import OllamaLLM
llm = OllamaLLM(model="mistral",temperature=0.2)
```
- Embedding model is used from [Cohere](https://cohere.com/developers) platform and model name is **embed-english-v3.0**. Get a Free (within limits) API key.
```bash
    import cohere
    import os
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    if not COHERE_API_KEY:
        raise ValueError(f"COHERE API KEY environment variable not set...")
    
    co = cohere.Client(COHERE_API_KEY)
    response = co.embed(
                    texts = batch,
                    model = "embed-english-v3.0",
                    input_type = "search_document"
            )
```
- FAISS is used as a vectorstore to perform efficient similarity search and extraction.
```bash
from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_documents(
                documents = docs,
                embedding = embeddings_model

        )
```
---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/lucidAkhtar/LangGraph-Agent-Retail.git
cd LangGraph-Agent-Retail
```

### 2️⃣ Create & activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate       # Windows
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
### 4️⃣ Setup environment variables
Create a .env file in the root directory:

```bash
COHERE_API_KEY=your-cohere-api-key
```
---
### 🛠 Usage

### Run FastAPI
```bash
uvicorn main:app --reload
```
### Run Streamlit UI
```bash
streamlit run streamlit_app.py
```
Access at -  http://localhost:8501

---

### 🧠 How It Works

#### 1. User Query → Given in natural language.

#### 2. Preference Extractor → Identifies category, budget, brand, etc.

#### 3. Product Retriever → Searches FAISS vector index for relevant items.

#### 4. Product Filter → Removes irrelevant products based on constraints.

#### 5. Comparison Reasoner → Compares shortlisted products.

#### 6. Recommendation Generator → Returns personalized product suggestions.

---

### 🧪 Testing (WIP)

---

### 🐳 Docker Deployment (WIP)


