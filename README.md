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
- **REST API** – Flask/FastAPI-style routes for integration into apps.

---


---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/lucidAkhtar/LangGraph-Agent-Retail.git
cd LangGraph-Agent-Retail


python -m venv venv
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate       # Windows


 How It Works
User Query → Given in natural language.

Preference Extractor → Identifies category, budget, brand, etc.

Product Retriever → Searches FAISS vector index for relevant items.

Product Filter → Removes irrelevant products based on constraints.

Comparison Reasoner → Compares shortlisted products.

Recommendation Generator → Returns personalized product suggestions.

