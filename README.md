LangGraph Agent – Retail
LangGraph-Agent-Retail is an AI-powered retail product recommendation and search agent built using LangGraph and modern LLM tooling.
It leverages multi-agent reasoning, vector search, and natural language understanding to deliver personalized product suggestions, comparison reasoning, and preference-based filtering for retail catalogs.

This project is designed for retail AI prototyping and can be deployed as an API or an interactive Streamlit application.

🚀 Features
Conversational Product Search – Understands natural language queries like "Find me budget-friendly running shoes under ₹3000".

Multi-Agent Graph Workflow – Modular agents for:

Preference extraction

Product retrieval

Filtering

Comparison reasoning

Recommendation generation

Vector Database Search – Uses FAISS and embeddings (e.g., Cohere) for semantic product retrieval.

Retail CSV Dataset Support – Works on preloaded multi-category retail datasets.

Multiple LLM Support – Works with Ollama, Cerebras, or Gemini APIs.

Streamlit UI – Easy-to-use frontend for live testing.

REST API – Flask/FastAPI-style routes for integration into apps.

📂 Project Structure
graphql
Copy
Edit
LangGraph-Agent-Retail/
├── agents/                     # Core LangGraph agent definitions
│   ├── graph.py                 # Graph workflow for multi-agent pipeline
│   └── nodes/                   # Individual agent node implementations
│       ├── base_agent.py
│       ├── cohere_embeddings.py
│       ├── preference_extractor.py
│       ├── product_filter.py
│       ├── product_retriever.py
│       ├── comparison_reasoner.py
│       └── recommendation_generator.py
├── app/                         # API Layer
│   ├── routes.py                # API endpoints
│   ├── schema.py                # Request/response schemas
│   └── services/recommender.py  # Service layer for recommendation logic
├── data/                        # Processed datasets
│   ├── category_wise/           # Category-specific product CSVs
│   ├── filtered_category_wise/  # Cleaned & merged CSVs ready for embedding
│   └── process.py                # Data processing scripts
├── data_raw/                    # Raw retail product CSV files
├── embeddings/                  # Precomputed FAISS vector index & metadata
├── llms/                        # LLM interface wrappers
│   └── ollama_interface.py
├── streamlit_app.py              # Streamlit frontend
├── main.py                       # Entry point for CLI/API
├── test_agents.py                # Unit tests for agent pipeline
├── requirements.txt              # Dependencies
├── Dockerfile                    # Containerization
└── retail-recommender-*.json     # API keys/config (keep secret)
⚙️ Installation
1️⃣ Clone the repository
bash
Copy
Edit
git clone https://github.com/lucidAkhtar/LangGraph-Agent-Retail.git
cd LangGraph-Agent-Retail
2️⃣ Create & activate a virtual environment
bash
Copy
Edit
python -m venv venv
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate       # Windows
3️⃣ Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
4️⃣ Setup environment variables
Create a .env file in the root directory:

ini
Copy
Edit
COHERE_API_KEY=your_cohere_api_key
GEMINI_API_KEY=your_gemini_api_key
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
🛠 Usage
Run Streamlit UI
bash
Copy
Edit
streamlit run streamlit_app.py
Access at http://localhost:8501

Run API Server
bash
Copy
Edit
python main.py
Example request:

bash
Copy
Edit
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{"query": "Suggest the best budget laptops for students"}'
🧠 How It Works
User Query → Given in natural language.

Preference Extractor → Identifies category, budget, brand, etc.

Product Retriever → Searches FAISS vector index for relevant items.

Product Filter → Removes irrelevant products based on constraints.

Comparison Reasoner → Compares shortlisted products.

Recommendation Generator → Returns personalized product suggestions.

📊 Datasets
The system is preloaded with retail datasets covering:

Electronics

Appliances

Clothing & Footwear

Sports & Fitness

Home & Kitchen

Baby Products

Beauty & Grooming
…and more.

🐳 Docker Deployment
bash
Copy
Edit
docker build -t langgraph-retail .
docker run -p 8501:8501 langgraph-retail
🧪 Testing
bash
Copy
Edit
pytest test_agents.py
📌 Roadmap
 Add multilingual support

 Real-time inventory sync

 Voice-based search

 Integrate payment/checkout flows

🤝 Contributing
Pull requests are welcome! Please:

Fork the repo

Create a new branch (feature/your-feature)

Commit changes

Submit a PR

📄 License
This project is licensed under the MIT License.