from fastapi import FastAPI,Depends
from pydantic import BaseModel
from typing import Tuple
from app.routes import router as recommendation_router
from agents.nodes.product_retriever import load_and_embed_products
import logging
logging.basicConfig(level=logging.INFO)


app = FastAPI(
    title = "Product Recommendation with LangGraph Agents",
    description = "Recommends products based on user preference.",
    version= "1.0"
)

# Dependency Injection
def get_vectorstore() -> Tuple:

    if not hasattr(get_vectorstore,"cache"):

        logging.info(f"loading vectorstore at the beginning...")
        get_vectorstore.cache = load_and_embed_products()

    return get_vectorstore.cache

# Pass the dependency to your routes
app.dependency_overrides[Tuple] = get_vectorstore

# Register routes
app.include_router(recommendation_router)

@app.get("/health")
async def health_check():
    return {"status":"ok"}

