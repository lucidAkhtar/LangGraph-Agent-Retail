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
def get_vectorstore_and_df() -> Tuple:

    if not hasattr(get_vectorstore_and_df,"cache"):
        logging.info(f"loading vectorstore and dataframe for the first time...")
        get_vectorstore_and_df.cache = load_and_embed_products("data/filtered_category_wise/all_data_v1.csv")

    return get_vectorstore_and_df.cache

# Pass the dependency to your routes
app.dependency_overrides[Tuple] = get_vectorstore_and_df

# Register routes
app.include_router(recommendation_router)

@app.get("/health")
async def health_check():
    return {"status":"ok"}

