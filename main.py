from fastapi import FastAPI
from app.routes import router as recommendation_router

app = FastAPI(
    title = "Product Recommendation with LangGraph Agents",
    description = "Recommends products based on user preference.",
    version= "1.0"
)

# Register routes
app.include_router(recommendation_router)

@app.get("/health")
async def health_check():
    return {"status":"ok"}

