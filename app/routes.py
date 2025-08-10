import asyncio
from fastapi import APIRouter
from app.schema import userRequest,RecommendationResponse,RecommendationItem
from app.services.recommender import get_recommendations
import logging
logging.basicConfig(level=logging.INFO)

router = APIRouter()

@router.post("/recommend",response_model = RecommendationResponse)
async def get_recommend(user_req:userRequest):
    """
    Handles recommenations requests without blocking FastAPIs event loop.
    
    """
    logging.info("Router is executing...")
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, # default thread pool
        get_recommendations,
        user_req.user_input
    )

    logging.info(f"Type of result is - {type(result)}")
    if isinstance(result,RecommendationResponse):
        return result
    
    if isinstance(result,list) and all(isinstance(i,dict) for i in result):
        return RecommendationResponse(
            recommendations = [RecommendationItem(**rec)
                               for rec in result ]
                               )

    raise ValueError(f"Unexpected result format : {result}")