from pydantic import BaseModel,Field
from typing import List,Optional

class userRequest(BaseModel):
    user_input:str

class RecommendationItem(BaseModel):
    product_name: Optional[str] = Field(None,alias="Product Name")
    brand_name: Optional[str] = Field(None,alias="Brand")
    match_score: Optional[float] = Field(None,alias="Match Score")
    justification: Optional[str] = Field(None,alias="Justification")

    
    model_config = {"validate_by_name": True}

class RecommendationResponse(BaseModel):
    recommendations: List[RecommendationItem] 