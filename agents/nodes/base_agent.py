# To enable modular,sequential and stateful processing by different 
# agents in a langgraph pipeline.
# state allows clean data passing across multiple nodes/agents
# decouples logic : each agent focuses on one task using only relevant
#                     parts of the state.
import pandas as pd
import os
from pydantic import BaseModel,Field,ConfigDict
from typing import List,Dict,Any

class AgentState(BaseModel):

    user_input: str # captures original user_query.Needed by all nodes.
    preferences: Dict[str,Any] = Field(default_factory=dict) # extracted preferences.(eg.-brand,price,range).
                     # extracted by `preference_extractor`
    retrieved_products: List[Any] = Field(default_factory=list)
    filtered_products: pd.DataFrame = Field(default_factory=pd.DataFrame) # products shortlisted based on preferences.
                                    # produced by `product_filter`
    compared_insights:List[str] = Field(default_factory=list) # textual comparison insights between products.
                           # output of `comparison_reasoner` 
    recommendations: List[Any] = Field(default_factory=list) # Final product recommendations
                          # produced by recommendation_generator


    model_config = ConfigDict(arbitrary_types_allowed=True)                     
