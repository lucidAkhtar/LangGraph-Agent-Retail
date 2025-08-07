from fastapi import FastAPI,Request
from pydantic import BaseModel
from agents.graph import build_agent_graph

app = FastAPI()

class userRequest(BaseModel):
    user_input: str

@app.post("/recommend")
async def recommend(user_req:userRequest):
    result = build_agent_graph(user_req.user_input)
    return {"recommendations":result}