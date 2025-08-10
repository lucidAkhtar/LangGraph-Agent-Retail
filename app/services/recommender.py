from agents.graph import run_agent

# You can unit test 'get_recommendations' without touching FastAPI.

def get_recommendations(user_input:str)-> list[str]:

    """
    Blocking call the runs the agent pipeline.
    Separated here so we can run it in background thread.
    
    """

    return run_agent(user_input)