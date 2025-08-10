from langgraph.graph import StateGraph
from agents.nodes.base_agent import AgentState
from agents.nodes.preference_extractor import extract_preferences
from agents.nodes.product_retriever import retrieve_products,load_and_embed_products
from agents.nodes.product_filter import filter_products
from agents.nodes.comparison_reasoner import compare_products
from agents.nodes.recommendation_generator import generate_recommendations
import logging
logging.basicConfig(level=logging.INFO)
import pandas as pd

# global cache dictionary
query_cache = {}

def build_agent_graph(vectorstore,df):

    logging.info(f"build_agent_graph script starts here...")

    builder = StateGraph(AgentState)
   
    # using partials to inject dependencies
    from functools import partial
    retrieve_with_data = partial(retrieve_products,vectorstore=vectorstore,df=df)

    builder.add_node("extract_preferences",extract_preferences)
    builder.add_node("retrieve_products",retrieve_with_data)
    builder.add_node("filter_products",filter_products)
    builder.add_node("compare_products",compare_products)
    builder.add_node("generate_recommendations",generate_recommendations)

    builder.set_entry_point("extract_preferences")
    builder.add_edge("extract_preferences","retrieve_products")
    builder.add_edge("retrieve_products","filter_products")
    builder.add_edge("filter_products","compare_products")
    builder.add_edge("compare_products","generate_recommendations")

    logging.info(f"build_agent_graph script ends here...")

    return builder.compile()


def run_agent(user_input:str):

    key = user_input.strip().lower()
    if key in query_cache:
        logging.info(f"Cache Hit,returning cached result...")
        return query_cache[key]

    vectorstore,df = load_and_embed_products("data/filtered_category_wise/all_data_v1.csv")
    logging.info("embeddings extracted and dataframe created...")

    agent = build_agent_graph(vectorstore,df)

    initial_state = AgentState(
            user_input = user_input,
            preferences = {},
            retrieved_products = [],
            filtered_products = pd.DataFrame(),
            compared_insights = [],
            recommendations = []

    )

    final_state = agent.invoke(initial_state)
    query_cache[key] = final_state['recommendations']

    return final_state['recommendations']


