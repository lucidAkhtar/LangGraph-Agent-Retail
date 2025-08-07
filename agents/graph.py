from langgraph.graph import StateGraph
from agents.base_agent import AgentState
from agents.preference_extractor import extract_preferences
from agents.product_retriever import retrieve_products,load_and_embed_products
from agents.product_filter import filter_products
from agents.comparison_reasoner import compare_products
from agents.recommendation_generator import generate_recommendations
import logging
logging.basicConfig(level=logging.INFO)


def build_agent_graph(vectorstore,df):

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

    return builder.compile()


def run_agent(user_input:str):

    vectorstore,df = load_and_embed_products("data/filtered_category_wise/all_data_v1.csv")
    agent = build_agent_graph(vectorstore,df)

    initial_state = AgentState(
            user_input = user_input,
            preferences = {},
            filtered_products = None,
            compared_insights = "",
            recommendations = []

    )

    final_state = agent.invoke(initial_state)
    return final_state.recommendations


