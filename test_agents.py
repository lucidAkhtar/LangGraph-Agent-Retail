from dotenv import load_dotenv
load_dotenv()
import time
import pandas as pd
from pprint import pprint


from agents.nodes.base_agent import AgentState
from agents.nodes.preference_extractor import extract_preferences
from agents.nodes.product_retriever import load_and_embed_products,retrieve_products
from agents.nodes.product_filter import filter_products
from agents.nodes.comparison_reasoner import compare_products
from agents.nodes.recommendation_generator import generate_recommendations


pprint("Loading vectorstore from FAISS index...")
vectorstore = load_and_embed_products()
pprint("Loading vectorstore from FAISS is done...")

pprint(f"AgentState initialising with user_input...")
state = AgentState(
    #user_input="I want a casual shoes under 2500." # 3 records as output
    user_input = "designer kurti for wedding under 3000" # 1 record dummy as output 
    #user_input = "mens sandals and boots under 2000" # 0 records as output
    #user_input = "sandals and boots under 2000" # 1 record as output
    )
pprint(f"AgentState initialised with user_input...")

state = extract_preferences(state)
pprint(f"\n [extract_preferences] => {state.preferences}")

df = pd.read_csv("data/filtered_category_wise/data_for_embedding.csv")
state = retrieve_products(state,vectorstore,df)
pprint(f"\n [retrieve_products] => {len(state.retrieved_products)} products found")


state.filtered_products = state.retrieved_products.copy()

state = filter_products(state)
pprint(f"\n [filtered_products] => {len(state.filtered_products)}products after filtering")

state = compare_products(state)
pprint(f"\n [compare_products]=> {state.compared_insights}")
#state.compared_insights.to_csv("compared_insights.csv",index=False)


state = generate_recommendations(state)
pprint(f"\n [generate_recommendations]=> {state.recommendations}")
#state.recommendations.to_csv("recommendations.csv",index=False)



pprint("DONE...")
      




