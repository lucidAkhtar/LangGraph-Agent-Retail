from dotenv import load_dotenv
load_dotenv()
import time
from pprint import pprint


from agents.base_agent import AgentState
from agents.preference_extractor import extract_preferences
from agents.product_retriever import load_and_embed_products,retrieve_products
from agents.product_filter import filter_products
from agents.comparison_reasoner import compare_products
from agents.recommendation_generator import generate_recommendations


pprint("Loading vectorstore from FAISS index...")
start = time.time()
vectorstore,df = load_and_embed_products("data/filtered_category_wise/all_data_v1.csv")
end = time.time()
pprint(f"time taken is {end-start:.2f}s")
pprint("Loading vectorstore from FAISS is done...")

pprint(f"AgentState initialising with user_input...")
state = AgentState(
    user_input="I want a casual shoes for office use under 1000 rupees."
    )
pprint(f"AgentState initialised with user_input...")

state = extract_preferences(state)
pprint(f"\n [extract_preferences] => {state.preferences}")

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
state.recommendations.to_csv("recommendations.csv",index=False)

pprint("DONE...")
      




