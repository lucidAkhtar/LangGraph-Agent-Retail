from langchain_core.prompts import PromptTemplate
from app.schema import RecommendationResponse,RecommendationItem
#from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd
import re,json
from io import StringIO
import logging
logging.basicConfig(level=logging.INFO)

#llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest",temperature=0.2)
from langchain_ollama import OllamaLLM
llm = OllamaLLM(model="mistral",temperature=0.2)


prompt_json = PromptTemplate.from_template(

    """
    Given the user preferences and and a list of product insights,analyze the products and return a JSON list of recommended products.
    
    User Preferences: {preferences}
    Product Insights (JSON list): {insights}
    
    TASK:
    Go through each product in the `Product Insights` list.If a product matches the user's preferences,create a new JSON object for it.
    Each new JSON object must have the following keys : `ProductName`,`Brand`,`MatchScore` and `Justification`.

    RULES:
    - Only use the data from the provided `Product Insights` list.Do not invent any new products.
    - If a field is missing ,use "N/A" for strings and 0 for numbers.
    - If no products in the list match the preferences, return an empty json list `[]`.
    - Do not inlcude any extra text,conversation,or markdown before or after the json list.
    - The final output must be a valid json list.

    Return the JSON list here: 
    """
)

def generate_recommendations(state):

    logging.info(f"generate_recommendations script starts here...")

    insights_json_list = None

    logging.info(f"type of state.compared_insights is - {type(state.compared_insights)}")
    logging.info(f"state.compared_insights df is - {state.compared_insights}")

    #1. check if the input is a list and convert it to a dataframe if needed
    if isinstance(state.compared_insights,list):
        insights_json_list = state.compared_insights
    elif isinstance(state.compared_insights,pd.DataFrame):
        insights_json_list = state.compared_insights.to_dict(orient='records')
    else:
        logging.error(f"unexpected data type for state.compared_insights-{type(state.compared_insights)}")
        state.recommendations = f"Failed to process insights due to incorrect data format..."
        return state

    # use the json-focused prompt
    if insights_json_list is not None:

        chain_json = prompt_json | llm 
        response = chain_json.invoke(
            {
                "preferences": state.preferences,
                #"insights": state.compared_insights
                # pass the structured data to the llm
                "insights": json.dumps(insights_json_list,indent=2)
            }
        )

        # the LLM response might be wrapped in ```json...```
        # we use regex to extract only the json part

        match = re.search(r'```json\n(.*)\n```',response,re.DOTALL)
        if match:
            json_string = match.group(1).strip()
        else:
            json_string = response.strip()

        logging.info(f"extracted json string - {json_string}")

        try:
            # load the raw string as a json list
            json_data = json.loads(json_string)
            # create the list of pydantic models directly
            recommendations = [RecommendationItem(**rec) for rec in json_data]

            logging.info(f"Successfully parsed {len(recommendations)} recommendations.")

            state.recommendations = RecommendationResponse(recommendations = recommendations)

        except json.JSONDecodeError as e:
            logging.error(f"failed to decode json from llm response- {e}")
            state.recommendations = f"Failed to extract recommendations from LLM response due to invalid json format..."
        
        except Exception as e:
            logging.error(f"an unxpected error occured- {e}")
            state.recommendations = "an error occured while processing the recommendations."

    else:
        state.recommendations = f"No insights data provided..."

    logging.info(f"generate recommendations scripts done...")

    return state
