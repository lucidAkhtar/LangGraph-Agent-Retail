from langchain_core.prompts import PromptTemplate
from app.schema import RecommendationResponse,RecommendationItem
#from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd
import re
from io import StringIO
import logging
logging.basicConfig(level=logging.INFO)

#llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest",temperature=0.2)
from langchain_ollama import OllamaLLM
llm = OllamaLLM(model="mistral",temperature=0.2)

prompt = PromptTemplate.from_template(
    "Given the user preferences {preferences} and insights {insights}, Return the product comparison table in Markdown format with columns: Product Name, Brand, Match Score, and Recommendation Justification."\
    "Do not provide any other information apart from recommendations and its justication. For example- Important considerations or Note. The justification should be in a way that it can be framed as a column in a dataframe."
)

def parse_markdown_table(markdown_text):

    """
    Parses a markdown table string into a pandas DataFrame.
    Cleans markdown formatting for CSV compatibility.
    """

    # extract the actual table content only (removes everything before the table header)
    lines = markdown_text.strip().splitlines()

     # Remove markdown separator line like ---|---
    lines = [line for line in lines if not set(line.strip()) <= set('-|: ')]
    cleaned = "\n".join(lines)
    try:
        df = pd.read_csv(StringIO(cleaned),sep="|",engine="python",skipinitialspace=True)
        # remove empty columns from leading/trailing pipes
        df = df.dropna(axis=1,how="all")
        df.columns = [col.strip() for col in df.columns]
        df = df.applymap(lambda x:x.strip() if isinstance(x,str)else x)
        return df
    
    except Exception as e:
        print(f"failed to parse markdown table...{e}")

def generate_recommendations(state):

    logging.info(f"generate_recommendations script starts here...")

    chain = prompt | llm 
    response = chain.invoke(
        {
            "preferences": state.preferences,
            "insights": state.compared_insights
        }
    )
    # Extract raw string from response
    #response_text = response.content.strip()
    response_text = response.strip()
    print(f"raw response content is :{response_text}")

    df = parse_markdown_table(response_text)

    if not df.empty:
        logging.info(f"Dataframe generated via parsing function...")
        
        # Dataframe for Streamlit/Internal use only
        #state.recommendations_df = df

        # Store API-safe json
        json_data = df.to_dict(orient='records')
        logging.info(f"shape of df is {df.shape}")
        logging.info(f"columns in df is {df.columns} ")

        print(f"JSON data is - {json_data}")

        clean_data = []
        for rec in json_data:
            logging.info(f"printing the json data type")

            if isinstance(rec,dict):
                clean_data.append(rec)

            elif isinstance(rec,tuple):
                try:
                    if len(rec) == 2 and isinstance(rec[0],str):
                        rec_dict = dict([rec])
                        clean_data.append(rec_dict)
                    else:
                        logging.warning(f"skipping invalid tuple structure - {e}")
                except Exception as e:
                    logging.warning(f"skipping invalid tuple - {e}")
            else:
                logging.warning(f"skipping invalid record - {rec}")
        
    
        state.recommendations = RecommendationResponse(
            recommendations = [RecommendationItem(**rec) for rec in clean_data]
        )

        logging.info(f"generate_recommendations script ends here...")
        return state
    else:
        #state.recommendations_df = None
        state.recommendations = f"Failed to extract recommendations from LLM response..."
        return state