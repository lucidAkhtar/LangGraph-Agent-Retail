from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
#from langchain_google_genai import ChatGoogleGenerativeAI
import logging
logging.basicConfig(level=logging.INFO)

#llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash-latest',temperature=0.2)
from langchain_ollama import OllamaLLM
llm = OllamaLLM(model="mistral",temperature=0.2)
parser = JsonOutputParser()

"""
You have a dataframe provided - {df}. In this , a column named category is available. Try to match the must have features as mentioned in the keys as close or equal to the category in the dataframe.
"""

prompt = PromptTemplate.from_template(

    """Extract user preferences as JSON from the following text:
    {user_input} 

    The JSON must have exactly these keys : "brand", "budget","must have features".
    - Always enclose keys and string values in double quotes.
    - do not include any explanation,commentary,or extra text - only the JSON object. """

)

def extract_preferences(state):

    logging.info(f"extract_preferences script starts here...")

    chain = prompt | llm | parser
    preferences = chain.invoke({"user_input":state.user_input})
    state.preferences = preferences
    logging.info(f"extract_preferences script ends here...")
    return state

