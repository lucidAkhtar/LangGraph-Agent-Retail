from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash-latest',temperature=0.2)
parser = JsonOutputParser()


prompt = PromptTemplate.from_template(
    "Extract user preferences as JSON (keys: brand,budget,must have features) from: \n\n {user_input}"
)

def extract_preferences(state):

    chain = prompt | llm | parser
    preferences = chain.invoke({"user_input":state.user_input})
    state.preferences = preferences
    return state

