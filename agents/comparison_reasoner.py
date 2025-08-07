from langchain_core.prompts import  PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd
# Parse markdown table to Dataframe
from io import StringIO 
import re
import csv

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest",temperature=0.2)

prompt = PromptTemplate.from_template(
    "You are a smart shopping assistant. The user is looking for a product with the following preferences: {preferences}." \
    "Here are some product options: {products}. " \
    
    "Compare the mentioned products based on user preferences." \
    "Return your output in **CSV format** with the following columns:"\
    "Product Name, Price, Brand, Features, Match Score." \
    "**Enclose every field in double quotes** to handle commas safely."\
    "No markdown formatting,no code block.Only the raw CSV text."
   
)

def parse_markdown_table(markdown_text:str) -> pd.DataFrame:

    lines = markdown_text.strip().splitlines()
    table_lines = [line for line in lines if "|" in line]

    # Drop alignment row (eg- |:---|:---- etc.)
    if len(table_lines) >= 2 and re.match(r'^\s*\|[:\- ]+\|\s*$',table_lines[1]):
        del table_lines[1]


    # Clean and prepare CSV-like string
    pseduo_csv = "\n".join(
        line.strip().strip('|').strip() for line in table_lines
    )

    # Replace multiple spaces (if any) inside cells with a single space
    pseduo_csv = re.sub(r'\s{2,}',' ',pseduo_csv)

    # parse with pandas
    df = pd.read_csv(StringIO(pseduo_csv))
    return df

def compare_products(state):

    # 1. Format Input prompt
    input_text = prompt.format(

        products = state.filtered_products.to_dict(orient="records"),
        preferences = state.preferences
    )
    # 2. Invoke LLM
    response = llm.invoke(input_text)
    # Extract actual string from AIMessage
    response_text = response.content.strip()
    print(f"Response text generated: {response_text}")
   
    try:
        # 3. Detect CSV header
        if "," in response_text and re.search(r"(?i)(product name|price|brand).*?,",response_text):
            print(f"Detected CSV format...")
            df = pd.read_csv(StringIO(response_text))
            print(f"parsed csv dataframe...")
            print(df.head())
            state.compared_insights = df
            return state
        
        # 4. Fallback- try markdown table
        elif "|" in response_text:
            print(f"Detected markdown table...")
            df = parse_markdown_table(response_text)
            print(f"parsed markdown table DataFrame:")
            print(df.head())
            state.compared_insights = df
            return state
        
        else:
            state.compared_insights = f"No table format detected..."
            return state
        
    except Exception as e:

        print(f"Failed to parse markdown table...:{e} ")
        state.compared_insights = "Failed to convert LLM response to table..."
        return state
