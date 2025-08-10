from langchain_core.prompts import  PromptTemplate
#from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd
# Parse markdown table to Dataframe
from io import StringIO 
import re
import logging
logging.basicConfig(level=logging.INFO)

#llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest",temperature=0.2)
from langchain_ollama import OllamaLLM
llm = OllamaLLM(model="mistral",temperature=0.2)

prompt = PromptTemplate.from_template(

    """
    You are a precise and detail-oriented shopping assistant.

    INPUT DATA:
    - user preferences: {preferences}
    - products name: {products}

    Your task is to compare ONLY the provided products against the user preferences.
    - Do NOT invent,assume or add products,brands,must have features,or prices that are not explicitly present in the product list data.

    OUTPUT REQUIREMENTS:
    - output must be in raw **CSV format** (no markdown,no code blocks, no extra text)
    - Columns must be exactly 
    1. Product Name
    2. Price
    3. Brand
    4. Features
    5. Match Score

    - Enclose every field in double quotes.
    - Use a numeric score between 0 and 100 for Match Score. Do not include %/ or percentages in that.
    - Preserve original wording for Product Name,Brand,and Features as given.
    - Return only the CSV data with one product per row.

    """
   
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

    logging.info(f"comparison_reasoner script starts here...")

    # 1. Format Input prompt
    input_text = prompt.format(

        products = state.filtered_products.to_dict(orient="records"),
        preferences = state.preferences
    )
    # 2. Invoke LLM
    response = llm.invoke(input_text)
    # Extract actual string from AIMessage
    #response_text = response.content.strip()
    response_text = response.strip()
    print(f"Response text generated: {response_text}")
   
    try:
        # 3. Detect CSV header
        logging.info(f"CSV header detection starts...")

        if "," in response_text and re.search(r"(?i)(product name|price|brand).*?,",response_text):
            print(f"Detected CSV format...")
            df = pd.read_csv(StringIO(response_text),on_bad_lines="skip")
            print(f"parsed csv dataframe...")
            print(df.head())
            state.compared_insights = df
            logging.info(f"comparison_reasoner script ends here...")
            return state
        
        # 4. Fallback- try markdown table
        elif "|" in response_text:
            print(f"Detected markdown table...")
            df = parse_markdown_table(response_text)
            print(f"parsed markdown table DataFrame:")
            print(df.head())
            state.compared_insights = df
            logging.info(f"comparison_reasoner script ends here...")
            return state
        
        else:
            state.compared_insights = f"No table format detected..."
            logging.info(f"comparison_reasoner script ends here...")
            return state
        
    except Exception as e:

        print(f"Failed to parse markdown table...:{e} ")
        state.compared_insights = "Failed to convert LLM response to table..."
        return state
