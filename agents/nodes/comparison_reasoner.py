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


"""
    You are a precise and detail-oriented shopping assistant.

    INPUT DATA:
    - user preferences: {preferences}
    - products name: {products}

    Your task is to compare ONLY the provided {products} against the user {preferences}.
    - Focus on the keywords present in the {preferences} and if those keywords are present in the {products}, do consider those records or products as well.Include them in your response. Also, you have to understand the semantics and contextual meaning between the words of {products} and {preferences}.
    - If there are no products in the {products},state clearly, that no products were found that matched the search criteria.
    - Do not provide any suggestions,code, examples, any kind of recommendation or justification apart from what is mentioned in OUTPUT REQUIREMENTS.
    - If the count of {products} is zero , do not add any product details from your end. Do not add any extra data point.
    - **Do NOT invent,assume or add** products,brands,must have features,or prices that are not explicitly present in the {products} list data.

    OUTPUT REQUIREMENTS:
    - output must be in raw **CSV format** (no markdown,no code blocks, no extra text)
    - Columns must be exactly 
    1. ProductName - Do not have any inverted commas at start or end.Keep it clean. type is string
    2. Price - type is float
    3. Features - type is string
    4. MatchScore - type is integer

    - Enclose every field in double quotes.
    - Use a numeric score between 0 and 100 for "MatchScore" column. Do not include %/ or percentages in that.
    - Preserve original wording for ProductName,Brand,and Features as given.
    - Return only the CSV data with one product per row.
    - Ensure that all the above (4) columns are returned with the correct data type.

"""

prompt = PromptTemplate.from_template(
    """
    You are a precise and detail-oriented shopping assistant.

    INPUT:
    - user preferences: {preferences}
    - products data: {products}

    TASK:
    - Compare each product in the `products data` table against the `user preference`.
    - Generate a CSV string of products that match the preferences.

    RULES:
    - **CRITICAL** - only use the provided `products data`.Do not invent,assume, or add any information.
    - Output must be valid CSV only.If no products match,return only the CSV header.
    - Do not use Python code, .join(), f-strings,markdown,or explanations.
    - Each row must be plain text csv (e.g.,`Fila Mens Shoes,1999.0,"casual,leather",90`) 
    - The output must be a valid ,raw CSV. No surrounding text,no code blocks, no explanations.
    - The CSV must have exactly these columns : ProductName,Price,Features,MatchScore.

    COLUMN REQUIREMENTS:
    - ProductName: Must be a string directed from the input.
    - Price: Must be a float from the input.
    - Features: Must be a single CSV-safe string enclosed in double quotes containing a comma-separated list of keywords (e.g.,"kids,cotton,red").
    - MatchScore: an integer from 0-100
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

def clean_llm_csv(text: str) -> str:
    # Remove markdown fences if present
    text = re.sub(r"^```(csv)?", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"```$", "", text.strip(), flags=re.MULTILINE)

    # Unescape \n to real newlines
    text = text.replace("\\n", "\n")

    # Strip wrapping quotes if the whole thing is enclosed in ""
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1]

    # Strip extra surrounding whitespace
    text = text.strip()

    return text

def compare_products(state):

    logging.info(f"comparison_reasoner script starts here...")
    logging.info(f"state.filtered_products is - {state.filtered_products}")
    logging.info(f"state.filtered_products in json is - {state.filtered_products.to_dict(orient='records')}")


    # products =  state.filtered_products.to_dict(orient='records')
    # preferences = state.preferences

    # rows = []
    # for product in products:
    #     features = product.get("must have features",[])
    #     if not isinstance(features,list):
    #         features = str(features).split(",")

    #     score = int(100*len(features) / max(1, len(preferences.get("must have features",[]))))

    #     row = f"\"{product['name']}\",{product['discount_price']},\"{','.join(features)}\",{score}"
    #     rows.append(row)


    # response_text = "ProductName,Price,Features,MatchScore\n" + "\n".join(rows)
    # logging.info(f"response text is - {response_text}")


    # 1. Format Input prompt
    input_text = prompt.format(

        products = state.filtered_products.to_dict(orient="records"),
        preferences = state.preferences
    )
    # 2. Inved_response_textoke LLM
    response = llm.invoke(input_text)
    # Extract actual string from AIMessage
    #response_text = response.content.strip()
    response_text = response.strip()
    print(f"Response text generated: {response_text}")
   
    try:
        # 3. Detect CSV header
        logging.info(f"CSV header detection starts...")

        if "," in response_text and re.search(r"(?i)(product name|price).*?,",response_text):
            print(f"Detected CSV format...")
            # regex to find the list-like string and replace it with a single ,quoted string
            # this makes the row a valid csv format 
            #example: replaces["kids","cotton"] with "[kids,cotton]"

            
            cleaned_response = response_text.strip().lstrip('"').rstrip('"')
            cleaned_response = cleaned_response.strip().strip('"')
            #cleaned_response = clean_llm_csv(response_text)
            

            if (cleaned_response.startswith('"') and cleaned_response.endswith('"')) or (cleaned_response.startswith("'")and cleaned_response.endswith("'")):
                cleaned_response = cleaned_response[1:-1]

            cleaned_response = response_text.replace("\\n","\n").strip()
            cleaned_response = re.sub(r'^\s*""\s*$','',cleaned_response,flags=re.MULTILINE)
            cleaned_response = re.sub(r'^\s*([^",][^,]+),',r'"\1"',cleaned_response,flags=re.MULTILINE)
            cleaned_response = re.sub(r'\n\s+','\n',cleaned_response)
            cleaned_response = re.sub(r'""([^"]+)"',r'"\1"',cleaned_response)
            cleaned_response = re.sub(r'\[(.*?)\]',lambda m: '"' + m.group(1).replace('"','')+ '"',cleaned_response)



           


            df = pd.read_csv(StringIO(cleaned_response),on_bad_lines="skip")
            print(f"parsed csv dataframe...")

            if 'Features' in df.columns:
                df['Features'] = df['Features'].apply(lambda x:[i.strip() for i in str(x).split(',')] if pd.notnull(x) else [])

            print(df)
            #df = df.sort_values(by='Match Score',ascending=False)
            df.to_csv('3.csv',index=False)
            
            state.compared_insights = df.to_dict(orient='records')
            
            logging.info(f"comparison_reasoner script ends here...")
            return state
        
        # 4. Fallback- try markdown table
        elif "|" in response_text:
            print(f"Detected markdown table...")
            df = parse_markdown_table(response_text)
            print(f"parsed markdown table DataFrame:")
            print(df)
            #df = df.sort_values(by='Match Score',ascending=False)
            df.to_csv('3.csv',index=False)
            
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
