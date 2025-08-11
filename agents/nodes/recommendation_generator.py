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
    """
    You are a recommendation engine that must compare products strictly using the provided information.
    INPUT DATA:
    - user preferences: {preferences}
    - Product Insights: {insights}

    TASK:
    - If there are no products in the insights,state clearly, that no products were found that matched the search criteria.
    - You do not need to mention `Here is an example of how the table would look if there were matching products:`. Just mention - NO products as Recommendations available.
    - Otherwise,generate a **markdown table** with the following columns, in this exact order:
    1. Product Name
    2. Brand
    3. Match Score (integer between 1 and 100, no %/ symbol)
    4. Justification (one short sentence explaining why the products matches the preferences)

    REQUIREMENTS:
    - only use product names and brands explicitly present in the provided insights.If a field is missin, write N/A
    - **Do NOT invent,add or assume** product names,brands or other details.
    - Match Score should be based on alignment between users preferences and product insights.If uncertain, score conservatively.
    - Justification must be short, factual,and suitable for a dataframe cell (avoid line breaks, avoid extra commentary).
    - The table must be a **valid Markdown** and contain no text before or after it.
    - Do not output explanations,notes, or considerations outside the table.
    
    OUTPUT FORMAT:

    |Product Name |Brand | Match Score | Justification |
    |-------------|----- |-------------|---------------|
    | ...         | ...  | ...         | ...           |

    """
)

def parse_markdown_table(markdown_text:str)-> pd.DataFrame:

    """
    Parses a markdown table string into a pandas DataFrame.
    Cleans markdown formatting for CSV compatibility.
    """
    if not markdown_text:
        return pd.DataFrame()
    
    # extract the actual table content only (removes everything before the table header)
    lines = markdown_text.strip().splitlines()
    table_lines = []

    # flag to indicate if we have found the table header
    in_table = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Look for a line that contains at least one pipe and is not a separator line
        if '|' in line and not set(line.replace(' ', '')) <= set('-|:'):
            # This is likely a header or a data row
            in_table = True
            table_lines.append(line)
        
        elif in_table and set(line.replace(' ', '')) <= set('-|:'):
            # This is the separator line, skip it but stay in table mode
            continue
        
        elif in_table:
            # This is a data row
            table_lines.append(line)
            
        else:
            # Skip any text before the table
            continue

    if not table_lines:
        logging.warning("No markdown table found in the LLM response.")
        return pd.DataFrame()
    
    cleaned_table_string = "\n".join(table_lines)

     # Remove markdown separator line like ---|---
    # lines = [line for line in lines if not set(line.strip()) <= set('-|: ')]
    # cleaned = "\n".join(lines)
    try:
        df = pd.read_csv(StringIO(cleaned_table_string),sep="|",engine="python",skipinitialspace=True)
        # drop any columns that start with "Unnamed" - common read_csv issue
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        # remove empty columns from leading/trailing pipes
        df = df.dropna(axis=1,how="all")
        df.columns = [col.strip() for col in df.columns]
        df = df.applymap(lambda x:x.strip() if isinstance(x,str)else x)
        if len(df.columns) != 4:
            logging.error(f"Expected 4 columns, but found {len(df.columns)}.")
            return pd.DataFrame()
        
        return df
    
    except Exception as e:
        print(f"failed to parse markdown table...{e}")
        return pd.DataFrame()

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

        # Store API-safe json
        json_data = df.to_dict(orient='records')
        logging.info(f"shape of df is {df.shape}")
        logging.info(f"columns in df is {df.columns} ")

        print(f"JSON data is - {json_data}")

        clean_data = []
        for rec in json_data:

            if rec.get("Match Score") == "N/A":
                rec['Match Score'] = 0

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
        state.recommendations = f"Failed to extract recommendations from LLM response..."
        return state