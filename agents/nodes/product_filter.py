
import re
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)
k = 10

def filter_products(state):

    logging.info(f"filter_products script starts here...")
    prefs = state.preferences

    print(f"prefs type: {type(prefs)}, prefs_value: {prefs}")
    df = state.filtered_products

    logging.info(f"Columns present in df are :-{df.columns}")
    logging.info(f"Shape of df is :-{df.shape}")

    if prefs.get("brand"):

        logging.info(f"prefs.get(brand) is TRUE...")
        df1 = df[df["name"].str.contains(prefs["brand"],case=False)]
        logging.info(f"df in brand is: {df1}")
    else:
        logging.info(f"prefs.get(brand) is FALSE...")
        df1 = None

    if prefs.get("budget"):
        # extract digits from amount mentioned in the user_input
        logging.info(f"prefs.get(budget) is TRUE...")
        budget_input = prefs["budget"]

        if isinstance(budget_input,int):
            budget = budget_input

        if isinstance(budget_input,dict):
            budget = str(budget_input.get("budget","")).strip()
        else:
            budget = str(budget_input).strip()
    
        # Extract first integer found
        match = re.search(r'\d+',budget)
        if match:
            budget = int(match.group()) #if match else None
        else:
            raise ValueError("Matching group related to budget is None...")


        if 'discount_price' not in df.columns:
            raise ValueError("Required column missing in dataframe...")
        
        #df["discount_price"] = df["discount_price"].str.replace("₹","",regex=False).str.replace(",","").str.strip()
        #print(df["discount_price"])
        #df["discount_price"] = pd.to_numeric(df["discount_price"],errors="coerce")
        logging.info(f"budget is: {budget}")

        df2 = df[df["discount_price"] <= budget]
        
        print(f"df in discount-price code is:{df2}")

    else:
        logging.info(f"prefs.get(budget) is FALSE...")
        df2 = None
    
    # key name should match
    if prefs.get("must have features"):
        logging.info(f"prefs.get(must have features) is TRUE...")

        pattern = "|".join(re.escape(feat) for feat in prefs["must have features"])
        logging.info(f"pattern found in must have features is: {pattern}")

        df3 = df[df["name"].str.contains(pattern,case=False)]
        print(f"df in must-have-features is :{df3}")

    else:
        logging.info(f"prefs.get(must have features) is FALSE...")
        df3 = None
    

    df_to_concat = [df1,df2,df3]
    filtered_dfs = [df for df in df_to_concat if df is not None]
    df_filtered = pd.concat(filtered_dfs,ignore_index=True)
    df_filtered = df_filtered.head(k)

    logging.info(f"filtered df is: {df_filtered.reset_index(drop=True)}")
    df_filtered.to_csv('2.csv',index=False)
    state.filtered_products = df_filtered.reset_index(drop=True)
    
    logging.info(f"filter_products script ends here...")

    return state

