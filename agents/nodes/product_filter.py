
import re
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)

def filter_products(state):

    logging.info(f"filter_products script starts here...")
    prefs = state.preferences

    print(f"prefs type: {type(prefs)}, prefs_value: {prefs}")
    df = state.filtered_products

    logging.info(f"Columns present in df are :-{df.columns}")
    logging.info(f"Shape of df is :-{df.shape}")

    if prefs.get("brand"):
        df = df[df["brand"].str.contains(prefs["brand"],case=False)]

    if prefs.get("budget"):
        # extract digits from amount mentioned in the user_input
        
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
            raise ValueError("Required columns missing in dataframe...")
        
        df["discount_price"] = df["discount_price"].str.replace("₹","",regex=False).str.replace(",","").str.strip()
        print(df["discount_price"])
        df["discount_price"] = pd.to_numeric(df["discount_price"],errors="coerce")

        df = df[df["discount_price"] <= budget]
        
        print(f"df in discount-price code is:{df.head()}")

    # key name should match
    if prefs.get("must have features"):
        pattern = "|".join(re.escape(feat) for feat in prefs["must have features"])
        df = df[df["name"].str.contains(pattern,case=False)]

    print(f"df in must-have-features is :{df.head()}")
    state.filtered_products = df.reset_index(drop=True)
    
    logging.info(f"filter_products script ends here...")

    return state

