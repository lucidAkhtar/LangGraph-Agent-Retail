
import re
import pandas as pd

def filter_products(state):

    prefs = state.preferences
    print("#############################")
    print(f"prefs type: {type(prefs)}, prefs_value: {prefs}")
    df = state.filtered_products

    if prefs.get("brand"):
        df = df[df["brand"].str.contains(prefs["brand"],case=False)]

    if prefs.get("budget"):
        # extract digits from 1000 rupees safely
        budget_str = prefs["budget"]
        budget_match = re.search(r"\d+",budget_str)
        if budget_match:
            budget = int(budget_match.group())
        
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

    return state

