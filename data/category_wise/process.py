import os
import pandas as pd

def smart_sample(df, rating_col='ratings', top_pct=0.2, bottom_pct=0.05, total_limit=500):
    # Drop rows with missing or non-numeric ratings
    df[rating_col] = pd.to_numeric(df[rating_col], errors='coerce')
    df = df.dropna(subset=[rating_col])

    # Sort by rating
    df_sorted = df.sort_values(by=rating_col, ascending=False)

    # Calculate top and bottom sizes
    top_n = int(total_limit * top_pct)
    bottom_n = int(total_limit * bottom_pct)
    remaining_n = total_limit - (top_n + bottom_n)

    # Sample from top-rated, low-rated, and random mid-rated products
    top_samples = df_sorted.head(top_n)
    bottom_samples = df_sorted.tail(bottom_n)
    middle_pool = df_sorted.iloc[bottom_n:-top_n]
    middle_samples = middle_pool.sample(n=remaining_n, random_state=42)

    # Combine all
    final_df = pd.concat([top_samples, bottom_samples, middle_samples]).sample(frac=1).reset_index(drop=True).dropna()
    return final_df

# all_appliances_df = pd.read_csv("data/category_wise/all_appliances.csv")
# sample_all_appliances_df = smart_sample(df = all_appliances_df)
# print(sample_all_appliances_df)

input_folder = "data/category_wise"
output_folder = "data/filtered_category_wise"

os.makedirs(output_folder,exist_ok=True)

for file in os.listdir(input_folder):
    if file.endswith(".csv"):
        # input_folder should give location from root of the directory
        df = pd.read_csv(os.path.join(input_folder,file))
        processed_df = smart_sample(df)
        output_path = os.path.join(output_folder,file)
        processed_df.to_csv(output_path,index=False)