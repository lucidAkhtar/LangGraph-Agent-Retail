import os
import pandas as pd
input_folder = "data/filtered_category_wise"
output_file = "data/filtered_category_wise/all_data_v1.csv"

def combine_csv_with_category(input_folder,output_file):  
    all_dfs = []

    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            filepath = os.path.join(input_folder,filename)
            df = pd.read_csv(filepath)
            category = os.path.splitext(filename)[0]
            df['category'] = category
            all_dfs.append(df)


    combined_df = pd.concat(all_dfs,ignore_index=True)
    combined_df.to_csv(output_file,index=False)

    print(f"Combined CSV saved to: {output_file}")

combine_csv_with_category(input_folder,output_file)