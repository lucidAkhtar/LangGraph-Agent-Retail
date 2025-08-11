import cohere
import pandas as pd
import numpy as np
import os
import time
import logging
from dotenv import load_dotenv

load_dotenv()

def generate_and_save_embeddings():

    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    if not COHERE_API_KEY:
        raise ValueError(f"COHERE API KEY environment variable not set...")
    
    co = cohere.Client(COHERE_API_KEY)

    df = pd.read_csv("data/filtered_category_wise/data_for_embedding.csv")
    texts_to_embed = df['text_for_embedding'].astype(str).tolist()

    batch_size = 96 # as per free tier
    trial_key_rate_limit = 100 # requests per minute for free tier
    seconds_per_request = 60 / trial_key_rate_limit
    safety_delay = seconds_per_request + 0.1


    all_embeddings = []

    for i in range(0,len(texts_to_embed),batch_size):

        batch = texts_to_embed[i:i+batch_size]
        try:
            response = co.embed(
                    texts = batch,
                    model = "embed-english-v3.0",
                    input_type = "search_document"
            )

            all_embeddings.extend(response.embeddings)

            print(f"Batch {i // batch_size+1} of {len(texts_to_embed)// batch_size+1} processed...")

        except cohere.core.api_error.ApiError as e:
            print(f"an api error occured:{e}")
            break

        time.sleep(safety_delay)

    embeddings_array = np.array(all_embeddings)

    os.makedirs("embeddings",exist_ok=True)

    np.save("embeddings/data_embeddings.npy",embeddings_array)
    logging.info(f"embeddings saved successfully to embeddings/data_embeddings.npy...Total embeddings: {embeddings_array.shape}")  
    logging.info(f"Generating and saving embeddings script end...")

    return embeddings_array,texts_to_embed

generate_and_save_embeddings()