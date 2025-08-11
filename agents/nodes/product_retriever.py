import os
import pandas as pd
import numpy as np
import re,json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import CohereEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from typing import Tuple,List,Optional
# from langchain.schema import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# #from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import logging
logging.basicConfig(level=logging.INFO)
from dotenv import load_dotenv
load_dotenv()

#embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
embeddings = CohereEmbeddings(
    cohere_api_key = os.getenv("COHERE_API_KEY"),
    user_agent = "langgraph-cohere-agent-recommendation",
    model = "embed-english-v3.0"
    )

#embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def generate_documents(csv_path):
    df = pd.read_csv(csv_path)
    df['text_for_embedding'] = df['main_category']+":"+df['name']

    docs = [
        Document(
            page_content = row['text_for_embedding'],
            metadata = {"index":i}
        )
        for i,row in df.iterrows()
    ]

    return docs

# Load CSV and convert to documents
def load_and_embed_products(embeddings_path="embeddings/data_embeddings.npy",texts_to_embed_path = "embeddings/texts_to_embed.json", embeddings_model = embeddings,docs_path="embeddings/docs.json"):

    logging.info(f"load_and_embed_products script starts here...")
   
    """
    df = pd.read_csv(csv_path)
    docs = [Document(page_content = row.to_json(),metadata={"index":i}) 
            for i,row in df.iterrows()]
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    chunked_docs = splitter.split_documents(docs)

    print(f"Total documents to embed after splitting is: {len(chunked_docs)}")
    """
    index_folder_path = "embeddings/faiss_index"
    
    if not os.path.exists(index_folder_path):

        logging.info(f"faiss embeddings do not exist hence creating...")

        os.makedirs(index_folder_path,exist_ok=True)
        if not os.path.exists(embeddings_path):
            logging.error(f"embeddings file not found at {embeddings_path}")
            return None

        logging.info(f"Loading pre-computed embeddings...")
        embeddings_array = np.load(embeddings_path)
        logging.info(f"embeddings loaded with shape: {embeddings_array.shape}")

        with open(docs_path,'r')as f:
            docs_data = json.load(f)

        docs = [Document(**doc_dict) for doc_dict in docs_data]

        if len(docs) != len(embeddings_array):
            logging.error(f"Mismatch between number of documents and embeddings.")
            return None
         

        # vectorstore = FAISS.from_embeddings(
        #     "index_data/faiss_index_v1",
        #     embeddings,
        #     allow_dangerous_deserialization=True
        #     )

        # with open(texts_to_embed_path,'r') as f:
        #     texts_to_embed = json.load(f)
        # logging.info(f"Loaded texts_to_embed.json successfully...")

        # Build the faiss index from the loaded data
        vectorstore = FAISS.from_documents(
                documents = docs,
                embedding = embeddings_model

        )
        logging.info(f"FAISS index created successfully...")

        #3. save the index to a local folder for future use
        
        vectorstore.save_local(index_folder_path) 

        logging.info(f"faiss index saved to local folder path...")
        logging.info(f"load_and_embed_products script ends here...")

    else:

        logging.info("FAISS index already exists in local. Loading...")
        vectorstore = FAISS.load_local(
                    index_folder_path,
                    embeddings_model,
                    allow_dangerous_deserialization=True

        )
        logging.info(f"FAISS index from local path loaded successfully...")


    # else:
    #     print(f"Creating new FAISS index...")
    #     vectorstore = None
    #     for i in range(0, len(chunked_docs),batch_size):
    #         batch = chunked_docs[i:i+batch_size]
    #         print(f"Embedding batch {i // batch_size+1}/{len(chunked_docs)//batch_size+1}({len(batch)} documents...)")

    #         if vectorstore is None:
    #             vectorstore = FAISS.from_documents(batch,embeddings)

    #         else:
    #             vectorstore.add_documents(batch)
            
    #         print("few batched embedded well...")

    #     if vectorstore:
    #         vectorstore.save_local("index_data/faiss_index_v1")
    #         print("FAISS index saved after creating...")
    #     else:
    #         print("No docs were processed to create FAISS index...")        

    #return vectorstore,df
    return vectorstore
    

def retrieve_products(state,vectorstore,df,k=5):

    logging.info(f"retrieve_products script starts here...")

    user_query = state.user_input
    logging.info(f"State user_input is:{user_query}")
    logging.info(f"Original df shape is: {df.shape}")
    logging.info(f"Original df columns is:{df.columns}")

    #1. First,get semantic matches from FAISS vector datastore
    results = vectorstore.similarity_search(state.user_input,k = k*4)
    indices = [doc.metadata["index"] for doc in results]

    # Initialize retrieved_df as an empty dataframe in case no matches are found
    retrieved_df = pd.DataFrame(columns=df.columns)

    if not indices:
        logging.info(f"No matching indices were found and/or available.")
    else:
        # if indices are found, populate the dataframe
        retrieved_df = df.iloc[indices].reset_index(drop=True)
        logging.info(f"Retrieved df by similarity search is: {retrieved_df.shape}")
        logging.info(f"Retrieved df by similarity search is: {retrieved_df.head()}")
        #logging.info(f"Retrieved df is saved to : {retrieved_df.to_csv("df.csv",index=False)}")

    """
    #2. Try to detect category from user_query (scalable keyword mapping)
    
    categories = [re.sub(r'[\s_]+',' ',cat.strip().lower()) for cat in df['category'].dropna().unique()]
    user_query_clean = re.sub(r"[\s_]+",' ',user_query.strip().lower())

    matched_categories = [original_cat for cat_clean,original_cat in zip(categories,df['category'].dropna().unique()) if cat_clean in user_query_clean]

    if matched_categories:
        logging.info(f"Detected categories are: {matched_categories}")
    else:
        logging.info("No categories got matched...")

    #3. if we found a matched category in the user_query,filter results by it

    if matched_categories:
        categories_lower = [cat.lower() for cat in matched_categories]
        filtered_df = retrieved_df[retrieved_df['category'].str.lower().isin(categories_lower)]
    
        if not filtered_df.empty:
            logging.info(f"filtered results to categories - {', '.join(matched_categories)}")
            retrieved_df = filtered_df
    """


    #4. If category filter leaves nothing,just keep semantic matches
    retrieved_df = retrieved_df.head(k).reset_index(drop=True)
    
    logging.info(retrieved_df.head())
    #5. Store in state
    state.retrieved_products = retrieved_df
    state.filtered_products = retrieved_df

    logging.info(f"retrieve_products script ends here...")

    return state


# ollama serve

if __name__ == "__main__":

    docs = generate_documents("data/filtered_category_wise/data_for_embedding.csv")

    with open("embeddings/docs.json","w") as f:
        json.dump([doc.dict() for doc in docs],f)

    load_and_embed_products()

