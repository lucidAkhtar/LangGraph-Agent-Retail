import os
import pandas as pd
import re
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import logging
logging.basicConfig(level=logging.INFO)

#embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

#1. Load embeddings globally once
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load CSV and convert to documents
def load_and_embed_products(csv_path,batch_size=10):
    logging.info(f"load_and_embed_products script starts here...")
   

    df = pd.read_csv(csv_path)
    docs = [Document(page_content = row.to_json(),metadata={"index":i}) 
            for i,row in df.iterrows()]
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    chunked_docs = splitter.split_documents(docs)

    print(f"Total documents to embed after splitting is: {len(chunked_docs)}")
    
    if os.path.exists("index_data/faiss_index_v1"):
        logging.info(f"Loading existing FAISS index...")
        vectorstore = FAISS.load_local(
            "index_data/faiss_index_v1",
            embeddings,
            allow_dangerous_deserialization=True
            )
        logging.info(f"FAISS index loaded...")
        logging.info(f"load_and_embed_products script ends here...")

    else:
        print(f"Creating new FAISS index...")
        vectorstore = None
        for i in range(0, len(chunked_docs),batch_size):
            batch = chunked_docs[i:i+batch_size]
            print(f"Embedding batch {i // batch_size+1}/{len(chunked_docs)//batch_size+1}({len(batch)} documents...)")

            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch,embeddings)

            else:
                vectorstore.add_documents(batch)
            
            print("few batched embedded well...")

        if vectorstore:
            vectorstore.save_local("index_data/faiss_index_v1")
            print("FAISS index saved after creating...")
        else:
            print("No docs were processed to create FAISS index...")        

    return vectorstore,df
    

def retrieve_products(state,vectorstore,df,k=5):

    logging.info(f"retrieve_products script starts here...")

    user_query = state.user_input
    logging.info(f"State user_input is:{user_query}")
    logging.info(f"Original df shape is: {df.shape}")
    logging.info(f"Original df columns is:{df.columns}")

    #1. First,get semantic matches from FAISS vector datastore
    results = vectorstore.similarity_search(state.user_input,k = k*4)
    indices = [doc.metadata["index"] for doc in results]

    if not indices:
        logging.info(f"No matching indices were found and/or available.")

    retrieved_df = df.iloc[indices].reset_index(drop=True)
    logging.info(f"Retrieved df by similarity search is: {retrieved_df.shape}")
    logging.info(f"Retrieved df by similarity search is: {retrieved_df.head()}")
    logging.info(f"Retrieved df is saved to : {retrieved_df.to_csv("df.csv",index=False)}")
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

    #4. If category filter leaves nothing,just keep semantic matches
    retrieved_df = retrieved_df.head(k).reset_index(drop=True)
    
    logging.info(retrieved_df.head())
    #5. Store in state
    state.retrieved_products = retrieved_df
    state.filtered_products = retrieved_df

    logging.info(f"retrieve_products script ends here...")

    return state


# ollama serve


