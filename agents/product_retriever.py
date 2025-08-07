import os
import pandas as pd
import time
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

#embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load CSV and convert to documents
def load_and_embed_products(csv_path,batch_size=10):

    df = pd.read_csv(csv_path)
    docs = [Document(page_content = row.to_json(),metadata={"index":i}) 
            for i,row in df.iterrows()]
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    chunked_docs = splitter.split_documents(docs)

    print(f"Total documents to embed after splitting is: {len(chunked_docs)}")
    
    if os.path.exists("index_data/faiss_index_v1"):
        print(f"Loading existing FAISS index...")
        vectorstore = FAISS.load_local(
            "index_data/faiss_index_v1",
            embeddings,
            allow_dangerous_deserialization=True
            )
        print(f"FAISS index loaded...")
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
            #time.sleep(sleep_time)

        if vectorstore:
            vectorstore.save_local("index_data/faiss_index_v1")
            print("FAISS index saved after creating...")
        else:
            print("No docs were processed to create FAISS index...")        

    return vectorstore,df
    

def retrieve_products(state,vectorstore,df,k=10):

    print(f"State user_input is:{state.user_input}")
    print(f"Original df shape is: {df.shape}")
    print(f"Original df columns is:{df.columns}")
    print(f"type of df is: {type(df)}")

    results = vectorstore.similarity_search(state.user_input,k = k)
    print(f"results from vectorstore is :{results}")

    indices = [doc.metadata["index"] for doc in results]
    print(f"Indices returned is: {indices}")

    print(f"Max index in df is: {df.index.max()}")
    print(f"are indices in range ? {all(i in df.index for i in indices)}")
    
    if not indices:
        print(f"No matching indices were found and/or available.")
        
    else:
        state.retrieved_products = df.iloc[indices].reset_index(drop=True)

    return state



