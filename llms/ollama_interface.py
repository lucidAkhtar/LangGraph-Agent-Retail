from dotenv import load_dotenv
import os
load_dotenv()

#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaLLM

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
#llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash-latest',temperature=0.2)
llm = OllamaLLM(model="mistral",temperature=0.2)

#embeddings = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001')
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("it worked")