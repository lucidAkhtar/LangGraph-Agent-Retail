import streamlit as st
import pandas as pd
import requests

#1. Function to fetch recommendations from your FastAPI backend
def fetch_recommendations(user_input:str,api_url:str)-> pd.DataFrame:

    payload = {"user_input":user_input}
    response = requests.post(api_url,json=payload)
    response.raise_for_status() # Fail fast on HTTP errors
    data = response.json()

    # Extract list of recommendations
    recs = data.get("recommendations",[])

    # Defensive: Check if list and each item is dict
    if not isinstance(recs,list) or not all(isinstance(r,dict) for r in recs):
        st.error("Invalid data format received from API")
        return pd.DataFrame()
    
    # Convert to dataframe
    return pd.DataFrame(recs)

#2. Streamlit app UI
def main():

    st.title("Product Recommendations using LangGraph")

    # session state for  Q&A history
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []

    # Text Input for the current question
    user_input = st.text_input(
        "Enter your query:",
        key = f"input_{len(st.session_state.qa_history)}"
    )

    # Button for current query
    if st.button("Get Recommendations",key=f"btn_{len(st.session_state.qa_history)}"):
        if user_input.strip():

            with st.spinner("Fetching Recommendations..."):
                api_url = "http://127.0.0.1:8000/recommend"
                df = fetch_recommendations(user_input,api_url)

                if df.empty:
                    st.warning("No recommendations found...")
                else:
                    #Save query and result to history
                    st.session_state.qa_history.append((user_input,df))
                    # Refresh UI to show history and new input
                    st.experimental_rerun()
                    
    
    # Display all Q&A pairs
    for idx,(q,df) in enumerate(reversed(st.session_state.qa_history),1):
        st.subheader(f"Q{len(st.session_state.qa_history)- idx + 1}:{q}")
        st.dataframe(df)
        st.markdown("---")


if __name__ == "__main__":
    main()

