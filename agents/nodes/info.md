- using similarity search, top K results are fetched , then chunked.
- the resultant df is served to retrieved_df and filtered_df

- The chunked_df should be passed to llm in comparison_reasoner to not get high token length issue.
- 