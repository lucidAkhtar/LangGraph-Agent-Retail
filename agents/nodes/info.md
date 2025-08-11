- using similarity search, top K results are fetched , then chunked.
- the resultant df is served to retrieved_df and filtered_df

- The chunked_df should be passed to llm in comparison_reasoner to not get high token length issue.
- 



- Chunking is for long documents. It's designed for use cases where a single document, like an article or a book, is too long for an embedding model's token limit. The splitter breaks the document into smaller pieces.
- Your data is already a chunk. The combined string for each product ("apparel_womens: Flaxum V-Neck Button Cardigan...") is already a short, self-contained unit of information. It's highly unlikely to exceed the 512-token limit of the Cohere model.