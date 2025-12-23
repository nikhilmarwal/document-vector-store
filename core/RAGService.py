class RAGService:
    def __init__(self, vector_service, context_service, llm):
        self.vector_service = vector_service
        self.context_service = context_service
        self.llm = llm

    def answer(self, query: str):
        """
        RAG main function generates response by orchestrating
        vector_service, context_service, and the main LLM.
        """

        # Rewrite query
        rewritten_query = self.context_service.build_query(query)
        print(f"User query rewritten as: {rewritten_query}")

        # Retrieve chunks
        retrieved_chunks = self.vector_service.search(rewritten_query, k=3)

        for chunk in retrieved_chunks:
            print(f"Retrieved chunk:\n\n{chunk['metadata']['content']}\n")

        # Rerank chunks
        num_chunks = len(retrieved_chunks)
        reranked_chunks = self.context_service.rerank(
            rewritten_query,
            retrieved_chunks,
            num_chunks
        )

        # Consolidate / compress context
        context = self.context_service.chunk_compressor(
            reranked_chunks,
            rewritten_query
        )

        # Final response generation
        prompt = f"""
Use the context below to answer the user query.

Context:
{context}

Question:
{rewritten_query}
"""

        return self.llm.invoke(prompt)

