from langchain_core.documents import Document
class ContextService():
	def __init__(self,compressor,rewriter_model,reRanker_client):
		""" Args:
			chunks: list of reranked chunks returned be vectorservice search.
			query: query is the user query
		"""
		self.compressor = compressor
		self.reRanker_client = reRanker_client 
		self.rewriter_model = rewriter_model
		
	def build_query(self,query):
		""" builds the query returned by vector service
			Returns: str -> rewritten query which is used by vector service"""
		rewritten_query = self.rewriter_model.invoke(query)
		print(f"rewritten_query is {rewritten_query} ")
		return rewritten_query
		
	def reRanker(self,rewritten_query:str, chunks_list: list,k:int):
		""" Receives the chunks_list retrieved by from database and reranks them on the basis of relevence using api call from cohere.
		Args: 
			list-> list of retrived chunks as langchain document from the vector database
			rewritten_query:str: rewritten_ query from the rewritter
			k: number of top we need 
		Returns:
			list -> list of reranked chunks with their ids.
		"""
		# chunks_list contanins the text in metadata['content']
		docs = [item['metadata']['content'] for item in chunks_list]
		response = self.reRanker_client.rerank(
			model='rerank-v3.5',
			query=rewritten_query,
			documents=docs,
			top_n=k)
			
		indices = [item.index for item in response.results]   # contains the indices in order returned by reranker
		results = []
		for index in indices:
			results.append(chunks_list[index])  # rearranging the chunks
		return results
			
	def chunk_compressor(self,chunks,query):
		""" takes the chunk and query and compress it buy removing umwanted and irrelevant chunks
			Args:  
				list: list of chunks
				str: query (rewritten)
			Returns:
				str: concatenated string from compressed chunks
		"""
		""" the content should never be list it must always be list we need to hard normalise it before passing it to LLMLinguaCompressor"""
		docs = []
		for chunk in chunks:
			content = chunk["metadata"]["content"]
			
			# if the content is list -> convert to string 
			if isinstance(content,list):
				content=" ".join(content)
			elif not isinstance(content,str):
				content=str(content)
			docs.append(
				Document(
					page_content=content,
					metadata=chunk["metadata"]
				)
			)
		compressed_docs = self.compressor.compress_documents(
			documents=docs,
			query=query
			)
		return "\n\n".join(d.page_content for d in compressed_docs)
		
		
