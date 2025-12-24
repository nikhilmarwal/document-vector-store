import os
import faiss
import pickle
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document   #Document object have two components, page_content(str) and metadata(dictionary)
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np

class VectorService():
	""" Main class does all the work required for loading the text from documents , 
	chunking it , making embeddings, storing them , storing the index into disk,
	searching using query."""
	
	def __init__(self,data_path):
		""" initialises the model, makes the path for persistence storage,
		loads the data if exists already.
		
		Args:
			data_path(str): path to store persistent data files(index, metadata,content)
		"""
		
		self.index_path = os.path.join(data_path,'index.faiss')
		self.meta_path = os.path.join(data_path,'metadata.pkl')
		self.content_path = os.path.join(data_path,'content.pkl')
		os.makedirs(data_path,exist_ok=True)
		
		# initalising the model 
		print('Loading the Model')
		self.embeddings_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
		print("Model Loaded")
		self.embedding_dim = self.embeddings_model.get_sentence_embedding_dimension() # get the model embedding dimension used in storing the embeddings
		
		# initialising attributes to hold the data in memory
		self.index = None # index initialisiation will contain the vector index
		self.metadata = [] # metadata array will contain the metadata of each chunk
		self.content = [] # content array will contain the content i.e. text of each chunk
		self._load_data() #calling the load data method (helper method) will load the index, metadata,content from memory if already exists
	def _load_data(self):
		""" Load the index, metadata, content from memory if does already exists"""
		if os.path.exists(self.index_path):
			print("\nLoading FAISS index")
			self.index = faiss.read_index(self.index_path)
		else:
			print("\nNo FAISS index found intialising a new one")
			self._initialise_empty_index()
		if os.path.exists(self.meta_path):
			with open(self.meta_path, 'rb') as f_meta:
				self.metadata = pickle.load(f_meta)
		else:
			self.metadata = []
		if os.path.exists(self.content_path):
			with open(self.content_path,'rb') as f_content:
				self.content = pickle.load(f_content)
		else:
			self.content = []
		print("\nData loaded successfully")
			
				
	def _initialise_empty_index(self):
		"""Initialises a empty FAISS index using HNSW algorithm."""
		self.index = faiss.IndexHNSWFlat(self.embedding_dim,32,faiss.METRIC_INNER_PRODUCT) # METRIC_INNER_PRODCUT calculates the dot product and when normalised gives the cosine simlilarity
		
	def process_store_pdf(self, pdf_file_path: str, filename: str):
			""" Loads and process the pdf given by chunking it, embedding it and storing the embeddings into vector store
			ARGS:
				pdf_file_path(str): the path of the file path passed as string
				filename(str): name of the file being processed
			RETURNS:
				None: this method modifies the internal state of the service
			RAISES:
				ValueError: If the file is already processed or no text can be extracted from the given file.
			"""
			if any(meta.get('source')== filename for meta in self.metadata):
				raise ValueError(f"file: {filename} is already processed. Try another file")
				
			all_pages = []
			try:
				reader = PdfReader(os.path.join(pdf_file_path,filename))
				for i, page in enumerate(reader.pages):
					text= page.extract_text()   # extract the text from each page
					all_pages.append(Document(
						page_content=text,
						metadata={"source": filename, "page_number": i+1}))  # adds the page_content i.e. text and metadata i.e. the source and the page number to the all_pages
				if not all_pages:   #if no text is extracted i.e. all_pages list is empty
					raise ValueError(f"No text could be extracted from file: {filename}") 
			except Exception as e:
				raise RuntimeError(f"Error opening the pdf error:{e}")
			
			
			text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200)
			chunks = text_splitter.split_documents(all_pages) #split_documents is used rather than split_text, it preserves the document type of all_pages , so chunks have text and metadata both
			
			chunk_text = [chunk.page_content for chunk in chunks]  # contains text from each chunk
			chunk_metadata = [chunk.metadata for chunk in chunks]   # contains metadata from each chunk
			
			# making embeddings of each chunk 
			embeddings = self.embeddings_model.encode(chunk_text)  # encoding the chunks into embeddings
			embeddings_np = np.array(embeddings).astype('float32') # faiss only takes numpy array and float32 
			faiss.normalize_L2(embeddings_np) # this normalizes the embeddigns which is required for cosine similarity search
			
			# saving the embeddings into index
			self.index.add(embeddings_np)
			self.metadata.extend(chunk_metadata)  # appends each element of chunk_metadata to the end of metadata list
			self.content.extend(chunk_text) 
			
			self._save_data() # saves the data into disk
			print("File saved successfully")
			
	def _save_data(self):
		""" saves the index , content, metadata into the disk"""
		faiss.write_index(self.index, self.index_path)
		with open(self.content_path, 'wb') as f_content:
			pickle.dump(self.content, f_content)
		with open(self.meta_path, 'wb') as f_meta:
			pickle.dump(self.metadata, f_meta)				
		
	def search (self, query:str, k:int):
		""" Semantically searches the index using the query provided by the user
		Args:
			query(str): query given by the user in natural language (rewritten)
			K(int): Number of most relevant searches should be returned
		Returns:
			Dictionary of either results or error message
			"""
		if self.index.ntotal == 0:
			return {"error": "The index is empty please upload a document first"}
		
		# query into vectors
		embeddings_query = self.embeddings_model.encode(query)
		embeddings_query_np = np.array(([embeddings_query]), dtype='float32')
		faiss.normalize_L2(embeddings_query_np)
		
		# searching the faiss index
		distances, indices = self.index.search(embeddings_query_np, k) # returns the distances and the indices of the k most similar searches, returns a 2D matrix (for each query ,but we have only one query)
		# for cosine similarity the greater the distance => more the similarity
		
		# fetching the chunk from the indices
		results = []
		for i,idx in enumerate(indices[0]):
			if idx!=-1 and idx<len(self.metadata):
				result_metadata = self.metadata[idx].copy()   # copying because changing result won't change the original metadata
				result_metadata['content'] = self.content[idx]
				results.append({
					"metadata": result_metadata,
					"similarity": float(distances[0][i])
					})
					
					
		return results
	
	
		
		
		
		
		
			
				
			
										
