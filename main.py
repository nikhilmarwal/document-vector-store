import sys
import os
from dotenv import load_dotenv
from langchain_community.document_compressors import LLMLinguaCompressor
from core.VectorService import VectorService
from core.ContextService import ContextService
from core.RAGService import RAGService
from utils import SYSTEM_PROMPT, READER_PROMPT
from models.llm import GeminiModel
import cohere
# loading dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
rewriter_model = GeminiModel(
			model_name="gemini-2.5-pro",
			system_prompt=SYSTEM_PROMPT		
		)
reader_model = GeminiModel(
			model_name="gemini-2.5-pro",
			system_prompt=READER_PROMPT
		)
reRanker_client = cohere.ClientV2()
compressor = LLMLinguaCompressor(
		model_name="gpt2",
		device_map="cpu")
		
data_path = "./data_store"
def main():
	vector_service = VectorService(data_path)  #makes a directory named data_store
	context_service = ContextService(
			compressor=compressor,
			rewriter_model=rewriter_model,
			reRanker_client=reRanker_client)
			
	rag_service = RAGService(
			vector_service=vector_service,
			context_service=context_service,
			llm=reader_model
			)
	while True:			
		print("\nSelect an Option:")
		print("\n1) Ingest Document(Build Embeddings)")
		print("\n2) Ask question")
		
		choice = input("\nEnter choice 1 or 2\n").strip()
		
		if choice== "1":
			filepath = input("\nEnter filepath: ").strip()
			filename = input("\nEnter filename: ").strip()
			
			try:
				vector_service.process_store_pdf(pdf_file_path=filepath,filename=filename)
			except Exception as e:
				print(f"\nError during ingestion :{e}")
		elif choice == "2":
			query = input("\nEnter your query").strip()
			
			try:
				response=rag_service.answer(query)
				print(response)
			except Exception as e:
				print(f"\nThere was problem answering your query: {e}")
		else:
			print("\nInvalid  choice")
			sys.exit(1)
		
if __name__ == "__main__":
	main()			
