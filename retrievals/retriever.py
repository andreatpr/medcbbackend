import sys
sys.path.append(".")
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from utils.document_loader import DocumentLoader

model_name = "nomic-ai/nomic-embed-text-v1"
model_kwargs = {
    'device': 'cpu',
    'trust_remote_code':True
    }
encode_kwargs = {'normalize_embeddings': True}
class Retriever():
    def __init__(self,directory)->None:
        self.embedding = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            query_instruction = "search_query:",
            embed_instruction = "search_document:"
        )
        self.documents = DocumentLoader().get_splited_documents(directory)
    def add_document(self, document):
        self.vector_store.add_document(document)
    def search(self, query):
        return self.vector_store.search(query)
    def get_retriever(self)->Chroma:
        return Chroma.from_documents(
            documents=self.documents,
            collection_name="clinic_db",
            embedding=self.embedding
        ).as_retriever()


        