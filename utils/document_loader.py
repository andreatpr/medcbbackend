from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
class DocumentLoader:
    def __init__(self):
        self.spliter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=100, chunk_overlap=50)
    def split_documents(self, documents):
        return self.spliter.split_documents(documents)
    def load_documents(self, directory):
        documents = []
        for archivo in os.listdir(directory):
            if archivo.endswith(".pdf"):
                ruta_archivo = os.path.join(directory, archivo)
                loader = PyPDFLoader(ruta_archivo)
                documents.append(loader.load())
        return documents
    def get_splited_documents(self,directory):
        documents=self.load_documents(directory)
        documents_list = [item for sublist in documents for item in sublist]
        return self.split_documents(documents_list)