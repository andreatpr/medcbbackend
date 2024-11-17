import sys
sys.path.append("..")
from langchain.tools.retriever import create_retriever_tool
from retrievals.retriever import Retriever
from langchain_core.tools.simple import Tool
class RetrieverTool:
    def __init__(self):
        self.retriever = Retriever("./retrievals/data").get_retriever()
    def create_retriever_tool(self,name,description)->Tool:
        return create_retriever_tool(
            self.retriever,
            name,
            description,
        )