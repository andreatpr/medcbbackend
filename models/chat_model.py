from langchain_groq import ChatGroq



class ChatGroq:
    def __init__(self, model: str, temperature: float, max_retries: int, api_key: str):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.api_key = api_key
    





llm = ChatGroq(
    model="llama-3.2-3b-preview",
    temperature=0.0,
    api_key="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"
)