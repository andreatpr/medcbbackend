import sys
sys.path.append(".")
from agents.graph_flow import get_graph
from langchain_core.messages import HumanMessage
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
graph=get_graph()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

class UserInput(BaseModel):
    user_propmt: str
    thread_id: str

@app.post("/chat")
def chat_with_bot(user_input: UserInput):
    """
    API endpoint para interactuar con el chatbot.

    Args:
        user_input (UserInput): Entrada del usuario y ID del hilo.

    Returns:
        dict: Respuesta del chatbot.
    """
    try:
        config = {"configurable": {"thread_id":user_input.thread_id}}
        result=graph.invoke({"messages": [HumanMessage(content=user_input.user_propmt)]},config)
        return {"response": result['messages'][-1].content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/")
def read_root():
    return {"Hello": "World"}
