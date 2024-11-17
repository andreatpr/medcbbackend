import sys
sys.path.append(".")
from datetime import datetime
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage,ToolMessage,AIMessage
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from agents.agent_tools import RetrieverTool
from langgraph.graph import END, StateGraph, START
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
retriever_tool=RetrieverTool().create_retriever_tool(name="retrieve_clinic_data",description="searches and returns information about the doctors and times available at the clinic. Use only if the patient explicitly requests information about the times and doctors available")
class GrahpState(TypedDict):
    messages: Annotated[list, add_messages]

def prompt_guard(state: GrahpState)->Literal["auxiliary","invalid_task"]:
    """
    Determines if the user's prompt is valid for the chatbot's objective.

    Args:
        state (messages): The current state

    Returns:
        str: A decision on whether to continue interacting with the chatbot or not.
    """
    class score(BaseModel):
        """Binary score for supported propmts."""
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")
    
    guard_llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
    api_key="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",streaming=True)
    guard_llm_with_tool = guard_llm.with_structured_output(score)
    guard_prompt = PromptTemplate(
        template="""You are a gatekeeper assessing whether the user's entire interaction history is valid for a medical chatbot assistant. \n
        Here is the user's interaction history: \n\n {user_history} \n
        The chatbot's purpose is strictly limited to:
        1. Recommending medical specialists or departments based on described symptoms.
        2. Providing information about the clinic, including its schedules, doctors, and departments.
        3. Assisting users in scheduling appointments.

        If the interaction history aligns with one of these purposes, grade it as 'yes'. 
        If any part of the history requests a diagnosis, suggests automedicating, or is unrelated to these purposes, grade it as 'no'.

        Provide a binary score: 'yes' or 'no' to indicate whether the interaction history is valid for the chatbot's purpose.""",
        input_variables=["user_history"],
    )
    guard_chain = guard_prompt | guard_llm_with_tool

    user_messages = [
        message.content for message in state["messages"]
        if isinstance(message, HumanMessage)
    ]
    user_history = "\n".join(user_messages)
    scored_result = guard_chain.invoke({"user_history": user_history})
    scores = scored_result.binary_score

    if scores == "yes":
        print("---DECISION: PROPMT SUPORTED---")
        return "auxiliary"

    else:
        print("---DECISION:PROPMT NOT SUPORTED---")
        print(scores)
        return "invalid_task"


def should_need_rag(state: GrahpState)->Literal["rag_query","chatbot"]:
    """
    Determines if the user's prompt requires the RAG triever.

    Args:
        state (messages): The current state

    Returns:
        str: A decision on whether to continue directly with the chatbot or call the retriever.
    """
    class score(BaseModel):
        """Binary score for propmts that require the retriever."""
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")
    
    rag_guard_llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
    api_key="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",streaming=True)
    rag_guard_llm_with_tool = rag_guard_llm.with_structured_output(score)
    rag_guard_prompt = PromptTemplate(
     template="""You are a gatekeeper assessing whether a user's prompt requires retrieving specific information about the clinic using the RAG model. \n
        Here is the user's prompt: {user_prompt} \n
        The chatbot's purpose is strictly limited to:
        1. Recommending medical specialists or departments based on described symptoms.
        2. Providing information about the clinic, including its schedules, doctors, and departments.
        3. Assisting users in scheduling appointments.

        If the prompt specifically requests detailed or specific information about the clinic (e.g., clinic schedules, availability of doctors, departments, or other structured data), grade it as 'yes'.
        If the prompt can be answered based on general knowledge or does not require retrieving structured information from the clinic's database, grade it as 'no'.

        Provide a binary score: 'yes' or 'no' to indicate whether the prompt requires the RAG model to retrieve specific clinic information.""",
        input_variables=["user_prompt"],
    )
    rag_guard_chain = rag_guard_prompt | rag_guard_llm_with_tool

    messages = state["messages"]
    user_prompt = messages[-1].content
    scored_result=rag_guard_chain.invoke({"user_prompt": user_prompt})
    scores = scored_result.binary_score

    if scores == "yes":
        print("---DECISION: RAG NEEDED---")
        return "rag_query"

    else:
        print("---DECISION: RAG NOT NEEDED---")
        print(scores)
        return "chatbot"

def axuiliary_node(state: GrahpState):
    """
    An auxiliary node to be used in the graph.

    Args:
        state (messages): The current state

    Returns:
        list: The messages to be displayed in the chat.
    """
    return state

def rag_query_generator(state: GrahpState):
    """
    Generates a query for the RAG retriever.

    Args:
        state (messages): The current state

    Returns:
        str: The query to be used for the RAG retriever.
    """
    class query(BaseModel):
        """Query for the RAG retriever."""
        query: str = Field(description="Query for the RAG retriever")
    
    rag_query_llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
    api_key="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",streaming=True)
    rag_query_llm_with_tool = rag_query_llm.with_structured_output(query)
    rag_query_prompt = PromptTemplate(
    template="""Eres un asistente de chatbot encargado de generar una consulta semántica para el modelo RAG con el fin de recuperar información específica y estructurada sobre la clínica. \n
    Este es el mensaje del usuario: {user_prompt} \n
    Y aquí está el historial del chat del usuario con el chatbot para entender el contexto de la consulta: {messages} \n
    Los datos estructurados de la clínica incluyen:
    - Nombres de los doctores
    - Departamentos (por ejemplo, Cardiología, Neurología, etc.)
    - Descripciones de cada departamento y la especialización de los doctores
    - Horarios con días y horas específicas para cada doctor

    Basándote en el mensaje del usuario:
    1. Extrae los elementos clave de la solicitud del usuario, como el día, hora, departamento o síntomas relevantes que correspondan a un departamento o doctor específico.
    2. Traduce la intención del usuario en una consulta técnica que esté alineada con los datos estructurados, incluyendo campos relevantes como el nombre del doctor, departamento, descripción o horario.
    3. Optimiza la consulta para la búsqueda semántica incorporando sinónimos, términos relacionados o conexiones inferidas a partir de la entrada del usuario. 
    Por ejemplo:
    - Si el usuario pregunta: "¿Qué cardiólogos están disponibles el martes por la tarde?", traduce esto a una consulta como: 
        "Buscar doctores en Cardiología disponibles el martes entre las 12:00 PM y las 7:00 PM."
    - Si el usuario pregunta: "Necesito un especialista en enfermedades del corazón para el viernes por la mañana", traduce esto a:
        "Buscar doctores en Cardiología disponibles el viernes entre las 8:00 AM y las 12:00 PM."

    Genera una consulta concisa y estructurada que pueda ser utilizada directamente por el modelo RAG para la recuperación de información. 

    Proporciona únicamente la consulta como resultado.""",
    input_variables=["user_prompt", "messages"],
)
    rag_query_chain = rag_query_prompt | rag_query_llm_with_tool

    messages = state["messages"]
    user_prompt = messages[-1].content
    query_result=rag_query_chain.invoke({**state,"user_prompt": user_prompt})
    querygenerated = query_result.query
    print("---QUERY GENERATED---")
    print(querygenerated)
    return {"messages": [AIMessage(content=querygenerated)]}

def rag_retriever(state: GrahpState):
    """
    Retrieves information from the RAG model.

    Args:
        state (messages): The current state

    Returns:
        list: The messages to be displayed in the chat.
    """
    query = state["messages"][-1].content
    response = retriever_tool.invoke({"query": query})
    print("---RAG RETRIEVER RESPONSE---")
    return {"messages": [ToolMessage(content=response,tool_call_id="retrieve_clinic_data")]}


def invalid_task(state: GrahpState):
    """
    The function to be executed when the user's prompt is not valid for the chatbot's objective.

    Args:
        state (messages): The current state

    Returns:
        list: The messages to be displayed in the chat.
    """
    return {"messages": [AIMessage(content="Lo siento, pero no puedo ayudar con eso. Proporcione un mensaje que se alinee con el propósito del chatbot.")]}

def chat(state: GrahpState):
    """
    The chat function for the chatbot.

    Args:
        state (messages): The current state

    Returns:
        list: The messages to be displayed in the chat.
    """
    primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
            (
                "system",
                "You are a helpful medical chatbot assistant for a medical clinic. Your primary responsibilities are:"
                "1. Start interactions by asking users for detailed information about their symptoms and possible causes (e.g., when symptoms started, what might have triggered them)."
                "2. Based on the user's descriptions, suggest the most appropriate medical specialist or department they should visit."
                "3. Provide information about clinic schedules, available doctors, and departments if the user requests."
                "4. If the user gives explicit approval to schedule an appointment, confirm that the appointment has been successfully scheduled, wish them a speedy recovery, and politely end the interaction."
                "5. Avoid providing medical advice, diagnosing conditions, or recommending medication under any circumstances."
                "6. Use the full interaction history (including ToolMessages) to maintain context and ensure accurate responses."
                "7. Translate user-provided symptoms into medical terms or related specialties to align with the clinic's structured data (e.g., 'stomach pain' -> 'Gastroenterology')."
                "Ensure your responses are clear, concise, and professional."
                "\n\nCurrent time: {time}."
            ),
            (
                "assistant",
                "Hello! Welcome to our medical clinic's virtual assistant. How can I help you today? Please tell me about your symptoms, when they started, and anything you think might have caused them."
            ),
            ("placeholder", "{messages}"),
        ]
    ).partial(time=datetime.now)
    llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
    api_key="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")
    chain=primary_assistant_prompt | llm
    response = chain.invoke({"messages": state["messages"]})
    print("---CHATBOT RESPONSE---")
    return {"messages": [response]}

graph_builder = StateGraph(GrahpState)
graph_builder.add_node("chatbot",chat)
graph_builder.add_node("invalid_task",invalid_task)
graph_builder.add_node("rag_query",rag_query_generator)
graph_builder.add_node("rag_retriever",rag_retriever)
graph_builder.add_node("auxiliary",axuiliary_node)
graph_builder.add_conditional_edges(START, prompt_guard)
graph_builder.add_conditional_edges("auxiliary", should_need_rag)
graph_builder.add_edge("chatbot",END)
graph_builder.add_edge("invalid_task",END)
graph_builder.add_edge("rag_retriever","chatbot")
graph_builder.add_edge("rag_query","rag_retriever")
graph = graph_builder.compile(checkpointer=memory)
#config = {"configurable": {"thread_id": "1"}}

def get_graph():
    return graph


"""
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [HumanMessage(content=user_input)]},config):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)
while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except Exception as e:
        print("An error occurred:", e)
        break
"""