from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory, RunnableLambda
from langchain_core.messages import trim_messages


load_dotenv()

def get_text_from_response(response):
    """Extrai o texto da resposta, independente do formato"""
    if isinstance(response.content, str):
        return response.content
    elif isinstance(response.content, list) and len(response.content) > 0:
        return response.content[0].get('text', str(response.content))
    else:
        return str(response.content)

chat_model = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.9)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer the user's questions based on the provided chat history."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

def prepare_inputs(payload: dict) -> dict:
    raw_history = payload.get("messages", [])
    
    trimmed = trim_messages(
        raw_history, 
        token_counter=len,
        max_tokens=4,
        strategy="last",
        start_on="human",
        include_system=True,
        allow_partial=False
    )

    return {
        "input": payload.get("input", ""),
        "history": trimmed
    }

prepare = RunnableLambda(prepare_inputs)

chain = prepare | prompt | chat_model

session_store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

conversational_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_message_key="input",
    history_messages_key="messages"
)

config = {
    "configurable":{
        "session_id": "demo-session"
    }
}

response1 = conversational_chain.invoke(
    {"input":"Hello, my name is Akira. Dont include my name in your answer except when necessary."},
    config=config
)

print("Assistant:", get_text_from_response(response1))
print("-"*30)

response2 = conversational_chain.invoke(
    {"input":"Tell me a fun fact about space. Dont include my name in your answer except when necessary."},
    config=config
)

print("Assistant:", get_text_from_response(response2))
print("-"*30)

response3 = conversational_chain.invoke(
    {"input":"Can you tell me a fun fact about my name?"},
    config=config
)

print("Assistant:", get_text_from_response(response3))
print("-"*30)