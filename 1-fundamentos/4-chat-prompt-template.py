from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

system = ("system", "Você é um assistente que irá responder perguntas no estilo {style}")
user = ("user", "{question}")
chat_prompt = ChatPromptTemplate([system, user])

messages = chat_prompt.format_messages(
    style="engraçado",
    question="Quem é Alan Turing?"
)

for message in messages:
    print(f"{message.type}: {message.content}")

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)
response = model.invoke(messages)

print("Resposta do modelo:")
print(response.content)

