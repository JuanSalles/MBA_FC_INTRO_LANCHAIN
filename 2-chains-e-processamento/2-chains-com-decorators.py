from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import chain
from dotenv import load_dotenv
load_dotenv()

@chain
def square(input: dict) -> int:
    x = input["value"]
    return {"square_result": x * x}


model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.5)

question_template2 =  PromptTemplate(
    input_variables=["square_result"],
    template="Tell me about the number {square_result} in a funny way."
)

chain2 = square | question_template2 | model
# result = chain.invoke({"name": "Alice"})
result = chain2.invoke({"value": 5})

print(result.content)

