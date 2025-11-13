from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

template_translate = PromptTemplate(
    input_variables=["text"],
    template="Translate the following text to English:\n ```{text}```",
)

template_summarize = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text in 4 words:\n ```{text}```\n\n",
)

llm_en = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)

translate = template_translate | llm_en | StrOutputParser()
pipeline = {"text": translate} | template_summarize | llm_en | StrOutputParser()

result = pipeline.invoke({
    "text": "LangChain é uma estrutura poderosa para construir aplicações com modelos de linguagem."
})

print(result)