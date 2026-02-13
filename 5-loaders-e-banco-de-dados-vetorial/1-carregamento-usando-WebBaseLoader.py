from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = WebBaseLoader("https://www.langchain.com/")
data = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)

chunks = splitter.split_documents(data)

for chunk in chunks:
    print(chunk)
    print("-"*30)