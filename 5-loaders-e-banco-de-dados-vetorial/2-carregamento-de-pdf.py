from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader("/home/jsalles/MBA/Nivelamento/Langchain-01/5-loaders-e-banco-de-dados-vetorial/brochura-fullcycle-3.0.pdf")
data = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)

chunks = splitter.split_documents(data)
print(f"Total de chunks gerados: {len(chunks)}\n")