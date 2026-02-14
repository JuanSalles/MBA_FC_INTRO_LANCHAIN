import os
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()
for k in ("PGVECTOR_URL", "PGVECTOR_COLLECTION", "GOOGLE_API_KEY", "GEMINI_EMBEDDING_MODEL"):
    if not os.getenv(k):
        raise RuntimeError(f"Env variable {k} is not set")
    
query = "Qual o principal foco do curso Full Cycle 3.0 pensando em pessoas que querem se tornar desenvolvedores Fullstack?"

embeddings = GoogleGenerativeAIEmbeddings(model=f"models/{os.getenv('GEMINI_EMBEDDING_MODEL')}")

store = PGVector(
    embeddings=embeddings,
    collection_name=os.getenv("PGVECTOR_COLLECTION"),
    connection=os.getenv("PGVECTOR_URL"),
    use_jsonb=True
)

results = store.similarity_search_with_score(query, k=3)

for i, (doc, score) in enumerate(results, start=1):
    print("-" * 50)
    print(f"Result {i} (score: {score:.4f}):")
    print("-" * 50)
    print("\nTexto:\n")
    print(doc.page_content.strip())
    print("\nMetadados:\n")
    for k, v in doc.metadata.items():
        print(f"{k}: {v}")