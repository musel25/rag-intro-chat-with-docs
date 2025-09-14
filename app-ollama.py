import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from ollama import Client  # for generation

load_dotenv()

# --- Embeddings via Ollama ---
ollama_ef = OllamaEmbeddingFunction(
    url="http://localhost:11434",
    model_name="embeddinggemma:300m",
)

# Chroma client + collection
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage_ollama")
collection = chroma_client.get_or_create_collection(
    name="document_qa_collection_ollama",
    embedding_function=ollama_ef,
)

# Function to load documents from a directory
def load_documents_from_directory(directory_path):
    print("==== Loading documents from directory ====")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(
                os.path.join(directory_path, filename), "r", encoding="utf-8"
            ) as file:
                documents.append({"id": filename, "text": file.read()})
    return documents


# Function to split text into chunks
def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks


# Load documents from the directory
directory_path = "./news_articles"
documents = load_documents_from_directory(directory_path)

print(f"Loaded {len(documents)} documents")
# Split documents into chunks
chunked_documents = []
for doc in documents:
    chunks = split_text(doc["text"])
    print("==== Splitting docs into chunks ====")
    for i, chunk in enumerate(chunks):
        chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text": chunk})

# print(f"Split documents into {len(chunked_documents)} chunks")

# Upsert without computing embeddings yourself (Chroma will call Ollama)
for doc in chunked_documents:
    print("==== Generating embeddings. and Inserting chunks into db;;;====")
    collection.upsert(ids=[doc["id"]], documents=[doc["text"]])

# Query (pass a list for query_texts)
def query_documents(question, n_results=2):
    res = collection.query(query_texts=[question], n_results=n_results)
    return [d for docs in res["documents"] for d in docs]

# --- Generation via Ollama chat model ---
llm = Client(host="http://localhost:11434")

def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    user_msg = (
        "You are an assistant for question-answering tasks. Use the context to answer. "
        "If you don't know, say so. Three sentences max.\n\n"
        f"Context:\n{context}\n\nQuestion:\n{question}"
    )
    resp = llm.chat(
        model="gemma3n:e4b",  # or "gemma2:9b-instruct", "qwen2.5:7b-instruct"
        messages=[{"role": "user", "content": user_msg}],
        # options={"temperature": 0.2}  # optional
    )
    return resp.message.content

# Example query
# query_documents("tell me about AI replacing TV writers strike.")
# Example query and response generation
question = "tell me about databricks"
relevant_chunks = query_documents(question)
answer = generate_response(question, relevant_chunks)

print(answer)
