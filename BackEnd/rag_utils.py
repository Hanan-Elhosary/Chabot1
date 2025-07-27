import os
import chromadb
from chromadb.utils import embedding_functions
from together import Together
from dotenv import load_dotenv

load_dotenv()
together_key = os.getenv("TOGETHER_API_KEY")

client = Together(api_key=together_key)

together_ef = embedding_functions.TogetherAIEmbeddingFunction(
    api_key=together_key,
    model_name="BAAI/bge-base-en-v1.5"
)

chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection = chroma_client.get_or_create_collection(
    name="document_qa_collection",
    embedding_function=together_ef
)

def load_documents_from_directory(directory_path: str):
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(os.path.join(directory_path, filename), "r", encoding="utf-8") as f:
                documents.append({"id": filename, "text": f.read()})
    return documents

def split_text(text: str, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

def get_embeddings(text: str):
    response = client.embeddings.create(input=text, model="BAAI/bge-base-en-v1.5")
    embedding = response.data[0].embedding
    return embedding

def prepare_documents(directory_path: str):
    documents = load_documents_from_directory(directory_path)
    chunked_documents = []
    for doc in documents:
        chunks = split_text(doc["text"])
        for i, chunk in enumerate(chunks):
            chunked_documents.append({
                "id": f"{doc['id']}_chunk{i+1}",
                "text": chunk
            })
    for doc in chunked_documents:
        embedding = get_embeddings(doc["text"])
        collection.upsert(
            ids=[doc["id"]],
            documents=[doc["text"]],
            embeddings=[embedding]
        )

def query_documents(question: str, n_results=2):
    results = collection.query(query_texts=[question], n_results=n_results)
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    return relevant_chunks

def generate_response(question: str, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant for question-answering tasks. "
                "Use the retrieved context to answer the question. "
                "If you don't know the answer, say that you don't know. "
                "Use three sentences maximum and keep the answer concise."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{question}",
        },
    ]
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=messages,
    )
    return response.choices[0].message.content
