import os
import time
import streamlit as st
from datasets import load_dataset
from langchain.docstore.document import Document
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import CollectionStatus
from openai import OpenAI


def prepare_dataset():
    # Load the humaneval dataset from huggingface
    data = load_dataset("openai_humaneval")
    # Get the test data only from the dataset 
    # The data set already has only test data
    data = data["test"]
    # Convert the data to documents format 
    tasks = [
        Document(
            page_content=example["prompt"],
            metadata={
                "task_id": example["task_id"],
                "solution": example["canonical_solution"]
            }
        ) for example in data
    ]
    return tasks

# Vector Store Setup
def setup_vectorstore(tasks):
    # Load BAAI model from huggingface to use as embeddings
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en",
        encode_kwargs={"normalize_embeddings": True}
    )
    # use qdrant to store the embedded data
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    if not qdrant_api_key:
        raise ValueError("Missing QDRANT_API_KEY environment variable.")

    qdrant_client = QdrantClient(
        url="https://0762546b-450b-4d4c-bc82-0c41230701da.us-west-2-0.aws.cloud.qdrant.io",
        api_key=qdrant_api_key,
        timeout=60.0
    )
    # Remove the existing collection (if any) to start fresh and avoid duplicates
    collection_name = "humaneval_tasks"
    collections = qdrant_client.get_collections().collections
    if any(c.name == collection_name for c in collections):
        qdrant_client.delete_collection(collection_name)
        
        while True:
            try:
                status = qdrant_client.get_collection(collection_name).status
                if status == CollectionStatus.GREEN:
                    break
            except:
                break 
        time.sleep(1)  
    
    # Create a new Qdrant collection and store task embeddings
    vectorstore = Qdrant.from_documents(
        documents=tasks,
        embedding=embeddings,
        collection_name=collection_name,
        url="https://0762546b-450b-4d4c-bc82-0c41230701da.us-west-2-0.aws.cloud.qdrant.io",
        api_key=qdrant_api_key,
        prefer_grpc=False,
        )

    return embeddings, vectorstore


# RAG Pipeline Class
class CodeRAGPipeline:
    def __init__(self, embeddings, vectorstore, client):
        self.embeddings = embeddings
        self.vectorstore = vectorstore
        self.client = client
        
    # Embed the user query and retrieve the top-K most similar tasks from the vector store.
    def retrieve_similar_tasks(self, query, k=3):
        query_embedding = self.embeddings.embed_query(query)
        return self.vectorstore.similarity_search_by_vector(query_embedding, k=k)

    def format_prompt(self, query, retrieved_docs):
        examples = ""
        for doc in retrieved_docs:
            examples += f"# Example Task:\n{doc.page_content}\n{doc.metadata['solution']}\n\n"
        return (
            "# Below are several examples of Python programming tasks and their solutions.\n"
            "# Use these to help write a function for the new task at the end.\n\n"
            f"{examples}"
            f"# New Task:\n{query}\n# Your Python solution:\n"
        )
    # Use Llama3 From Groq
    def generate_code(self, query):
        retrieved = self.retrieve_similar_tasks(query)
        prompt = self.format_prompt(query, retrieved)
        response = self.client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a helpful python coding assistant, based on the examples provided you will generate code for the new task."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=500
        )
        return response.choices[0].message.content

# Streamlit App
def main():
    st.set_page_config(page_title="Python Code Generator", layout="centered")
    st.title("Python Code Generator")
    st.markdown("Enter a programming task and get auto-generated Python code", width=500)

    user_prompt = st.text_area("Describe your programming task:", height=150)

    if st.button("Generate Code"):
        if user_prompt.strip() == "":
            st.warning("Please enter a task description.")
        else:
            with st.spinner("Generating code..."):
                try:
                    generated_code = rag_pipeline.generate_code(user_prompt)
                    st.success("Code generated!")
                    st.code(generated_code, language="python")
                except Exception as e:
                    st.error(f"\u274C Error: {e}")

# Run Pipeline Setup Once
tasks = prepare_dataset()
embeddings, vectorstore = setup_vectorstore(tasks)

Groq_API_KEY = os.getenv("GROQ_API_KEY")
if not Groq_API_KEY:
    raise ValueError("Missing GROQ_API_KEY environment variable.")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=Groq_API_KEY
)

rag_pipeline = CodeRAGPipeline(embeddings, vectorstore, client)

if __name__ == '__main__':
    main()