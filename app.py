from flask import Flask, render_template, request
from src.helper import download_huggingface_embeddding
from langchain_pinecone import PineconeVectorStore
from langchain.llms import Ollama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__, template_folder="templates")  # Ensure Flask looks for templates

load_dotenv()

index_name = "mediclbot"

# Safely get the API key
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("Missing PINECONE_API_KEY in environment variables")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

embedding = download_huggingface_embeddding()

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding
)

llm = Ollama(model="llama2")
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 5})

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt=prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg")  # ✅ Fixed
    if not msg:
        return "Error: No message provided", 400  # Handle empty input

    print(f"User input: {msg}")

    response = rag_chain.invoke({"input": msg})

    answer = response.get("answer", "Sorry, I couldn't generate a response.")
    print(f"Response: {answer}")

    return answer


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
