from flask import Flask , render_template , jsonify , request
from src.helper import download_huggingface_embeddding
from langchain_pinecone import PineconeVectorStore
from langchain.llms import Ollama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)
load_dotenv()

index_name = "medibot"

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

embedding = download_huggingface_embeddding()

docsearch = PineconeVectorStore.from_existing_index(
    index_name = index_name,
    embedding= embedding
)

llm = Ollama(model='llama2')
retriever = docsearch.as_retriever(search_type = "similarity" , search_kwargs = {"k" : 1})

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        ("human","{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(llm,prompt=prompt)
rag_chain = create_retrieval_chain(retriever,question_answer_chain)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get",methods = ["GET" , "POST"])
def chat():
    msg = request.form("msg")
    input = msg
    print(input)
    response = rag_chain.invoke({"input":msg})
    print("Response : ",response['answer'])
    return str(response['answer'])