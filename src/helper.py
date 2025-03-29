from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_pdf(data):
    loader = PyPDFLoader(data)  # Load the PDF file
    documents = loader.load()  # Extract text from PDF
    return documents

def text_split(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size= 500 , chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(data)
    return text_chunks

def download_huggingface_embeddding():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings


