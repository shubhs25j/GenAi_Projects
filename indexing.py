''' This script indexes a PDF document into a vector database using LangChain and Qdrant.'''
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

pdf_path = Path(__file__).parent / "nodejs.pdf"
# html_path = Path(__file__).parent / "TCPIP_ConsolidatedReport.html"

# Loading
loader = PyPDFLoader(file_path=pdf_path)
# loader = UnstructuredHTMLLoader(file_path=html_path)
docs = loader.load()  # Read PDF File

# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10
)

split_docs = text_splitter.split_documents(documents=docs)

# Vector Embeddings
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

# Using [embedding_model] create embeddings of [split_docs] and store in DB

vector_store = QdrantVectorStore.from_documents(
    documents=split_docs,
    url="http://localhost:6333/",
    collection_name="learning_vectors",
    # collection_name="TCPIConsolidatedReport_vectors",
    embedding=embedding_model
)

print("Indexing of Documents Done...")
