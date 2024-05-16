import langchain
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS, pinecone
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from tqdm.auto import tqdm

load_dotenv()

llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-flash", google_api_key=os.getenv("GOOGLE_API_KEY")
)

pdf_pages = []
for file in os.listdir('pdfs/PMB271'):
    pdf_loader = PyPDFLoader(file, extract_images=True)
    pdf_pages += pdf_loader.load_and_split()

faiss_index = FAISS.from_documents(pdf_pages, GoogleGenerativeAIEmbeddings())
