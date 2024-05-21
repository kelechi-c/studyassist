import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain.chat_models import
from dotenv import load_dotenv
from tqdm.auto import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import CacheBackedEmbeddings
from langchain.chains import ConversationalRetrievalChain
import asyncio
import streamlit as st
import streamlit_chat as stchat

load_dotenv()

# Configs

asyncio.set_event_loop(asyncio.new_event_loop())

# Initialize app resources
st.set_page_config(page_title="StudyAssist", page_icon=":book:")
st.title("Study Assist")


@st.cache_resource
def initialize_reources():
    llm_gemini = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest", google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    underlying_embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
    )

    store = LocalFileStore("./cache/")

    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings, store, namespace=underlying_embeddings.model
    )

    return llm_gemini, cached_embedder

chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=os.getenv("GOOGLE_API_KEY"))

store = LocalFileStore("./cache/")
underlying_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

cached_embedder = CacheBackedEmbeddings.from_bytes_store(underlying_embeddings, store, namespace=underlying_embeddings.model)
store = LocalFileStore("./cache/")

cached_embedder = CacheBackedEmbeddings.from_bytes_store(underlying_embeddings, store, namespace=underlying_embeddings.model)

chat_model, embedder = initialize_reources()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, chunk_overlap=20, length_function=len, is_separator_regex=False
)


def load_pdf(pdf_file):
    pdf_loader = PyPDFLoader(pdf_file, extract_images=True)
    pages = pdf_loader.load_and_split()
    documents = text_splitter.split_documents(pages)

    faiss_index_db = FAISS.from_documents(documents, cached_embedder)
    retriever = faiss_index_db.as_retriever()
    
    return retriever


def query_response(query, retriever):
    prompt = ChatPromptTemplate.from_template(
        """Answer the question based only on the following context:
    
            {context}

            Question: {question}
        """
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | chat_model
        | StrOutputParser()
    )

    response = chain.invoke(query)

    return response


# Streamlit UI

# Course list and pdf retrieval

courses = ["PMB", "PCL"]  #  "GSP", "CPM", "PCG",  "PCH",
course_pdfs = None

try:
    course = st.sidebar.selectbox('Choose course', (courses))  
    docs_path = f"pdfs/{course}"
    course_pdfs = os.listdir(docs_path)
    pdfs = [os.path.join(docs_path, pdf) for pdf in course_pdfs]
    
except Exception as e:
    st.error("Course materials not found")


course_material = '{Not selected}'

if st.sidebar.button('Get available course pdfs'):
    if course_pdfs:
        course_material = st.sidebar.selectbox("Select course pdf", (pdf[8:] for pdf in pdfs))

uploaded_file = st.sidebar.file_uploader("Upload your own pdf", type="pdf")

st.write(f"AI Chatbot for **{course}**: {course_material}")

doc_retriever = None

# Start retriever
try:
    if st.sidebar.button('Load pdf'):
        with st.spinner('Loading material...'):
            doc_retriever = load_pdf(course_material)
            conversational_chain = ConversationalRetrievalChain.from_llm(chat_model, doc_retriever)
                
except Exception as e:
    st.error(e)

# Initialize session state

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = [f'Hello! ask me about the pdf: {course_material}']

if "past" not in st.session_state:
    st.session_state["past"] = ["Hey ! 👋"]

# Chat UI
chat_container = st.container()
response_container = st.container()

def chat_ui_function():
    with chat_container:
        with st.form(key="my_form", clear_on_submit=True):
            user_input = st.text_input(
                "Query:", placeholder="Converse with material", key="input"
            )
            submit_button = st.form_submit_button(label="Send")

    # Submit and generate
        if submit_button and user_input:
            response = conversational_chain(user_input)
            st.session_state["past"].append(user_input)
            st.session_state["generated"].append(response)


    if st.session_state["generated"]:
        with response_container:
            for k in range(len(st.session_state["generated"])):
                stchat.message(
                    st.session_state["past"][k],
                    is_user=True,
                    key=str(k) + "_user",
                    avatar_style="big-smile")

                stchat.message(st.session_state["generated"][k], key=str(k), avatar_style="thumbs")


# Begin chat
try:
    if st.button('Initialize chatbot'):
        chat_ui_function()
        
except Exception as e:
    st.error(e)
    
    

st.write("")
st.write("")
st.write("")


st.markdown(
    """
    <div style="text-align: center; padding: 1rem;">
        Project by <a href="https://github.com/kelechi-c" target="_blank" style="color: white; font-weight: bold; text-decoration: none;">
         kelechi(tensor)</a>
    </div>
""",
    unsafe_allow_html=True,
)
