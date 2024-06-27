import os
from typing_extensions import Doc
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.vectorstores import DocArrayInMemorySearch
from langchain_core.prompts import ChatPromptTemplate
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from tqdm.auto import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
import streamlit as st

# import asyncio
import nest_asyncio

nest_asyncio.apply()
# asyncio.set_event_loop()

load_dotenv()

# Initialize app resources
st.set_page_config(page_title="StudyAssist", page_icon=":book:")
st.title("Study Assist")
st.write(
    "An AI/RAG application to aid students in their studies, specially optimized for the pharm 028 students"
)


@st.cache_resource
def initialize_resources():
    llm_gemini = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest", google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    return llm_gemini


@st.cache_resource
def get_retriever(pdf_file):
    pdf_loader = PyPDFLoader(pdf_file, extract_images=False)
    pages = pdf_loader.load()

    underlying_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
        separators="\n",
    )
    documents = text_splitter.split_documents(pages)
    vectorstore = DocArrayInMemorySearch.from_documents(
        documents, underlying_embeddings
    )
    doc_retiever = vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10}
    )

    return doc_retiever


chat_model = initialize_resources()


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

courses = ["PMB", "PCL", "Kelechi_research"]  # "GSP", "CPM", "PCG",  "PCH"
course_pdfs = None

course = st.sidebar.selectbox("Choose course", (courses))
docs_path = f"pdfs/{course}"
course_pdfs = os.listdir(docs_path)
pdfs = [os.path.join(docs_path, pdf) for pdf in course_pdfs]

course_material = "{Not selected}"

try:
    # if st.sidebar.button('Get available course pdfs'):
    if course_pdfs:
        course_material = st.sidebar.selectbox(
            "Select course pdf", (pdf for pdf in pdfs)
        )
    if course_material:
        st.write(f"AI Chatbot for **{course}**: {course_material}")
        st.success("File loading successful, vector db initialized")

    else:
        uploaded_file = st.sidebar.file_uploader("or Upload your own pdf", type="pdf")
        if uploaded_file is not None:
            course_material = uploaded_file
            st.write(f"AI Chatbot for **{course}**: {uploaded_file.filename}")

            retriever = get_retriever(course_material)

        st.success("File loading successful, vector db initialized")

except Exception as e:
    st.error(e)

doc_retriever = None
conversational_chain = None

if st.sidebar.button("Load pdf"):
    with st.spinner("Loading material..."):
        doc_retriever = load_pdf(course_material)
        conversational_chain = ConversationalRetrievalChain.from_llm(
            chat_model, doc_retriever
        )

conversational_chain = ConversationalRetrievalChain.from_llm(chat_model, doc_retriever)


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
