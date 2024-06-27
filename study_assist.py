import os
from posix import write
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import faiss
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from pandas.io.formats.style import doc
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


def get_retriever(pdf_file):
    # path = os.path.join(os.getcwd(), pdf_file.name)

    with NamedTemporaryFile(suffix="pdf") as temp:
        temp.write(pdf_file.getvalue())
        pdf_loader = PyPDFLoader(temp.name, extract_images=False)
        pages = pdf_loader.load()

    st.write(f"AI Chatbot for {course_material}")

    # underlying_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    underlying_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
        separators="\n",
    )
    documents = text_splitter.split_documents(pages)
    vectorstore = faiss.FAISS.from_documents(documents, underlying_embeddings)
    doc_retiever = vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10}
    )

    return doc_retiever


chat_model = initialize_resources()

# Streamlit UI
# Course list and pdf retrieval

courses = ["PMB", "PCL", "Kelechi_research"]  # "GSP", "CPM", "PCG",  "PCH"
course_pdfs = None
doc_retriever = None
conversational_chain = None

course = st.sidebar.selectbox("Choose course", (courses))
docs_path = f"pdfs/{course}"
course_pdfs = os.listdir(docs_path)
pdfs = [os.path.join(docs_path, pdf) for pdf in course_pdfs]

course_material = "{Not selected}"


@st.cache_resource
def query_response(query, _retriever):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=chat_model, retriever=_retriever, memory=memory, verbose=False
    )
    response = conversational_chain.run(query)

    return response


def get_file():
    # if course_pdfs:
    # course_material = st.sidebar.selectbox(
    #   "Select course pdf", (pdf for pdf in pdfs), key="pdf"
    # )
    course_material = st.file_uploader("or Upload your own pdf", type="pdf", key="doc")
    with NamedTemporaryFile(suffix="pdf") as temp:
        temp.write(course_material.getvalue())

    st.write(f"AI Chatbot for {course_material}")

    print(course_material.getvalue())
    return course_material.name


# try:
# if st.sidebar.button('Get available course pdfs'):
# if course_pdfs:
#   course_material = st.sidebar.selectbox("Select course pdf", (pdf for pdf in pdfs))

# if course_material:
# st.write(f"AI Chatbot for **{course}**: {course_material}")
# st.success("File loading successful, vector db initialized")

# else:
#   uploaded_file = st.sidebar.file_uploader("or Upload your own pdf", type="pdf")
#  if uploaded_file is not None:
#     course_material = uploaded_file
#    st.write(f"AI Chatbot for **{course}**: {uploaded_file.filename}")

# doc_retriever = get_retriever(course_material)


# if st.button("Load pdf"):
#   with st.spinner("Loading material..."):
#      # doc_retriever = load_pdf(course_material)
#     doc_retriever = get_retriever(course_material)
#
#       st.success("File loading successful, vector db initialized")
#      input_text = st.text_input(label="Ask...")

# response = conversational_chain.invoke(input)

if "doc" not in st.session_state:
    st.session_state.doc = ""


course_material = st.file_uploader("or Upload your own pdf", type="pdf")

if st.session_state != "":
    # urls = get_page_urls(base_url)
    doc_retriever = get_retriever(course_material)
    st.success("File loading successful, vector db initialize")

    # We store the conversation in the session state.
    # This will be use to render the chat conversation.
    # We initialize it with the first message we want to be greeted with.
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "How may I help you today?"}
        ]

    if "current_response" not in st.session_state:
        st.session_state.current_response = ""

    # We loop through each message in the session state and render it as
    # a chat message.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # We initialize the quantized LLM from a local path.
    # Currently most parameters are fixed but we can make them
    # configurable.
    # llm_chain = create_chain(retriever)

    # We take questions/instructions from the chat input to pass to the LLM
    if user_prompt := st.chat_input("Your message here", key="user_input"):
        # Add our input to the session state
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        # Add our input to the chat window
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Pass our input to the llm chain and capture the final responses.
        # It is worth noting that the Stream Handler is already receiving the
        # streaming response as the llm is generating. We get our response
        # here once the llm has finished generating the complete response.
        response = query_response(user_prompt, doc_retriever)
        # Add the response to the session state
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Add the response to the chat window
        with st.chat_message("assistant"):
            st.markdown(response)
#
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
