import streamlit as st
import os
from PIL import Image
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import chromadb
from chromadb.config import Settings

client=chromadb.HttpClient(host="localhost",port=8000,settings=Settings(anonymized_telemetry=False))

model_path = "/home/tuncube/.cache/gpt4all/all-MiniLM-L6-v2.gguf2.f16.gguf"

vectorstore = Chroma(
    client=client,
    collection_name="STM32H503",
    embedding_function=GPT4AllEmbeddings(model_name=model_path,gpt4all_kwargs={'allow_download': False})
    )

retriever = vectorstore.as_retriever(search_kwargs={'k':5})

def query_analyser(user_query):
    template="""You are an AI assistant specialized in analyzing user queries for a Retrieval-Augmented Generation (RAG) system. Your task is to process the input question and generate a list of strings that
    can be used for retrieving relevant information from a vector database.
    Input
    {question}
    Output
    Provide the following:
    A tiny list of 1-2 strings that can be used for retrieval. 
    The list of strings should be as small as possible.
    These strings should be:
    Very relevant to the user's question
    Specific enough to retrieve targeted information
    Each string should be of 1-3 words
    Each string should be on a new line preceded by a hyphen (-).
    Guidelines for generating retrieval strings:
    Do not change semantic or intent
    Include keywords from the question
    Output will be used by a function that splits on '\n-' to store the retrieval strings. 
    Make sure to generate only the retrieval strings separated by '\n-'. 
    Do not provide any explanation, any preamble and anytitle."""
    
    prompt=ChatPromptTemplate.from_template(template)
    model = ChatOllama(model="llama3")
    chain = prompt | model | StrOutputParser()
    response=chain.invoke({"question":user_query})
    return [i.strip() for i in response.split('-') if len(i.strip())!=0]

def grade_doc(question,doc):
    template="""
    **Task:**

    Evaluate the relevance of the Text Document to Question using the following strict criteria:

    1. **Semantic overlap**: Does the text share the same semantic similarity with the user's question? Consider factors such as:
        * Keywords and phrases that match or are related to the question
        * Entities, concepts, and ideas mentioned in the text that align with the question
    2. **Intent alignment**: Is the intent behind the text consistent with the user's question?
    3. **Purpose matching**: Does the purpose of the text match or complement the user's question?
    4. **Goal congruence**: Are the goals and objectives mentioned in the text aligned with those implied by the user's question?
    5. **Meaning coherence**: Is there a clear and logical connection between the meaning conveyed by the text and the user's question?
    
    **Question:**: '{question}'
    **Text Document:**
     
    Title or intent : '{title}'
    Content or description: '{document}'
    
    Assign a grade of 'Relevant' only if all above criteria are met,if any of these criteria are not met, assign a grade of 'Not Relevant'."

    Please use this format : ##Answer: "relevant" or "not relevant"
    """
    prompt=ChatPromptTemplate.from_template(template)
    model = ChatOllama(model="llama3")
    chain = prompt | model | StrOutputParser()
    response=chain.invoke({"document":doc.page_content,"question":question,"title":doc.dict().get("metadata","").get("title","untitled")})
    if 'not relevant' in response.lower():
        print("Not relevant: {}".format(doc.page_content))
    else:
        print("Relevant: {}".format(doc.page_content))
    return not 'not relevant' in response.lower()
     

def grader(user_query,docs):
    print("## Grading Documents ##")
    relevant_docs=list()
    for doc in docs:
        if grade_doc(user_query,doc):
            relevant_docs.append(doc.page_content)
    print(relevant_docs)
    return relevant_docs

def generate_answer(user_query, relevant_docs):
    print("#Generating Answer#")

    template="""You have been provided with the following independent text documents: 
    '{document}'
    User question/statement: '{question}'
    The user has asked a question/statement that requires information from one or more of these independent text documents. 
    Your task is to identify the most relevant and credible text documents related to the user's query, and generate a direct and accurate response.

    **Note:** Please consider the following when evaluating the sources:
    * Assess the relevance of each document to the user's question.
    * Identify any potential biases, inconsistencies, or contradictions in each document that may impact its credibility.
    * Use your knowledge base to verify information and provide additional context where necessary.
    * Synthesize the most reliable information from the relevant sources to generate a concise and accurate response.

    If multiple sources are relevant, provide a summary of the key points and any discrepancies or areas of agreement. 

    Highlight any gaps in the available information and suggest potential avenues for further exploration.

    **Output:** Provide a direct and correct answer that addresses the user's question. """
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOllama(model="llama3")
    chain = prompt | model | StrOutputParser()
    response=chain.invoke({"question":user_query,"document":"\n".join(relevant_docs)})
    print(response)
    return response

    
def askMe(user_query,retriever=retriever):
    relevant_docs=list()
    question=user_query
    queries=query_analyser(question)
    print(queries)
    for query in queries:
        retrieved_docs=retriever.invoke(query)
        relevant_docs.extend(grader(user_query,retrieved_docs))
    answer=generate_answer(user_query, relevant_docs)
    return answer

import streamlit as st
from streamlit_toggle import st_toggle_switch

# Function to apply light or dark mode
def set_theme(is_light_mode):
    if is_light_mode:
        st.markdown("""
        <style>
        .stApp, .stSidebar, .stTextInput, .stButton>button, .stFileUploader {
            background-color: white;
            color: black;
        }
        .stTextInput>div>div>input, .stSelectbox, .css-10trblm, .css-16idsys p, .stMarkdown, .css-183lzff {
            color: black;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .stApp, .stSidebar, .stTextInput, .stButton>button, .stFileUploader {
            background-color: #262730;
            color: white;
        }
        .stTextInput>div>div>input, .stSelectbox, .css-10trblm, .css-16idsys p, .stMarkdown, .css-183lzff {
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)

# App configuration
st.set_page_config(page_title="🤖Chatbot", layout="wide")

# Initialize session state
if 'light_mode' not in st.session_state:
    st.session_state.light_mode = True

# Apply theme
set_theme(st.session_state.light_mode)

# Sidebar
with st.sidebar:
    # Title and Dark/Light mode toggle side by side
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🤖💬 Chatbot")
    with col2:
        mode = st_toggle_switch(
            label="Dark mode",
            key="dark_mode",
            default_value=not st.session_state.light_mode,
            label_after=False,
            inactive_color="#D3D3D3",  # light grey
            active_color="#11567f",  # dark blue
            track_color="#29B5E8"  # lighter blue
        )
        st.session_state.light_mode = not mode

    # Rerun the app to apply theme changes
    if st.session_state.light_mode != (not mode):
        st.rerun()
    
    st.header("File Upload Options")

    # 1. Upload a single PDF file
    pdf_file = st.file_uploader("Upload a PDF file", type="pdf")
    if pdf_file is not None:
        st.success(f"PDF file '{pdf_file.name}' uploaded successfully!")

    # 2. Upload a folder of PDF files
    folder = st.file_uploader("Upload a folder of PDF files", type="pdf", accept_multiple_files=True)
    if folder:
        st.success(f"Uploaded {len(folder)} PDF files from the folder.")
        # Here you would typically process the files and add them to the database

    # 3. Upload a ZIP file
    zip_file = st.file_uploader("Upload a ZIP file", type="zip")
    if zip_file is not None:
        st.success(f"ZIP file '{zip_file.name}' uploaded successfully!")

    # Clear chat history button
    if st.button('Clear Chat History'):
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
        st.rerun()

# Main content
st.title("Chatbot Interface")

# Chat interface
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Function for generating response
def generate_response(prompt_input):
    # Placeholder function, replace with actual implementation
    return f"This is a response to: {prompt_input}"

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)


