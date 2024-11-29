import httpx
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import os
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from streamlit_chat import message
from langchain_core.prompts import MessagesPlaceholder
from langchain_groq import ChatGroq

# Create an httpx client with disabled proxies
client = httpx.Client(proxies={})

load_dotenv()

# Initialize the LLM with the httpx client
groq_api_key = os.getenv('GROQ_API_KEY')

# Pass the httpx client to the ChatGroq model
llm = ChatGroq(groq_api_key=groq_api_key, 
              model_name="llama3-8b-8192", 
              http_client=client)

# Prompt Template for model
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([ 
    ("system", " Answer the user's question based on the context: {context} and keep the answers concise"),
    MessagesPlaceholder(variable_name="messages"),
    ("human", "{input}")
])

document_chain = create_stuff_documents_chain(llm, prompt)

# Retrieving data from the store
vector_store = FAISS.load_local("faiss_index", HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5"), allow_dangerous_deserialization=True)
retriever = vector_store.as_retriever()

retrieval_chain = create_retrieval_chain(retriever, document_chain)

def generate_answer(user_input, messages):
    response = retrieval_chain.invoke({"input": user_input, "messages": messages})
    return response['answer']  # Return the answer from the response

# Set custom page title and icon
st.set_page_config(page_title="Chat Bot", page_icon="robot")
st.title("Chatbot")

# Add custom CSS for font styling
st.markdown("""
    <style>
        /* Set font for the entire page */
        body {
            font-family: 'Arial', sans-serif;
        }
        /* Style the user input text */
        .user-message {
            font-family: 'Courier New', monospace;
            font-size: 16px;
            color: #1e90ff;
        }
        /* Style the bot response text */
        .bot-message {
            font-family: 'Times New Roman', serif;
            font-size: 16px;
            color: #32cd32;
        }
    </style>
""", unsafe_allow_html=True)

# Display messages in the chat
if 'messages' not in st.session_state:
    st.session_state.messages = []

for i, msg in enumerate(st.session_state.messages):
    if i % 2 == 0:
        message(msg, is_user=True, key=f"user_{i}", avatar_style="circle", custom_css="user-message")
    else:
        message(msg, is_user=False, key=f"bot_{i}", avatar_style="circle", custom_css="bot-message")

# Get user input and generate the response
user_query = st.chat_input("Your message")
if user_query and isinstance(user_query, str):
    message(user_query, is_user=True, key=f"user_input", avatar_style="circle", custom_css="user-message")
    st.session_state.messages.append(user_query)
    response = generate_answer(user_query, st.session_state.messages)
    message(response, is_user=False, key=f"bot_response", avatar_style="circle", custom_css="bot-message")
    st.session_state.messages.append(response)
