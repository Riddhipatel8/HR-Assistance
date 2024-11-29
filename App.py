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
 
# Set page configuration
st.set_page_config(page_title="Chat Bot", page_icon="robot")
st.title("Chatbot")
 
# Inject custom CSS for styling the chat, dynamically adjust based on theme
st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500&display=swap');
        /* Apply custom font globally */
        body {{
            font-family: 'Roboto', sans-serif;
        }}
 
        /* Custom styling for user input */
        .user-message {{
            font-family: 'Roboto', sans-serif;
            font-size: 16px;
            color: #ffffff;
            background-color: #3399FF;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 20px;
            display: inline-block; /* Shrinks background to text size */
            max-width: 80%; /* Prevents the message from stretching too wide */
            word-wrap: break-word; /* Ensures long words wrap to the next line */
            float: right; /* Align user messages to the right */
        }}
 
        /* Custom styling for bot response */
        .bot-message {{
            font-family: 'Times New Roman', serif;
            font-size: 16px;
            color: #484848;
            background-color: #D8D8D8;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
            display: inline-block; /* Shrinks background to text size */
            max-width: 80%; /* Prevents the message from stretching too wide */
            word-wrap: break-word; /* Ensures long words wrap to the next line */
            float: left; /* Align bot messages to the left */
        }}
 
        /* Clear floats after messages */
        .chat-container {{
            overflow: hidden;
        }}
    </style>
""", unsafe_allow_html=True)
 
# Initialize session state for messages if it's not already initialized
if 'messages' not in st.session_state:
    st.session_state.messages = []
 
# Create a container for chat messages to manage floats
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
 
# Display the chat history
for i, msg in enumerate(st.session_state.messages):
    if i % 2 == 0:
        st.markdown(f'<div class="user-message">{msg}</div>', unsafe_allow_html=True)  # User's message
    else:
        st.markdown(f'<div class="bot-message">{msg}</div>', unsafe_allow_html=True)  # Bot's message
 
st.markdown('</div>', unsafe_allow_html=True)
 
# Input for user query
user_query = st.chat_input("Your message")
 
# If the user inputs a query, generate and display the response
if user_query and isinstance(user_query, str):
    st.markdown(f'<div class="user-message">{user_query}</div>', unsafe_allow_html=True)  # Display user's message
    st.session_state.messages.append(user_query)  # Add user's message to session state
    response = generate_answer(user_query, st.session_state.messages)  # Generate response
    st.markdown(f'<div class="bot-message">{response}</div>', unsafe_allow_html=True)  # Display bot's response
    st.session_state.messages.append(response)  # Add bot's response to session state
