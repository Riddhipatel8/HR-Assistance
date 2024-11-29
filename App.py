from langchain_community.vectorstores import FAISS
import gradio as gr
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import os
import streamlit as st
from langchain_core.messages import HumanMessage,AIMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from streamlit_chat import message
from langchain_core.prompts import MessagesPlaceholder
load_dotenv()

os.system("python Setup.py")

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

vector_store = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

groq_api_key = os.getenv('GROQ_API_KEY')
from langchain_groq import ChatGroq
llm=ChatGroq(groq_api_key=groq_api_key, 
             model_name="llama3-8b-8192")

# Prompt Template for model
from langchain_core.prompts import ChatPromptTemplate
# prompt = ChatPromptTemplate.from_template("""
# Answer the following question based only on the provided context. 
# Think step by step before providing a detailed answer. 
# Don't make up answer if you don't have it 
# <context>
# {context}
# </context>
# Question: {input}""")
prompt = ChatPromptTemplate.from_messages([
    ("system"," Answer the user's question based on the context: {context} and keep the answers concise"),
    MessagesPlaceholder(variable_name="messages"),
    ("human","{input}")
])

document_chain=create_stuff_documents_chain(llm,prompt)

# retrivel of data from the store
retriever=vector_store.as_retriever()
retrieval_chain=create_retrieval_chain(retriever,document_chain)


def generate_answer(user_input, messages):
    response = retrieval_chain.invoke({"input": user_input,"messages":messages})
    return response['answer']  # Return the answer from the response


st.set_page_config(page_title="Chat Bot", page_icon="robot")
st.title("chatbot")

if 'messages' not in st.session_state:
    st.session_state.messages = []
for i, msg in enumerate(st.session_state.messages):
    if i % 2 == 0:
        message(msg,is_user=True)
    else:
        message(msg, is_user=False)    


user_query = st.chat_input("your message")
if user_query and isinstance(user_query, str):
    message(user_query, is_user = True)
    st.session_state.messages.append(user_query)
    st.markdown(user_query)
    response = generate_answer(user_query,st.session_state.messages)
    message(response, is_user=False)
    # with st.chat_message("human"):
    #     st.markdown(user_query)

    # with st.chat_message("ai"):
    #     ai_response = generate_answer(user_query,st.session_state.chat_history)
    #     st.markdown(ai_response)

    st.session_state.messages.append(response)  

  
