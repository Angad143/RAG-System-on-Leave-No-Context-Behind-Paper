import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from IPython.display import Markdown as md
import streamlit as st

# Setup API Key
with open('keys/.gemini_API_key.txt', 'r') as f:
    GOOGLE_API_KEY = f.read().strip()

# Initialize the chat model
chat_model = ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model="gemini-1.5-pro-latest")

# Setup embedding model
embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model="models/embedding-001")

# Setting up a connection with the ChromaDB
db_connection = Chroma(persist_directory="./chroma_db_", embedding_function=embedding_model)

# Convert CHROMA db_connection to a retriever object
retriever = db_connection.as_retriever(search_kwargs={"k": 5})

# Define chat prompt template
chat_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a Helpful AI Bot. 
    You take the context and question from user. Your answer should be based on the specific context."""),
    HumanMessagePromptTemplate.from_template("""Answer the question based on the given context.
    Context:
    {context}
    
    Question: 
    {question}
    
    Answer: """)
])

# Setup output parser
output_parser = StrOutputParser()

# Define the RAG chain
rag_chain = (
    {"context": retriever, "question": retriever}
    | chat_template
    | chat_model
    | output_parser
)

# Streamlit UI
st.title("🤖 Q&A using Retrieval-Augmented Generation System")
st.subheader('Pose questions specific to the content of the "Leave No Context Behind" Paper')

user_input = st.text_input("Enter your question here....")

if st.button("Generate"):
    response = rag_chain.invoke(user_input)
    st.markdown(response)
