import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

#sample API key
OPENAI_API_KEY = "sk-tVuJ3yfoT8EjCrvso43HT3BlbkFJvITk234657nubvarygv" #openai key

#upload pdf files
st.header ("Hello,Welcome back!")
st.text ("Upload your document and start asking questions...")

st.checkbox ("Verify you are a human")

with st.sidebar:
    #st.title("Your documents")
    file = st.file_uploader("Upload a file and start asking questions", type="pdf")

#extract the text
if file is not None:
    pdf_reader = PdfReader(file)
    text =""
    for page in pdf_reader.pages:
        text += page.extract_text()
    #st.write(text)
#break it into chunks
    textsplitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap= 150,
        length_function=len
    )
    chunks = textsplitter.split_text(text)
    #st.write(chunks)

    #generating embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    #creating vector store
    vector_store = FAISS.from_texts(chunks,embeddings)

    #get user question
    user_question = st.text_input("Type your question here")

    #do similarity search
    if user_question is not None:
        match = vector_store.similarity_search(user_question)
        #st.write(match)

        #define llm
        llm = ChatOpenAI(
            openai_api_key = OPENAI_API_KEY,
            temperature = 0,
            max_tokens =5000,
            model_name = "gpt-4-turbo"
        )

        #output results
        chain = load_qa_chain(llm,chain_type="stuff")
        response = chain.run(input_documents=match,question=user_question)
        st.write(response)
