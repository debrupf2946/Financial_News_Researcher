import os
import streamlit as st
import pickle
import time
import langchain
from langchain.llms import GooglePalm
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import SeleniumURLLoader
#from langchain.embeddings import OpenAIEmbeddings
#from langchain.embeddings import CohereEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


#os.environ["OPENAI_API_KEY"] = "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"
llm=GooglePalm(google_api_key="API_KEY", temperature=0.9)

st.title("News Research Tool")
st.sidebar.title("News article titles")

URLs=[]

for i in range(3):
    url=st.sidebar.text_input(f"URL {i+1}")
    URLs.append(url)

process_url_clicked=st.sidebar.button("Process URLs")
file_path = "vecotr_index.pkl"

main_placeholder=st.empty()

if process_url_clicked:
    #Load Data
    loader=SeleniumURLLoader(urls=URLs)
    main_placeholder.text("data loading")
    data=loader.load()

    #Split Data
    r_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],
        chunk_size=1000,
        chunk_overlap=200
    )
    main_placeholder.text("text is splitting")
    docs = r_splitter.split_documents(data)

    #embeddings
    embeddings = HuggingFaceEmbeddings()
    vector_index = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("embedding vector started building")
    time.sleep(2)


    with open(file_path, "wb") as f:
        pickle.dump(vector_index, f)

query=main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path,"rb") as f:
            vector_store=pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_store.as_retriever())
            result=chain({"question": query}, return_only_outputs=True)
            st.header("Answer")
            st.subheader(result["answer"])
