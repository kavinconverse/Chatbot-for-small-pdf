import os
import streamlit as st
import base64
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import Replicate


#column 1
#function to save file
def save_uploadedfile(uploadfile):
    with open(os.path.join('data',uploadfile.name),'wb') as f:
        f.write(uploadfile.getbuffer())
    return st.success(f'Saved File:{uploadfile.name} to directory')

#function to display the pdf of given file
def displayPDF(file):
    #opening file from file path
    with open(file,'rb') as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    #embedding PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'

    #displaying file
    st.markdown(pdf_display,unsafe_allow_html=True)

def load_data(pdf_file):
    loader = PyPDFLoader(pdf_file)
    pages = loader.load_and_split()
    return pages

def split_data(pages):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=50)
    text_chunks = text_splitter.split_documents(pages)
    return text_chunks

def create_embeddings():
    embeddings = SentenceTransformerEmbeddings(model_name  = 'all-MiniLM-L6-v2')
    return embeddings

def create_vectorstore(text_chunks,embeddings):
    vectorstore = FAISS.from_documents(text_chunks,embeddings)
    vectorstore.save_local('vector_faiss')
    return vectorstore

def semantic_search(vectorstore,query):
    relevant_docs = vectorstore.similarity_search(query,k=2)
    return relevant_docs

def qa_response(relevant_docs,query):
    llm = Replicate(
        model="meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
        model_kwargs={'temperature': 0.2,
                      'top_p': 0.5,
                      'max_length': 512, }
    )

    chain = load_qa_chain(llm=llm,chain_type = 'stuff')
    response = chain.run(input_documents=relevant_docs,question = query )
    return response

