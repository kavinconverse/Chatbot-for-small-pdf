import os
import streamlit as st
from dotenv import load_dotenv
from back import *


def main():
    load_dotenv()

    st.set_page_config(layout='wide')
    st.title("PDF Chatbot for Small Pdf")
    st.markdown('<style>h1{color: Green; text-align: center;}</style>', unsafe_allow_html=True)

    uploaded_pdf = st.file_uploader('Upload your PDF',type = ['pdf'])

    if uploaded_pdf is not None:
        col1 , col2 = st.columns([2,2])
        with col1:
            input_file = save_uploadedfile(uploaded_pdf)  #save file
            pdf_file = 'data/' + uploaded_pdf.name  #for later use store in variable
            pdf_view = displayPDF(pdf_file)

        with col2:
            pages = load_data(pdf_file)
            #st.write('Length of document',len(pages))
            text_chunks = split_data(pages)
            #st.write('Length of chunks',len(text_chunks))
            embeddings = create_embeddings()
            #st.write('embeddings created')
            vectorstore = create_vectorstore(text_chunks, embeddings)
            #st.write('Embeddings pushed to vectorstore')
            st.success('Search Area')
            query_search = st.text_area('search your query')
            if query_search:
                relevant_docs = semantic_search(vectorstore, query_search)
                if st.button('Result'):
                    st.write('semantic search relevant_docs', relevant_docs)
                    result = qa_response(relevant_docs,query_search)
                    st.write(result)

if __name__ == '__main__':
    main()