import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import logging
import tempfile
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to load and process the PDF
def process_pdf(pdf_path):
    try:
        logger.info(f"Attempting to load PDF from {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        logger.info(f"Successfully loaded PDF. Number of pages: {len(documents)}")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)
        logger.info(f"Split documents into {len(texts)} chunks")
        
        return texts
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        st.error(f"Error processing PDF: {str(e)}")
        return []

# Function to create vector store
def create_vector_store(documents):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

# Function to get response from LLama 2 model
def get_llama_response(query, vector_store):
    try:
        llm = CTransformers(
            model=r'C:\Users\vaibh\Documents\RAG\pdf_rag\models\llama-2-7b-chat.ggmlv3.q8_0.bin',
            model_type='llama',
            config={'max_new_tokens': 1000, 'temperature': 0.01}
        )
        
        relevant_docs = vector_store.similarity_search(query, k=2)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        template = """
        Context information is below.
        ---------------------
        {context}
        ---------------------
        Given the context information and not prior knowledge, answer the query.
        Query: {query}
        Answer: """
        
        prompt = PromptTemplate(
            input_variables=["context", "query"],
            template=template,
        )
        
        full_prompt = prompt.format(context=context, query=query)
        
        logger.info("Sending prompt to LLaMa model...")
        response = llm.invoke(full_prompt)
        logger.info("Received response from LLaMa model.")
        
        return response if response else "The model did not generate a response. Please try again."
    except Exception as e:
        logger.error(f"An error occurred in get_llama_response: {str(e)}")
        return f"An error occurred while generating the response: {str(e)}"

# Streamlit UI
st.title("PDF Question Answering with LLaMa 2")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_file_path = temp_file.name
        
        # Process the PDF
        documents = process_pdf(temp_file_path)
        if documents:
            vector_store = create_vector_store(documents)
            st.success("PDF processed successfully!")
        else:
            st.error("Failed to process the PDF. Please try a different file.")
        
        # Clean up the temporary file
        os.unlink(temp_file_path)

    # Question input
    query = st.text_input("Ask a question about the PDF:")

    if query and 'vector_store' in locals():
        with st.spinner("Generating answer..."):
            response = get_llama_response(query, vector_store)
            st.write("Answer:", response)