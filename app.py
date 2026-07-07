# app.py
# -----------------------------------------------
# This is the user interface and the main entry point of the application.
# Builds the web interfave using Streamlit.
# Handles the user's PDF upload, securely saves it to a temporary file, and deletes that file once processed to prevent storage bloating.
# Keeps the AI model & chat history loaded in memory so it doesn't have to restart every time the user types a new question.
# Takes the output from the backend and renders both the final answer and the interactive "Show Your Work" citations.
# -----------------------------------------------


# =========================================
# Libraries
# ------------------------------------
# Streamlit : For building the web interface.
# tempfile : For securely handling temporary files.
# os : For file operations like deleting temporary files.
# src.create_vector_store : For creating the vector store from the uploaded PDF.
# src.rag_pipeline : For building the Q&A chain using the vector store.
# =========================================
import streamlit as st
import tempfile
import os
from src.vector_store import create_vector_store
from src.rag_pipeline import build_qa_chain

# Configure the Streamlit page & title
st.set_page_config(page_title="Local LLM RAG Explorer", layout="wide")
st.title("🧠 Local Generative AI RAG System")
st.markdown("Upload a PDF document and ask questions. Powered by LangChain, FAISS, and Ollama (Phi-3).")

# Initialize session state to hold the QA chain across interactions
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# Sidebar for document upload
with st.sidebar:
    st.header("1. Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF to analyze", type=["pdf"])

    # Process the file if uploaded and not already processed in this session
    if uploaded_file is not None and st.session_state.qa_chain is None:
        with st.spinner("Processing document and building vector store..."):
            
            # Save uploaded file temporarily for LangChain's PyPDFLoader
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            try:
                # Create vector store and build the QA chain dynamically
                vectorstore = create_vector_store(tmp_path)
                st.session_state.qa_chain = build_qa_chain(vectorstore)
                st.success("✅ Document processed successfully!")
            except Exception as e:
                st.error(f"Error processing document: {e}")
            finally:
                # Clean up the temporary file to prevent storage bloat
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    # Allow the user to reset the session to upload a new document
    if st.session_state.qa_chain is not None:
        if st.button("Reset Session / Upload New File"):
            st.session_state.qa_chain = None
            st.rerun()

# Main Chat Interface
st.header("2. Ask Questions")
if st.session_state.qa_chain:
    user_query = st.text_input("What would you like to know about the uploaded document?")

    if user_query:
        with st.spinner("Searching knowledge base and generating answer..."):
            try:
                # Get response from the QA chain (.invoke replaces the deprecated
                # __call__ interface, i.e. qa_chain({"query": ...})).
                response = st.session_state.qa_chain.invoke({"query": user_query})
                
                # Display the main generated answer
                st.subheader("Answer")
                st.write(response["result"])
                
                # Display Source Citations dynamically
                with st.expander("📄 View Source Citations (Show Your Work)"):
                    if "source_documents" in response:
                        for i, doc in enumerate(response["source_documents"]):
                            # Extract page number if available in metadata from PyPDFLoader
                            page_num = doc.metadata.get('page', 'Unknown')
                            st.markdown(f"**Source {i+1} (Page {page_num}):**")
                            st.info(doc.page_content)
                    else:
                        st.write("No source documents returned.")
            except Exception as e:
                st.error(f"An error occurred while generating the answer: {e}")
else:
    st.info("👈 Please upload a PDF document in the sidebar to begin.")