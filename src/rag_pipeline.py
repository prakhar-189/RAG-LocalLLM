# src/rag_pipeline.py
# -----------------------------------------------
# This module acts as the bridge connecting data to the AI model.
# It defines the specific specific set of instructions that tekks the AI how to answer the question.
# Configures the FAISS database to fetch the top 3 most relevant chunks of text whenever a question is asked.
# Wraps the database, the prompt, and the AI model together into a LangChain pipeline.
# Activates the "return_source_documents = True" flag so the frontend can access the exact paragraphs the AI read to get its answer.
# -----------------------------------------------


# =========================================
# Libraries
# ------------------------------------
# RetrievealQA : For building a question-answering chain that retrieves relevance documents before answering.
# PromptTemplate : For defining a custom prompt template that instructs the AI how to use the retrieved context.
# load_llm : For loading the local Ollama model.
# =========================================
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from src.llm_model import load_llm


# =========================================
# build_qa_chain Function
# ------------------------------------
# This function takes a FAISS vector store as input, sets up the retriever to fetch the top 3 most relevant chunks.
# It defines a custom prompt template to instruct the AI how to use the retrieved context, and builds a RetrievalQA chain that combines the retriever, the prompt, and the local Ollama LLM.
# It returns the fully configured QA chain ready to be used in the frontend.
# The "return_source_documents = True" flag is activated to allow the frontend to access the exact paragraphs the AI read to get its answer, enabling the "Show Your Work" feature.
# =========================================
def build_qa_chain(vectorstore):
    """Builds the QA chain using a dynamically generated vector store."""

    # Set up the retriever to fetch the top 3 most relevant chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Load the LLM
    llm = load_llm()

    # Strict grounding: the model must answer ONLY from the retrieved context.
    # The previous prompt allowed "answer using your general knowledge" when the
    # context was insufficient -- which defeats the purpose of a RAG system and
    # made the "Show Your Work" citations misleading (they were shown even when
    # the answer came from the model's own knowledge, not those sources).
    template = """
    You are a helpful assistant answering questions strictly from the provided context.
    Use ONLY the information in the context below to answer the question.
    If the context does not contain enough information to answer, reply exactly:
    "I could not find the answer in the provided document."
    Do not use any outside knowledge.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    # return_source_documents=True is required to show citations in the UI
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True, 
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain