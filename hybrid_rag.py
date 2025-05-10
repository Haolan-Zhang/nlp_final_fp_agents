"""
Description:
This script implements a hybrid RAG system using LangChain.
It combines keyword-based and vector-based retrieval methods to answer questions based on a given context.
The system is designed to work with both structured and unstructured text data, allowing for flexible document processing.
The script includes functions for loading documents, splitting them into chunks, creating retrievers, and generating responses.
It also evaluates the system's performance by comparing the generated responses with a set of predefined questions and answers.
"""

from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import pandas as pd
import os

from naive_rag import get_naive_vector_retriever


os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"

# 1. Indexing and building chunks
def get_chunks(md_file_path='doc.md', structure_loader=False, chunk_size=1000, chunk_overlap=100, separators=[r"(?=\n\d{4}(?:\.\d+)+\s)"]):
    """
    Function to split the content of a file into chunks.
    
    :param md_file_path: Path to the preprocessed markdown file.
    :param structure_loader: Boolean indicating whether to use structured loader.
    :param chunk_size: Size of each chunk.
    :param chunk_overlap: Overlap between chunks.
    :param separators: List of regex patterns to split the content.
    :return: List of Document objects representing the chunks.
    """
    if structure_loader:
        # Preserve the structure of the text
        with open(md_file_path, "r", encoding="utf-8") as f:
            structured_content = f.read()
        # Create a Document manually
        data = [Document(page_content=structured_content, metadata={"source": md_file_path})]
    else:
        # Load the markdown file without preserving structure
        loader = UnstructuredMarkdownLoader(md_file_path)
        data = loader.load()
    
    content = data[0].page_content
    metadata = data[0].metadata

    if separators: # by regex
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators, # Regex to split documents by code index
            is_separator_regex=True,
            length_function=len
        )
    else: # by characters
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
    # Split the content into chunks
    chunks = text_splitter.create_documents(texts=[content], metadatas=[metadata])
    return chunks

# 2. Creating hybrid retrievers
def get_retrievers(chunks, k=3, weight_kw=0.5):
    """
    Function to create hybrid retrievers.
    
    :param md_split: List of Document objects representing the chunks.
    :param k: Number of top documents to retrieve.
    :param weight_kw: Weight for the keyword retriever.
    :return: EnsembleRetriever object.
    """
    # Creating vector retriever
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever()

    # Creating keyword retriever
    keyword_retriever = BM25Retriever.from_documents(chunks)
    keyword_retriever.k =  k

    # Creating ensemble retriever
    ensemble_retriever = EnsembleRetriever(retrievers=[retriever, keyword_retriever], weights=[1-weight_kw, weight_kw])
    
    return ensemble_retriever


# 3. RAG chain
def get_rag_chain(ensemble_retriever, template):
    """
    Function to create RAG chain.
    
    :param ensemble_retriever: EnsembleRetriever object.
    :return: RAG chain object.
    """
    # Creating llm
    llm = ChatOpenAI()

    prompt = ChatPromptTemplate.from_template(template)

    # Setup RAG pipeline
    rag_chain = (
        {"context": ensemble_retriever,  "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# 4. Evaluation
def get_response(qa_list, rag_chain, ensemble_retriever, output_filename):
    """
    Function to get responses from the RAG chain and save them to a CSV file.
    :param qa_list: List of dictionaries containing questions and answers.
    :param rag_chain: RAG chain object.
    :param ensemble_retriever: EnsembleRetriever object.
    :param output_filename: Name of the output CSV file.
    """
    # create dataset 1
    answers = [item["answer"] for item in qa_list]
    questions = [item["question"] for item in qa_list]
    response = []
    contexts = []

    # Inference
    for query in questions:
        response.append(rag_chain.invoke(query))
        contexts.append([docs.page_content for docs in ensemble_retriever.get_relevant_documents(query)])

    # To dict
    data = {
        "query": questions,
        'answer': answers,
        "response": response,
        'manual_eval': None,
        "context": contexts,
    }
    df = pd.DataFrame(data)
    df.to_csv(f'{output_filename}.csv', index=False)



if __name__ == "__main__":
    # Template for QA prompting
    template = """
    You are a helpful assistant that answers questions strictly based on the provided context.
    If the answer is not present in the context, respond with: "I don't know."

    Context:
    {context}

    Question:
    {input}

    Answer:
    """

    # Sample expert set
    from qa import qa_data_accessibility
    # Sample amateur set
    from qa import qa_data_accessibility_paraphrased

    # Model 1: Hybrid RAG + raw text + char-based chunking
    chunks = get_chunks(md_file_path='doc.md', structure_loader=False, separators=None)
    ensemble_retriever = get_retrievers(chunks)
    rag_chain = get_rag_chain(ensemble_retriever, template)
    get_response(qa_data_accessibility, rag_chain, 'hybrid_raw_char_expert')
    get_response(qa_data_accessibility_paraphrased, rag_chain, 'hybrid_raw_char_amateur')

    # Model 2: Hybrid RAG + raw text + regex-based chunking
    chunks = get_chunks(md_file_path='doc.md', structure_loader=False, separators=[r"(?=\n\d{4}(?:\.\d+)+\s)"])
    ensemble_retriever = get_retrievers(chunks)
    rag_chain = get_rag_chain(ensemble_retriever, template)
    get_response(qa_data_accessibility, rag_chain, 'hybrid_raw_regex_expert')
    get_response(qa_data_accessibility_paraphrased, rag_chain, 'hybrid_raw_regex_amateur')

    # Model 3: Hybrid RAG + structured text + char-based chunking
    chunks = get_chunks(md_file_path='doc.md', structure_loader=True, separators=None)
    ensemble_retriever = get_retrievers(chunks)
    rag_chain = get_rag_chain(ensemble_retriever, template)
    get_response(qa_data_accessibility, rag_chain, 'hybrid_structured_char_expert')
    get_response(qa_data_accessibility_paraphrased, rag_chain, 'hybrid_structured_char_amateur')

    # Model 4: Hybrid RAG + structured text + regex-based chunking
    chunks = get_chunks(md_file_path='doc.md', structure_loader=True, separators=[r"(?=\n\d{4}(?:\.\d+)+\s)"])
    ensemble_retriever = get_retrievers(chunks)
    rag_chain = get_rag_chain(ensemble_retriever, template)
    get_response(qa_data_accessibility, rag_chain, 'hybrid_structured_regex_expert')
    get_response(qa_data_accessibility_paraphrased, rag_chain, 'hybrid_structured_regex_amateur')


     # Model 5: Naive RAG + raw text + char-based chunking
    chunks = get_chunks(md_file_path='doc.md', structure_loader=False, separators=None)
    retriever = get_naive_vector_retriever(chunks)
    rag_chain = get_rag_chain(retriever, template)
    get_response(qa_data_accessibility, rag_chain, retriever, 'naive_raw_char_expert')
    get_response(qa_data_accessibility_paraphrased, rag_chain, retriever, 'naive_raw_char_amateur')
