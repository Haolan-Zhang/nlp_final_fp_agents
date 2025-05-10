from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from hybrid_rag import get_rag_chain, get_response, get_chunks



def get_naive_vector_retriever(chunks):
    """
    Create a naive vector-based retriever without keyword fusion.
    
    :param chunks: List of Document objects.
    :return: Retriever object.
    """
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever()
    return retriever

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


    # Model 5: Naive RAG + raw text + char-based chunking
    chunks = get_chunks(md_file_path='doc.md', structure_loader=False, separators=None)
    retriever = get_naive_vector_retriever(chunks)
    rag_chain = get_rag_chain(retriever, template)
    get_response(qa_data_accessibility, rag_chain, retriever, 'naive_raw_char_expert')
    get_response(qa_data_accessibility_paraphrased, rag_chain, retriever, 'naive_raw_char_amateur')
