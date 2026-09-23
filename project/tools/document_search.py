from langchain_core.tools import tool
from project.services.vector_store import get_retriever

@tool
def document_search(query: str) -> str:
    """CRITICAL: Use this tool FIRST for ANY questions about company policies (like PTO, HR rules), 
    internal documents, local knowledge base, or uploaded PDFs/articles. 
    Never use web search for company-specific or internal document questions.
    
    Args:
        query: The search query or keywords to look up in the local corpus.
    """
    retriever = get_retriever()
    if not retriever:
        return "Error: Vector store could not be initialized or is empty."
        
    try:
        # Retrieve relevant documents
        docs = retriever.invoke(query)
        
        if not docs:
            return "No relevant documents found for your query in the corpus."
            
        # Format the output with metadata/citations
        formatted_results = []
        for i, doc in enumerate(docs):
            # FIX: our Chunker stores 'filename' in metadata, not 'source'
            source = doc.metadata.get("filename") or doc.metadata.get("source", "Unknown Source")
            content = doc.page_content.replace("\n", " ") # Clean up newlines for cleaner output
            formatted_results.append(f"[Citation {i+1}] Source: {source}\nContent: {content}\n")
            
        return "\n".join(formatted_results)
    except Exception as e:
        return f"Error during document retrieval: {str(e)}"
