from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from utils.pdf_loader import extract_text_from_pdf
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def create_retriever_from_pdf(file_path: str):
    # 1. Extract text from PDF
    raw_text = extract_text_from_pdf(file_path)

    # 2. Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_text(raw_text)

    # 3. Convert to LangChain Documents
    docs = [Document(page_content=t) for t in texts]

    # 4. Load Gemini embeddings
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key="AIzaSyB3tcSYrrGlN55saJOOc5rBJ9D68BBYCQE"
    )

    # 5. Create Chroma vector store
    vectordb = Chroma.from_documents(docs, embedding_model)

    return vectordb.as_retriever()
