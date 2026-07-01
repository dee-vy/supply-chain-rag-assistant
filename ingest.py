# loads docs into ChromaDB


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# Paths
DATA_PATH = "data/supply_chain.txt"
CHROMA_PATH = "chroma_db"


def ingest_documents():
    print("Loading documents...")
    loader = TextLoader(DATA_PATH)
    documents = loader.load()

    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    print("Creating embeddings and storing in ChromaDB...")
    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    print(f"Done! {len(chunks)} chunks stored in ChromaDB.")
    return db


if __name__ == "__main__":
    ingest_documents()
