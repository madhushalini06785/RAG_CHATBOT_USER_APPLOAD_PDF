import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone


# --------------------------------------------------
# INGEST UPLOADED PDF
# --------------------------------------------------

def ingest_pdf(
    pdf_path,
    pinecone_api_key,
    pinecone_index,
    namespace
):

    # --------------------------------------------------
    # 1. Connect to Pinecone
    # --------------------------------------------------

    print("Connecting to Pinecone...")

    pc = Pinecone(
        api_key=pinecone_api_key
    )

    index = pc.Index(
        pinecone_index
    )


    # --------------------------------------------------
    # 2. Load PDF
    # --------------------------------------------------

    print("Loading PDF...")

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            "Uploaded PDF could not be found."
        )

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    print(
        f"PDF loaded successfully: {len(documents)} pages"
    )


    # --------------------------------------------------
    # 3. Split PDF into chunks
    # --------------------------------------------------

    print("Splitting document...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    splits = splitter.split_documents(
        documents
    )

    print(
        f"Created {len(splits)} chunks"
    )


    # --------------------------------------------------
    # 4. Create embeddings
    # --------------------------------------------------

    print("Creating embeddings...")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    texts = [
        doc.page_content
        for doc in splits
    ]

    embeddings = embedding_model.embed_documents(
        texts
    )


    # --------------------------------------------------
    # 5. Prepare vectors
    # --------------------------------------------------

    print("Preparing vectors...")

    vectors = []

    for i, (doc, emb) in enumerate(
        zip(splits, embeddings)
    ):

        metadata = {
            "text": doc.page_content,
            "page": doc.metadata.get(
                "page",
                "unknown"
            ),
            "source": os.path.basename(
                pdf_path
            )
        }

        vectors.append(
            {
                "id": f"{namespace}-chunk-{i}",
                "values": emb,
                "metadata": metadata
            }
        )


    # --------------------------------------------------
    # 6. Upload vectors to Pinecone
    # --------------------------------------------------

    print("Uploading vectors to Pinecone...")

    index.upsert(
        vectors=vectors,
        namespace=namespace
    )


    print(
        "SUCCESS: PDF indexed successfully!"
    )


    # --------------------------------------------------
    # 7. Return statistics
    # --------------------------------------------------

    return len(documents), len(splits)