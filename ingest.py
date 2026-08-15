import os
import hashlib

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone


# --------------------------------------------------
# LOAD EMBEDDING MODEL ONCE
# --------------------------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# INGEST PDF
# --------------------------------------------------

def ingest_pdf(
    pdf_path,
    pinecone_api_key,
    pinecone_index,
    namespace,
    progress_callback=None
):

    # --------------------------------------------------
    # 1. Connect to Pinecone
    # --------------------------------------------------

    pc = Pinecone(
        api_key=pinecone_api_key
    )

    index = pc.Index(
        pinecone_index
    )


    # --------------------------------------------------
    # 2. Check PDF
    # --------------------------------------------------

    if not os.path.exists(pdf_path):

        raise FileNotFoundError(
            "PDF file not found."
        )


    # --------------------------------------------------
    # 3. Load PDF
    # --------------------------------------------------

    if progress_callback:
        progress_callback(
            0.05,
            "📖 Reading PDF..."
        )

    loader = PyPDFLoader(
        pdf_path
    )

    documents = loader.load()

    pages = len(documents)


    # --------------------------------------------------
    # 4. Split PDF
    # --------------------------------------------------

    if progress_callback:
        progress_callback(
            0.15,
            f"✂️ Splitting {pages} pages..."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    splits = splitter.split_documents(
        documents
    )

    total_chunks = len(splits)


    # --------------------------------------------------
    # 5. Prepare text
    # --------------------------------------------------

    texts = [
        doc.page_content
        for doc in splits
    ]


    # --------------------------------------------------
    # 6. File name
    # --------------------------------------------------

    filename = os.path.basename(
        pdf_path
    )

    file_hash = hashlib.sha256(
        filename.encode()
    ).hexdigest()[:10]


    # --------------------------------------------------
    # 7. Generate embeddings in batches
    # --------------------------------------------------

    batch_size = 64

    vectors = []

    for start in range(
        0,
        total_chunks,
        batch_size
    ):

        end = min(
            start + batch_size,
            total_chunks
        )

        batch_texts = texts[
            start:end
        ]

        # Generate embeddings
        batch_embeddings = (
            embedding_model.embed_documents(
                batch_texts
            )
        )


        # Create vectors
        for local_index, emb in enumerate(
            batch_embeddings
        ):

            global_index = (
                start + local_index
            )

            doc = splits[
                global_index
            ]

            metadata = {

                "text": doc.page_content,

                "page": doc.metadata.get(
                    "page",
                    "unknown"
                ),

                "source": filename
            }


            vectors.append(
                {
                    "id": (
                        f"{file_hash}-"
                        f"chunk-{global_index}"
                    ),

                    "values": emb,

                    "metadata": metadata
                }
            )


        # Progress: 20% → 75%
        embedding_progress = (
            end / total_chunks
        )

        progress = (
            0.20 +
            embedding_progress * 0.55
        )

        if progress_callback:

            progress_callback(
                progress,
                (
                    f"🧠 Creating embeddings: "
                    f"{end}/{total_chunks} chunks"
                )
            )


    # --------------------------------------------------
    # 8. Upload to Pinecone in batches
    # --------------------------------------------------

    pinecone_batch_size = 100

    total_vectors = len(
        vectors
    )

    for start in range(
        0,
        total_vectors,
        pinecone_batch_size
    ):

        end = min(
            start + pinecone_batch_size,
            total_vectors
        )

        batch = vectors[
            start:end
        ]

        index.upsert(
            vectors=batch,
            namespace=namespace
        )


        upload_progress = (
            end / total_vectors
        )

        progress = (
            0.75 +
            upload_progress * 0.20
        )

        if progress_callback:

            progress_callback(
                progress,
                (
                    f"☁️ Uploading to Pinecone: "
                    f"{end}/{total_vectors}"
                )
            )


    # --------------------------------------------------
    # 9. Complete
    # --------------------------------------------------

    if progress_callback:

        progress_callback(
            1.0,
            (
                f"✅ {filename} processed "
                f"successfully"
            )
        )


    return pages, total_chunks
