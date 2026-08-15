import os
import hashlib

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from pinecone import Pinecone


# ==================================================
# LOAD EMBEDDING MODEL ONCE
# ==================================================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ==================================================
# INGEST PDF
# ==================================================

def ingest_pdf(
    pdf_path,
    pinecone_api_key,
    pinecone_index,
    namespace,
    progress_callback=None
):

    # ==================================================
    # CONNECT TO PINECONE
    # ==================================================

    pc = Pinecone(
        api_key=pinecone_api_key
    )

    index = pc.Index(
        pinecone_index
    )


    # ==================================================
    # CHECK FILE
    # ==================================================

    if not os.path.exists(pdf_path):

        raise FileNotFoundError(
            "PDF file not found."
        )


    # ==================================================
    # READ PDF
    # ==================================================

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


    # ==================================================
    # SPLIT DOCUMENT
    # ==================================================

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


    if total_chunks == 0:

        raise ValueError(
            "No readable text was found in the PDF."
        )


    # ==================================================
    # PREPARE TEXT
    # ==================================================

    texts = [
        doc.page_content
        for doc in splits
    ]


    # ==================================================
    # GET ORIGINAL FILE NAME
    # ==================================================

    filename = os.path.basename(
        pdf_path
    )


    # ==================================================
    # CREATE HASH FROM ACTUAL PDF CONTENT
    # ==================================================

    with open(
        pdf_path,
        "rb"
    ) as f:

        file_hash = hashlib.sha256(
            f.read()
        ).hexdigest()[:12]


    # ==================================================
    # GENERATE EMBEDDINGS
    # ==================================================

    embedding_batch_size = 64

    vectors = []


    for start in range(
        0,
        total_chunks,
        embedding_batch_size
    ):

        end = min(
            start + embedding_batch_size,
            total_chunks
        )


        batch_texts = texts[
            start:end
        ]


        # ----------------------------------------------
        # Generate embeddings
        # ----------------------------------------------

        batch_embeddings = (
            embedding_model.embed_documents(
                batch_texts
            )
        )


        # ----------------------------------------------
        # Create Pinecone vectors
        # ----------------------------------------------

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

                "text":
                    doc.page_content,

                "page":
                    doc.metadata.get(
                        "page",
                        "unknown"
                    ),

                "source":
                    filename
            }


            vectors.append(
                {

                    "id":
                        f"{file_hash}-"
                        f"chunk-{global_index}",

                    "values":
                        emb,

                    "metadata":
                        metadata
                }
            )


        # ----------------------------------------------
        # Update progress
        # ----------------------------------------------

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


    # ==================================================
    # UPLOAD TO PINECONE
    # ==================================================

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


        # ----------------------------------------------
        # Update upload progress
        # ----------------------------------------------

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


    # ==================================================
    # COMPLETE
    # ==================================================

    if progress_callback:

        progress_callback(
            1.0,
            (
                f"✅ {filename} processed "
                f"successfully"
            )
        )


    return pages, total_chunks
