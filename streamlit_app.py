import streamlit as st
import os
import tempfile
import uuid

from dotenv import load_dotenv

from rag_chain import get_rag_chain
from ingest import ingest_pdf

load_dotenv()


# ---------------------------------------
# LOAD SECRETS
# ---------------------------------------

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")


# ---------------------------------------
# PAGE CONFIG
# ---------------------------------------

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="🤖"
)

st.title("📚 AI Document Assistant")
st.write("Upload a PDF and ask questions about it.")


# ---------------------------------------
# SESSION STATE
# ---------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "namespace" not in st.session_state:
    st.session_state.namespace = None

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None


# ---------------------------------------
# PDF UPLOAD
# ---------------------------------------

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


if uploaded_file is not None:

    # Process only when a NEW PDF is uploaded
    if st.session_state.uploaded_file_name != uploaded_file.name:

        st.session_state.uploaded_file_name = uploaded_file.name

        # Create unique namespace
        namespace = "pdf-" + uuid.uuid4().hex

        st.session_state.namespace = namespace

        # Save uploaded PDF temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(
                uploaded_file.getbuffer()
            )

            temp_path = temp_file.name

        # ---------------------------------------
        # INGEST PDF
        # ---------------------------------------

        with st.spinner("Processing your PDF... 📄"):

            pages, chunks = ingest_pdf(
                temp_path,
                PINECONE_API_KEY,
                PINECONE_INDEX,
                namespace
            )

        # Delete temporary file
        os.remove(temp_path)

        # ---------------------------------------
        # CREATE RAG CHAIN
        # ---------------------------------------

        st.session_state.rag_chain = get_rag_chain(
            PINECONE_API_KEY,
            PINECONE_INDEX,
            namespace
        )

        # Clear previous conversation
        st.session_state.messages = []

        st.success(
            f"PDF processed successfully! "
            f"{pages} pages and {chunks} chunks created."
        )


# ---------------------------------------
# CLEAR CHAT
# ---------------------------------------

if st.button("🔄 Clear Chat"):

    st.session_state.messages = []

    st.rerun()


# ---------------------------------------
# DISPLAY CHAT HISTORY
# ---------------------------------------

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])


# ---------------------------------------
# USER INPUT
# ---------------------------------------

user_prompt = st.chat_input(
    "Ask a question about your document..."
)


if user_prompt:

    # Make sure PDF was uploaded
    if st.session_state.rag_chain is None:

        st.warning(
            "Please upload a PDF first."
        )

    else:

        # Display user message
        st.chat_message("user").markdown(user_prompt)

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_prompt
            }
        )

        # Generate answer
        with st.chat_message("assistant"):

            with st.spinner("Thinking... 🤖"):

                response, sources = (
                    st.session_state.rag_chain(
                        user_prompt
                    )
                )

            st.markdown(response)

            # Display sources
            if sources:

                st.markdown("### 📖 Sources")

                for src in sources:

                    st.info(src)

        # Save assistant response
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response
            }
        )