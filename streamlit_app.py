import os
import tempfile
import uuid

import streamlit as st

from dotenv import load_dotenv

from rag_chain import get_rag_chain
from ingest import ingest_pdf

from database import (
    create_chat,
    get_all_chats,
    get_messages,
    get_documents,
    update_chat_title,
    save_message,
    save_document,
    clear_messages
)


# ==================================================
# ENVIRONMENT
# ==================================================

load_dotenv()

PINECONE_API_KEY = os.getenv(
    "PINECONE_API_KEY"
)

PINECONE_INDEX = os.getenv(
    "PINECONE_INDEX"
)


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(

    page_title="AI Document Assistant",

    page_icon="🤖",

    layout="wide"
)


# ==================================================
# DATABASE → SESSION STATE
# ==================================================

def load_chats():

    db_chats = get_all_chats()

    chats = {}

    for chat in db_chats:

        chats[
            chat["chat_id"]
        ] = {

            "title":
                chat["title"],

            "messages":
                get_messages(
                    chat["chat_id"]
                ),

            "namespace":
                chat["namespace"],

            "files":
                get_documents(
                    chat["chat_id"]
                ),

            "rag_chain":
                None
        }

    return chats


# ==================================================
# INITIAL SESSION STATE
# ==================================================

if "chats" not in st.session_state:

    st.session_state.chats = (
        load_chats()
    )


# ==================================================
# CREATE FIRST CHAT
# ==================================================

if not st.session_state.chats:

    chat_id = str(
        uuid.uuid4()
    )

    namespace = (
        "chat-" + chat_id
    )

    create_chat(

        chat_id,

        "New Chat",

        namespace
    )


    st.session_state.chats = {

        chat_id: {

            "title":
                "New Chat",

            "messages":
                [],

            "namespace":
                namespace,

            "files":
                [],

            "rag_chain":
                None
        }
    }


    st.session_state.current_chat = (
        chat_id
    )


# ==================================================
# RESTORE CURRENT CHAT
# ==================================================

elif "current_chat" not in st.session_state:

    st.session_state.current_chat = (
        next(
            iter(
                st.session_state.chats
            )
        )
    )


# ==================================================
# CREATE NEW CHAT
# ==================================================

def create_new_chat():

    chat_id = str(
        uuid.uuid4()
    )

    namespace = (
        "chat-" + chat_id
    )


    create_chat(

        chat_id,

        "New Chat",

        namespace
    )


    st.session_state.chats[
        chat_id
    ] = {

        "title":
            "New Chat",

        "messages":
            [],

        "namespace":
            namespace,

        "files":
            [],

        "rag_chain":
            None
    }


    st.session_state.current_chat = (
        chat_id
    )


# ==================================================
# CURRENT CHAT
# ==================================================

current_chat_id = (
    st.session_state.current_chat
)


current_chat = (
    st.session_state.chats[
        current_chat_id
    ]
)


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.title(
        "📚 AI Assistant"
    )


    st.divider()


    # ==================================================
    # NEW CHAT
    # ==================================================

    if st.button(

        "➕ New Chat",

        use_container_width=True
    ):

        create_new_chat()

        st.rerun()


    st.divider()


    # ==================================================
    # CHAT HISTORY
    # ==================================================

    st.subheader(
        "💬 Chats"
    )


    for chat_id, chat in (
        st.session_state.chats.items()
    ):

        title = chat["title"]


        if title == "":

            title = "New Chat"


        if chat_id == current_chat_id:

            button_text = (
                f"🟢 {title}"
            )

        else:

            button_text = (
                f"💬 {title}"
            )


        if st.button(

            button_text,

            key=f"chat_{chat_id}",

            use_container_width=True
        ):

            st.session_state.current_chat = (
                chat_id
            )

            st.rerun()


    st.divider()


    # ==================================================
    # DOCUMENTS
    # ==================================================

    st.subheader(
        "📄 Documents"
    )


    if current_chat["files"]:

        for file_info in (
            current_chat["files"]
        ):

            st.caption(
                f"📄 {file_info['name']}"
            )

    else:

        st.caption(
            "No documents uploaded."
        )


    st.divider()


    # ==================================================
    # CLEAR CHAT
    # ==================================================

    if st.button(

        "🗑️ Clear Current Chat",

        use_container_width=True
    ):

        clear_messages(
            current_chat_id
        )


        current_chat[
            "messages"
        ] = []


        st.rerun()


# ==================================================
# MAIN PAGE
# ==================================================

st.title(
    "📚 AI Document Assistant"
)

st.write(
    "Upload one or more PDFs and ask questions about them."
)


# ==================================================
# PDF UPLOADER
# ==================================================

uploaded_files = st.file_uploader(

    "Upload your PDFs",

    type=["pdf"],

    accept_multiple_files=True,

    key=f"uploader_{current_chat_id}"
)


# ==================================================
# PROCESS PDFs
# ==================================================

if uploaded_files:

    processed_files = {

        (
            file_info["name"],

            file_info["size"]
        )

        for file_info in current_chat[
            "files"
        ]
    }


    new_files = [

        file

        for file in uploaded_files

        if (
            file.name,
            file.size
        ) not in processed_files
    ]


    if new_files:

        st.subheader(
            "📚 Processing Documents"
        )


        overall_progress = (
            st.progress(0)
        )


        status_text = st.empty()


        total_files = len(
            new_files
        )


        for file_number, uploaded_file in enumerate(

            new_files,

            start=1
        ):

            status_text.write(

                f"📄 Processing "
                f"{file_number}/{total_files}: "
                f"**{uploaded_file.name}**"
            )


            # ==================================================
            # TEMPORARY FILE
            # ==================================================

            with tempfile.NamedTemporaryFile(

                delete=False,

                suffix=".pdf"
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getbuffer()
                )

                temp_path = (
                    temp_file.name
                )


            try:

                # ==================================================
                # PROGRESS
                # ==================================================

                def update_progress(
                    progress,
                    message
                ):

                    current_progress = (

                        (
                            file_number - 1
                        )
                        / total_files

                    ) + (

                        progress
                        / total_files
                    )


                    overall_progress.progress(

                        min(
                            current_progress,
                            1.0
                        )
                    )


                    status_text.write(

                        f"📄 "
                        f"{file_number}/{total_files} — "
                        f"{uploaded_file.name}"
                        f"<br>{message}",

                        unsafe_allow_html=True
                    )


                # ==================================================
                # INGEST
                # ==================================================

                pages, chunks = ingest_pdf(

                    temp_path,

                    PINECONE_API_KEY,

                    PINECONE_INDEX,

                    current_chat[
                        "namespace"
                    ],

                    progress_callback=(
                        update_progress
                    )
                )


                # ==================================================
                # SAVE DATABASE
                # ==================================================

                save_document(

                    current_chat_id,

                    uploaded_file.name,

                    uploaded_file.size,

                    pages,

                    chunks
                )


                # ==================================================
                # UPDATE SESSION STATE
                # ==================================================

                current_chat[
                    "files"
                ].append({

                    "name":
                        uploaded_file.name,

                    "size":
                        uploaded_file.size,

                    "pages":
                        pages,

                    "chunks":
                        chunks
                })


            finally:

                if os.path.exists(
                    temp_path
                ):

                    os.remove(
                        temp_path
                    )


        # ==================================================
        # CREATE RAG CHAIN
        # ==================================================

        current_chat[
            "rag_chain"
        ] = get_rag_chain(

            PINECONE_API_KEY,

            PINECONE_INDEX,

            current_chat[
                "namespace"
            ]
        )


        overall_progress.progress(
            1.0
        )


        status_text.success(
            "🎉 All documents processed successfully!"
        )


# ==================================================
# DISPLAY CHAT HISTORY
# ==================================================

for message in current_chat[
    "messages"
]:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ==================================================
# CHAT INPUT
# ==================================================

user_prompt = st.chat_input(

    "Ask something about your documents..."
)


if user_prompt:

    # ==================================================
    # CHECK DOCUMENTS
    # ==================================================

    if not current_chat["files"]:

        st.warning(
            "Please upload at least one PDF first."
        )

        st.stop()


    # ==================================================
    # CREATE RAG CHAIN
    # ==================================================

    if current_chat[
        "rag_chain"
    ] is None:

        current_chat[
            "rag_chain"
        ] = get_rag_chain(

            PINECONE_API_KEY,

            PINECONE_INDEX,

            current_chat[
                "namespace"
            ]
        )


    # ==================================================
    # CREATE CHAT TITLE
    # ==================================================

    if current_chat[
        "title"
    ] == "New Chat":

        title = (
            user_prompt.strip()
        )


        if len(title) > 30:

            title = (
                title[:30]
                + "..."
            )


        current_chat[
            "title"
        ] = title


        update_chat_title(

            current_chat_id,

            title
        )


    # ==================================================
    # BUILD CHAT HISTORY
    # ==================================================

    chat_history = ""


    for message in current_chat[
        "messages"
    ]:

        chat_history += (

            f'{message["role"]}: '

            f'{message["content"]}\n'
        )


    # ==================================================
    # DISPLAY USER MESSAGE
    # ==================================================

    with st.chat_message(
        "user"
    ):

        st.markdown(
            user_prompt
        )


    # ==================================================
    # SAVE USER MESSAGE
    # ==================================================

    current_chat[
        "messages"
    ].append({

        "role":
            "user",

        "content":
            user_prompt
    })


    save_message(

        current_chat_id,

        "user",

        user_prompt
    )


    # ==================================================
    # GENERATE ANSWER
    # ==================================================

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Thinking... 🤖"
        ):

            response, sources = (

                current_chat[
                    "rag_chain"
                ](

                    user_prompt,

                    chat_history
                )
            )


        st.markdown(
            response
        )


        # ==================================================
        # SOURCES
        # ==================================================

        if sources:

            st.markdown(
                "### 📖 Sources"
            )


            for source in sources:

                st.info(
                    source
                )


    # ==================================================
    # SAVE ASSISTANT RESPONSE
    # ==================================================

    current_chat[
        "messages"
    ].append({

        "role":
            "assistant",

        "content":
            response
    })


    save_message(

        current_chat_id,

        "assistant",

        response
    )
