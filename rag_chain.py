import os

import streamlit as st

from dotenv import load_dotenv
from pinecone import Pinecone

from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain.prompts import ChatPromptTemplate


# ==================================================
# ENVIRONMENT
# ==================================================

load_dotenv()

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)


# ==================================================
# EMBEDDING MODEL
# ==================================================

@st.cache_resource
def get_embedding_model():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ==================================================
# ANSWER PROMPT
# ==================================================

answer_template = """

You are an AI assistant that answers questions
strictly using the uploaded documents.

Rules:

1. Use only the provided document context.
2. Do NOT use outside knowledge.
3. If the answer cannot be found in the documents, say:
"I could not find the answer in the uploaded documents."
4. Understand follow-up questions using the conversation history.
5. Give detailed answers when the user asks for more detail.
6. Do not invent information.
7. If the user asks a follow-up question, use previous
   conversation to understand what they are referring to.

Conversation History:

{chat_history}

Context from Documents:

{context}

Question:

{question}

Answer clearly and accurately:
"""


answer_prompt = ChatPromptTemplate.from_template(
    answer_template
)


# ==================================================
# QUESTION REWRITING PROMPT
# ==================================================

rewrite_template = """

You are a question-rewriting assistant for a
document question-answering system.

Your task is to convert the user's latest question
into a standalone question.

Use the conversation history to understand references
such as:

- it
- this
- that
- these
- those
- its
- they
- them
- explain more
- give me in depth
- tell me more
- why
- how
- examples
- advantages
- disadvantages
- applications
- explain the first one
- explain the above

Example:

Conversation:

User: What is computer vision?

Assistant: Computer vision is a field...

User: Give me in depth.

Standalone question:

Give an in-depth explanation of computer vision.

---

Conversation:

User: What are CNNs?

Assistant: CNNs are...

User: Explain its architecture.

Standalone question:

Explain the architecture of CNNs.

---

Conversation:

User: What is object detection?

Assistant: ...

User: What are its advantages?

Standalone question:

What are the advantages of object detection?

---

Conversation:

{chat_history}

Latest user question:

{question}

Return ONLY the rewritten standalone question.

Do not answer the question.
"""


rewrite_prompt = ChatPromptTemplate.from_template(
    rewrite_template
)


# ==================================================
# CREATE RAG CHAIN
# ==================================================

def get_rag_chain(
    pinecone_api_key,
    pinecone_index,
    namespace
):

    # ==================================================
    # EMBEDDINGS
    # ==================================================

    embedding = get_embedding_model()


    # ==================================================
    # PINECONE
    # ==================================================

    pc = Pinecone(
        api_key=pinecone_api_key
    )

    index = pc.Index(
        pinecone_index
    )


    # ==================================================
    # VECTOR STORE
    # ==================================================

    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embedding,
        namespace=namespace
    )


    # ==================================================
    # RETRIEVER
    # ==================================================

    retriever = vectorstore.as_retriever(

        search_type="similarity",

        search_kwargs={
            "k": 5
        }
    )


    # ==================================================
    # GROQ
    # ==================================================

    llm = ChatGroq(

        groq_api_key=GROQ_API_KEY,

        model_name="openai/gpt-oss-20b",

        temperature=0
    )


    # ==================================================
    # CREATE CHAINS ONCE
    # ==================================================

    rewrite_chain = (
        rewrite_prompt
        | llm
    )

    answer_chain = (
        answer_prompt
        | llm
    )


    # ==================================================
    # ASK FUNCTION
    # ==================================================

    def ask(
        query,
        chat_history=None
    ):

        if chat_history is None:

            chat_history = ""


        # ==================================================
        # STEP 1 — REWRITE
        # ==================================================

        rewrite_result = rewrite_chain.invoke({

            "chat_history":
                chat_history,

            "question":
                query
        })


        standalone_question = (
            rewrite_result.content.strip()
        )


        # ==================================================
        # STEP 2 — RETRIEVE
        # ==================================================

        docs = retriever.invoke(
            standalone_question
        )


        # ==================================================
        # STEP 3 — CONTEXT
        # ==================================================

        context = "\n\n".join(

            doc.page_content

            for doc in docs
        )


        # ==================================================
        # STEP 4 — ANSWER
        # ==================================================

        answer_result = answer_chain.invoke({

            "chat_history":
                chat_history,

            "context":
                context,

            "question":
                standalone_question
        })


        answer = answer_result.content


        # ==================================================
        # STEP 5 — SOURCES
        # ==================================================

        sources = []

        seen = set()


        for doc in docs:

            source = doc.metadata.get(
                "source",
                "Unknown document"
            )

            page = doc.metadata.get(
                "page",
                None
            )


            if isinstance(page, int):

                page_number = (
                    page + 1
                )

            else:

                try:

                    page_number = (
                        int(page) + 1
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    page_number = None


            if page_number is not None:

                source_text = (
                    f"📄 {source} — "
                    f"Page {page_number}"
                )

            else:

                source_text = (
                    f"📄 {source}"
                )


            if source_text not in seen:

                sources.append(
                    source_text
                )

                seen.add(
                    source_text
                )


        return answer, sources


    return ask
