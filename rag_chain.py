import os

from dotenv import load_dotenv
from pinecone import Pinecone

from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA


# --------------------------------------------------
# ENVIRONMENT VARIABLES
# --------------------------------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# --------------------------------------------------
# EMBEDDING MODEL
# --------------------------------------------------

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# PROMPT
# --------------------------------------------------

template = """
You are an AI assistant that answers questions strictly using the document.

Rules:
1. Only use the provided context.
2. Do NOT use outside knowledge.
3. If the answer is missing, say:
"I could not find the answer in the document."

Context:
{context}

Question:
{question}

Answer clearly and briefly.
"""

prompt = ChatPromptTemplate.from_template(
    template
)


# --------------------------------------------------
# CREATE RAG CHAIN
# --------------------------------------------------

def get_rag_chain(
    pinecone_api_key,
    pinecone_index,
    namespace
):

    # ----------------------------------------------
    # Connect to Pinecone
    # ----------------------------------------------

    pc = Pinecone(
        api_key=pinecone_api_key
    )

    index = pc.Index(
        pinecone_index
    )


    # ----------------------------------------------
    # Create vector store
    # ----------------------------------------------

    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embedding,
        namespace=namespace
    )


    # ----------------------------------------------
    # Create retriever
    # ----------------------------------------------

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 4
        }
    )


    # ----------------------------------------------
    # Create LLM
    # ----------------------------------------------

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="openai/gpt-oss-20b",
        temperature=0
    )


    # ----------------------------------------------
    # Create RetrievalQA chain
    # ----------------------------------------------

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt
        }
    )


    # ----------------------------------------------
    # Function to ask questions
    # ----------------------------------------------

    def ask(query):

        result = qa_chain.invoke(
            {
                "query": query
            }
        )

        answer = result["result"]

        docs = result["source_documents"]


        # ------------------------------------------
        # Extract source pages
        # ------------------------------------------

        pages = set()

        for doc in docs:

            if "page" in doc.metadata:

                page = doc.metadata["page"]

                if isinstance(page, int):

                    pages.add(page + 1)

                else:

                    try:
                        pages.add(
                            int(page) + 1
                        )
                    except:
                        pass


        # ------------------------------------------
        # Create source information
        # ------------------------------------------

        sources = []

        if pages:

            for page in sorted(pages):

                sources.append(
                    f"📄 Page {page}"
                )


        return answer, sources


    return ask