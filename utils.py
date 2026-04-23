import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings


# ✅ PDF Processing (RAG)
def process_pdf(uploaded_file):
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
    )

    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore


# ✅ Create RAG Chain (LOCAL MODEL - NO ERRORS)
def create_chain(vectorstore):
    from transformers import pipeline
    from langchain_community.llms import HuggingFacePipeline
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationalRetrievalChain

    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=512
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return qa_chain


# ✅ Guardrails
def is_safe_query(query):
    blocked = ["hack", "bypass", "ignore instructions", "malicious"]
    return not any(word in query.lower() for word in blocked)
