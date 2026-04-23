import streamlit as st
from utils import process_pdf, create_chain, is_safe_query
from dotenv import load_dotenv
import os

load_dotenv()
st.set_page_config(page_title="AI Study Assistant Pro", layout="wide")
st.title("🚀 AI Study Assistant Pro ")

# Session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# Upload PDF
uploaded_file = st.file_uploader("📄 Upload your PDF", type="pdf")

if uploaded_file:
    with st.spinner("Processing PDF..."):
        st.session_state.vectorstore = process_pdf(uploaded_file)
        st.session_state.qa_chain = create_chain(st.session_state.vectorstore)
    st.success("✅ PDF processed! Ask your questions.")

# Chat input
user_input = st.chat_input("Ask something from your PDF...")

if user_input:
    if not is_safe_query(user_input):
        st.warning("⚠️ Unsafe or irrelevant query detected!")
    elif st.session_state.qa_chain is None:
        st.warning("⚠️ Please upload a PDF first.")
    else:
        response = st.session_state.qa_chain.run(user_input)

        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("AI", response))

# Display chat
for role, msg in st.session_state.chat_history:
    if role == "You":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(msg)