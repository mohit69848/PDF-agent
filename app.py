import asyncio
import nest_asyncio
import streamlit as st
import tempfile
import re
from agent import PDFQAAgent

# Setup asyncio for Streamlit
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
nest_asyncio.apply()

# Streamlit page config
st.set_page_config(page_title="PDF QA", page_icon="✅", layout="centered")
st.title("📄 PDF Question-Answering Agent")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None

# Initialize the agent once
if st.session_state.agent is None:
    st.session_state.agent = PDFQAAgent()

agent = st.session_state.agent

# PDF Upload Section
uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(uploaded.read())
        st.session_state.pdf_path = tmp.name

    st.info("📥 Processing PDF... this may take some time for large files")
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(current, total):
        progress = int((current / total) * 100)
        progress_bar.progress(progress)
        status_text.text(f"Processing chunk {current} of {total}...")

    try:
        count = agent.ingest(st.session_state.pdf_path, progress_callback=update_progress)
        progress_bar.progress(100)
        status_text.text(f"✅ Ingested {count} chunks from PDF!")
    except Exception as e:
        st.error(f"⚠️ Failed to ingest PDF: {str(e)}")
        st.session_state.pdf_path = None

# Ask question section
question = st.text_input("Ask a question about the PDF:")

if st.button("Get Answer") and question:
    if not st.session_state.pdf_path:
        st.warning("⚠️ Please upload a PDF first!")
    else:
        with st.spinner("🤔 Generating answer... this may take a few seconds"):
            try:
                res = agent.answer(question, top_k=10)
            except Exception as e:
                st.error(f"⚠️ Failed to generate answer: {str(e)}")
                res = None

        if res:
            answer = res.get("answer", "⚠️ No summary available.")
            sources = res.get("sources", [])

            # Append to history
            st.session_state.history.append({
                "question": question,
                "answer": answer,
                "sources": sources
            })

# Display chat history
if st.session_state.history:
    st.subheader("💬 Chat History")
    for idx, chat in enumerate(reversed(st.session_state.history)):
        st.markdown(f"**Q:** {chat['question']}")
        st.markdown(f"**A:** {chat['answer']}")
        
        if chat["sources"]:
            st.markdown("**Sources:**")
            for d in chat["sources"]:
                page_num = d.metadata.get("page_number", "N/A")
                snippet = d.page_content[:300].replace("\n", " ")
                st.markdown(f"- Page {page_num}: `{snippet}...`")

        st.write("---")
