import os
from dotenv import load_dotenv
import streamlit as st

from pdf_processor import extract_text_from_pdf, chunk_text
from vector_store import FaissVectorStore
from chatbot import GeminiClient, QAEngine
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from typing import Optional

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")


def main():
    st.set_page_config(page_title="GenAI StudyMate — Dashboard", layout="wide")
    st.markdown(
        """
        <style>
        .main-title {font-size:28px; font-weight:700; color:#0f172a;}
        .card {background: white; padding: 16px; border-radius:8px; box-shadow: 0 4px 12px rgba(15,23,42,0.06);}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Top bar + theme
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("<div class='main-title'>GenAI StudyMate — Professional Dashboard</div>", unsafe_allow_html=True)
        st.markdown("Modern dashboard to generate notes, quizzes and chat with PDFs.")
    with col2:
        theme = st.selectbox("Theme", ["Light", "Dark"], index=0)
        if theme == "Dark":
            st.markdown(
                "<style>body{background:#0b1220;color:#e6eef8} .card{background:#071029;color:#e6eef8}</style>",
                unsafe_allow_html=True,
            )

    # Sidebar navigation and upload
    with st.sidebar:
        st.markdown("## Navigation")
        page = st.radio("Go to", ["Home", "Upload", "Chat", "Notes", "Quiz", "Dashboard"], index=0)
        st.markdown("---")
        st.markdown("## Upload & Project")
        uploaded = st.file_uploader("Upload a PDF", type=["pdf"], key="uploader")
        st.markdown("---")
        st.markdown("**API Key**")
        st.write("Using GEMINI_API_KEY from .env")
        st.markdown("---")
        st.markdown("**Quick Actions**")
        if st.button("Run connectivity test"):
            st.write("Run test via test_gemini_key.py in terminal")

    # Main content area layout
    left, right = st.columns([2, 1])

    def ingest_pdf(uploaded_file) -> Optional[str]:
        if not uploaded_file:
            return None
        with st.spinner("Extracting text..."):
            text = extract_text_from_pdf(uploaded_file)
        st.success("Text extracted")
        st.write(f"Document length: {len(text)} characters")
        chunks = chunk_text(text, chunk_size=1000, overlap=200, max_chunks=400)
        client = GeminiClient(api_key=API_KEY)
        embeddings = client.embed_texts(chunks)
        # Initialize Faiss with the embedding dimension returned by the SDK
        embedding_dim = len(embeddings[0]) if embeddings and len(embeddings[0]) > 0 else 1536
        store = FaissVectorStore(embedding_dim=embedding_dim)
        store.add_documents(chunks, embeddings)
        st.session_state["store"] = store
        st.session_state["chunks"] = chunks
        st.success("Document ingested into vector store")
        return text

    # Page rendering
    if page == "Home":
        with left:
            st.markdown("### Welcome")
            st.markdown("Upload a PDF and use the study tools to generate notes, quizzes, and chat with your document.")
        with right:
            st.markdown("### Quick Status")
            st.write("SDK:", getattr(GeminiClient(api_key=API_KEY), '_genai_name', None))
    elif page == "Upload":
        with left:
            st.markdown("### PDF Upload & Ingestion")
            st.markdown('<div class="card">', unsafe_allow_html=True)
            if uploaded:
                ingest_pdf(uploaded)
            else:
                st.info("Upload a PDF to get started")
            st.markdown('</div>', unsafe_allow_html=True)
        with right:
            st.markdown("### Project Dashboard")
            st.write("Status: ready")
    elif page == "Chat":
        with left:
            st.markdown("### Chat with PDF")
            if "store" in st.session_state:
                engine = QAEngine(gemini_client=GeminiClient(api_key=API_KEY), store=st.session_state["store"])
                q = st.text_input("Ask a question about the document", key="chat_q")
                if st.button("Send", key="send_chat") and q:
                    with st.spinner("Generating answer..."):
                        try:
                            ans = engine.answer_question(q)
                            st.markdown("**Answer:**")
                            st.write(ans)
                        except Exception as e:
                            st.error(f"Generation error: {e}")
            else:
                st.info("Upload a PDF to enable chat")
        with right:
            st.markdown("### Key Concepts")
            if "chunks" in st.session_state:
                try:
                    engine = QAEngine(gemini_client=GeminiClient(api_key=API_KEY), store=st.session_state["store"])
                    summary = engine.summarize()
                    lines = [ln.strip() for ln in summary.splitlines() if ln.strip()]
                    for ln in lines[:6]:
                        st.markdown(f"- {ln}")
                except Exception:
                    top_k = 5
                    concepts = st.session_state["chunks"][:top_k]
                    for c in concepts:
                        st.markdown(f"- {c[:80]}...")
            else:
                st.write("No document loaded")
    elif page == "Notes":
        with left:
            st.markdown("### Notes Generation")
            if "store" in st.session_state:
                engine = QAEngine(gemini_client=GeminiClient(api_key=API_KEY), store=st.session_state["store"])
                if st.button("Generate Notes"):
                    with st.spinner("Creating notes..."):
                        try:
                            notes = engine.generate_notes()
                            st.subheader("Notes")
                            st.write(notes)
                        except Exception as e:
                            st.error(f"Notes generation error: {e}")
                        def notes_to_pdf_bytes(text: str) -> bytes:
                            buf = io.BytesIO()
                            p = canvas.Canvas(buf, pagesize=letter)
                            width, height = letter
                            margin = 40
                            y = height - margin
                            lines = text.splitlines()
                            p.setFont("Helvetica", 11)
                            for line in lines:
                                if y < margin:
                                    p.showPage()
                                    p.setFont("Helvetica", 11)
                                    y = height - margin
                                if len(line) > 100:
                                    while len(line) > 100:
                                        p.drawString(margin, y, line[:100])
                                        line = line[100:]
                                        y -= 14
                                    if line:
                                        p.drawString(margin, y, line)
                                        y -= 14
                                else:
                                    p.drawString(margin, y, line)
                                    y -= 14
                            p.save()
                            buf.seek(0)
                            return buf.read()
                        try:
                            pdf_bytes = notes_to_pdf_bytes(notes)
                            st.download_button("Download Notes as PDF", data=pdf_bytes, file_name="notes.pdf", mime="application/pdf")
                        except Exception:
                            st.warning("Could not generate PDF for download.")
            else:
                st.info("Upload a PDF to generate notes")
        with right:
            st.markdown("### Key Concepts")
            st.write("Use the Notes page to extract key concepts")
    elif page == "Quiz":
        with left:
            st.markdown("### Quiz Generation")
            if "store" in st.session_state:
                engine = QAEngine(gemini_client=GeminiClient(api_key=API_KEY), store=st.session_state["store"])
                if st.button("Generate Quiz"):
                    with st.spinner("Creating MCQs..."):
                        try:
                            mcqs = engine.generate_mcqs()
                            st.subheader("Quiz")
                            for i, mcq in enumerate(mcqs, 1):
                                st.markdown(f"**Q{i}. {mcq['question']}**")
                                for opt in mcq['options']:
                                    st.write(f"- {opt}")
                                st.write(f"**Answer:** {mcq['answer']}")
                        except Exception as e:
                            st.error(f"Quiz generation error: {e}")
            else:
                st.info("Upload a PDF to generate quizzes")
        with right:
            st.markdown("### Project Dashboard")
            st.write("Status: ready")
            st.write(f"SDK: {getattr(GeminiClient(api_key=API_KEY), '_genai_name', None)}")


if __name__ == "__main__":
    main()
