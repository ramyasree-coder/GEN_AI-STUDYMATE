import os
from dotenv import load_dotenv
import streamlit as st
import time

from pdf_processor import extract_text_from_pdf, chunk_text
from vector_store import FaissVectorStore
from chatbot import GeminiClient, QAEngine
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from typing import Optional
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False
from collections import Counter
import re
import string
try:
    from wordcloud import WordCloud
except Exception:
    WordCloud = None
import users

load_dotenv()


def safe_rerun():
    """Attempt to rerun the Streamlit script in a way that works across versions."""
    try:
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
            return
    except Exception:
        pass
    # Fallback: tweak query params to force a rerun
    try:
        st.experimental_set_query_params(_rerun=int(time.time()))
        return
    except Exception:
        pass
    # Final fallback: stop execution (user can interact to rerun)
    try:
        st.stop()
    except Exception:
        pass

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

    # Initialize user DB and session user
    users.init_db()
    if "user" not in st.session_state:
        st.session_state["user"] = None

    if st.session_state.get("user") is None:
        st.markdown("# Welcome to GenAI StudyMate")
        with st.form("auth_form"):
            mode = st.radio("Mode", ["Login", "Register"], index=0)
            uid = st.text_input("User ID")
            pwd = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Submit")
            if submitted:
                if not uid or not pwd:
                    st.error("Provide both user id and password")
                else:
                    if mode == "Register":
                        if users.user_exists(uid):
                            st.error("User already exists")
                        else:
                            ok = users.create_user(uid, pwd)
                            if ok:
                                st.success("User created — please login")
                            else:
                                st.error("Could not create user")
                    else:
                        if users.verify_user(uid, pwd):
                            st.session_state["user"] = uid
                            st.success(f"Logged in as {uid}")
                            # Continue rendering in this run; attempt a rerun for browsers that support it
                            safe_rerun()
                        else:
                            st.error("Invalid credentials")
        # If user is still not set, stop here. Otherwise continue rendering the app in this same run.
        if st.session_state.get("user") is None:
            return

    # Top bar + theme
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("<div class='main-title'>GenAI StudyMate — Professional Dashboard</div>", unsafe_allow_html=True)
        st.markdown("Modern dashboard to generate notes, quizzes and chat with PDFs.")
    # Theme selector (persist in session state). Streamlit will rerun when the widget value changes.
    if "theme" not in st.session_state:
        st.session_state["theme"] = "Light"

    with col2:
        theme = st.selectbox("Theme", ["Light", "Dark"], key="theme")
        if st.session_state.get("theme") == "Dark":
            # apply stronger CSS with !important to override Streamlit defaults
            st.markdown(
                """
                <style>
                html, body, .stApp {
                  background: #0b1220 !important;
                  color: #e6eef8 !important;
                }
                .stApp .card, .stApp .st-card {
                  background: #071029 !important;
                  color: #e6eef8 !important;
                }
                /* Buttons, inputs */
                .stApp button, .stApp .st-bx, .stApp .st-c1 {
                  color: #e6eef8 !important;
                }
                /* Links */
                a { color: #8fb3ff !important; }
                </style>
                """,
                unsafe_allow_html=True,
            )

    # Sidebar navigation and upload
    with st.sidebar:
        st.markdown("## Navigation")
        page = st.radio("Go to", ["Home", "Upload", "Chat", "Notes", "Quiz", "Explain", "Dashboard"], index=0)
        st.markdown("---")
        st.markdown("## Upload & Project")
        uploaded = st.file_uploader("Upload a PDF", type=["pdf"], key="uploader")
        st.markdown("---")
        st.markdown("**API Key**")
        st.write("Using GEMINI_API_KEY from .env")
        st.markdown("---")
        # Runtime dependency warning
        missing = []
        if not HAS_MATPLOTLIB:
            missing.append("matplotlib")
        if WordCloud is None:
            # wordcloud depends on Pillow
            missing.extend(["wordcloud", "Pillow"])
        if missing:
            pkg_cmd = "pip install " + " ".join(sorted(set(missing)))
            st.warning("Optional packages missing: " + ", ".join(sorted(set(missing))) + 
                       ". To enable charts/install run:\n" + pkg_cmd)
            st.markdown("Or install all requirements: `pip install -r requirements.txt`")
            # prepare downloadable install instructions
            req_text = """
Install missing optional packages to enable charts and wordclouds:

Run this command in your activated virtual environment:

%s

Or install full requirements:
pip install -r requirements.txt
""" % (pkg_cmd,)
            # append requirements.txt contents if available
            try:
                with open(os.path.join(os.path.dirname(__file__), "requirements.txt"), "r", encoding="utf-8") as rf:
                    reqs = rf.read()
                req_text += "\n---\nrequirements.txt:\n\n" + reqs
            except Exception:
                pass
            st.download_button("Download install instructions", data=req_text, file_name="install_instructions.txt", mime="text/plain")
            st.markdown("---")
        st.markdown("**Quick Actions**")
        if st.button("Run connectivity test"):
            st.write("Run test via test_gemini_key.py in terminal")
        # Per-user storage path and indices manager
        base_user_dir = os.path.join(os.path.dirname(__file__), "user_data", st.session_state.get("user"))
        indices_dir = os.path.join(base_user_dir, "indices")
        os.makedirs(indices_dir, exist_ok=True)
        st.markdown("---")
        st.markdown("## Storage")
        index_name = st.text_input("Index name", value="default")
        save_target = os.path.join(indices_dir, index_name)
        if st.button("Save Index to my workspace"):
            if "store" in st.session_state:
                try:
                    st.session_state["store"].save(save_target)
                    st.success(f"Saved index '{index_name}'")
                except Exception as e:
                    st.error(f"Save failed: {e}")
            else:
                st.info("No index in memory to save. Ingest a PDF first.")

        st.markdown("### Saved Indices")
        try:
            saved = [d for d in os.listdir(indices_dir) if os.path.isdir(os.path.join(indices_dir, d))]
        except Exception:
            saved = []
        if saved:
            for name in saved:
                col_a, col_b, col_c = st.columns([3, 1, 1])
                col_a.write(name)
                if col_b.button("Load", key=f"load_{name}"):
                    try:
                        store = FaissVectorStore.load(os.path.join(indices_dir, name))
                        st.session_state["store"] = store
                        st.session_state["chunks"] = store._metadatas
                        st.success(f"Loaded index '{name}'")
                    except Exception as e:
                        st.error(f"Load failed: {e}")
                if col_c.button("Delete", key=f"del_{name}"):
                    import shutil

                    try:
                        shutil.rmtree(os.path.join(indices_dir, name))
                        safe_rerun()
                    except Exception as e:
                        st.error(f"Delete failed: {e}")
        else:
            st.write("No saved indices")

        # Persistent chatbot in the sidebar
        st.markdown("---")
        st.markdown("## Chatbot")
        sidebar_q = st.text_input("Ask the assistant (global)", key="sidebar_q")
        if st.button("Send (sidebar)", key="sidebar_send") and sidebar_q:
            try:
                client = GeminiClient(api_key=API_KEY)
                resp = client.generate_text(sidebar_q, max_tokens=200)
                st.session_state.setdefault("chat_history", []).append(("You", sidebar_q))
                st.session_state.setdefault("chat_history", []).append(("Bot", resp))
            except Exception as e:
                st.error(f"Chat error: {e}")
        if "chat_history" in st.session_state:
            for who, msg in st.session_state["chat_history"][-6:]:
                st.markdown(f"**{who}**: {msg}")

        # Logout
        st.markdown("---")
        if st.button("Logout"):
            st.session_state["user"] = None
            st.success("Logged out")
            safe_rerun()

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

    def generate_chart_from_text(text: str):
        # Simple word-frequency based pie/bar chart for top concepts
        if not HAS_MATPLOTLIB:
            return None
        if not text:
            return None
        # Normalize and split
        txt = text.lower()
        txt = re.sub(r"[^a-z0-9\s]", " ", txt)
        words = [w for w in txt.split() if len(w) > 2]
        stop = {
            "the","and","for","with","that","this","from","have","are","was","were","which",
            "their","they","not","you","your","can","will","but","about","what","when","where",
        }
        words = [w for w in words if w not in stop]
        if not words:
            return None
        counts = Counter(words)
        top = counts.most_common(6)
        labels = [t[0] for t in top]
        sizes = [t[1] for t in top]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
        ax.axis('equal')
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return buf

    def generate_bar_chart_from_text(text: str):
        if not HAS_MATPLOTLIB:
            return None
        top = None
        try:
            txt = text.lower()
            txt = re.sub(r"[^a-z0-9\s]", " ", txt)
            words = [w for w in txt.split() if len(w) > 2]
            stop = {"the","and","for","with","that","this","from","have","are","was","were","which","their","they","not","you","your","can","will","but","about","what","when","where"}
            words = [w for w in words if w not in stop]
            counts = Counter(words)
            top = counts.most_common(8)
        except Exception:
            return None
        if not top:
            return None
        labels = [t[0] for t in top]
        values = [t[1] for t in top]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(labels[::-1], values[::-1], color='tab:blue')
        ax.set_xlabel('Frequency')
        ax.set_title('Top terms')
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return buf

    def generate_wordcloud_image(text: str):
        if WordCloud is None:
            return None
        try:
            wc = WordCloud(width=600, height=400, background_color='white').generate(text)
            buf = io.BytesIO()
            wc.to_image().save(buf, format='PNG')
            buf.seek(0)
            return buf
        except Exception:
            return None

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
    elif page == "Explain":
        with left:
            st.markdown("### Explain a Topic — Text + Charts")
            topic = st.text_area("Enter topic or paste text to explain", height=150)
            chart_type = st.selectbox("Chart type", ["Pie", "Bar", "WordCloud", "Multiple", "Image (GenAI)"])
            if st.button("Explain & Chart") and topic:
                with st.spinner("Generating explanation and chart..."):
                    try:
                        client = GeminiClient(api_key=API_KEY)
                        # Prefer using the SDK to generate a short explanation
                        explanation = client.generate_text(f"Summarize and explain this topic for a student:\n{topic}", max_tokens=250)
                    except Exception:
                        # Fallback: simple echo-summary
                        explanation = "Summary: " + (topic[:100] + "...")
                    st.subheader("Explanation")
                    st.write(explanation)
                    combined = explanation + " " + topic
                    chart_buf = None
                    if chart_type == "Pie":
                        chart_buf = generate_chart_from_text(combined)
                    elif chart_type == "Bar":
                        chart_buf = generate_bar_chart_from_text(combined)
                    elif chart_type == "WordCloud":
                        chart_buf = generate_wordcloud_image(combined) or generate_bar_chart_from_text(combined)
                    elif chart_type == "Multiple":
                        c1 = generate_chart_from_text(combined)
                        c2 = generate_bar_chart_from_text(combined)
                        if c1:
                            st.image(c1)
                        if c2:
                            st.image(c2)
                        chart_buf = None
                    elif chart_type == "Image (GenAI)":
                        # Attempt to call an image-generation method if available on client
                        img_buf = None
                        try:
                            if hasattr(client, "generate_image"):
                                # user SDK-specific; attempt to call
                                out = client.generate_image(topic)
                                # if SDK returns bytes or path, try to handle common types
                                if isinstance(out, (bytes, bytearray)):
                                    img_buf = io.BytesIO(out)
                                elif isinstance(out, str) and out.startswith("http"):
                                    # fetch image URL
                                    import requests

                                    r = requests.get(out)
                                    if r.status_code == 200:
                                        img_buf = io.BytesIO(r.content)
                        except Exception:
                            img_buf = None
                        if img_buf:
                            st.image(img_buf)
                        else:
                            st.info("Image generation not available with current SDK; try WordCloud or Bar charts.")
                    if chart_buf:
                        st.image(chart_buf)
        with right:
            st.markdown("### Tips")
            st.markdown("- Paste paragraphs or upload a PDF and copy text.\n- Use the Chatbot for follow-up questions.\n- Use Notes page to convert explanations into downloadable PDFs.")


if __name__ == "__main__":
    main()
