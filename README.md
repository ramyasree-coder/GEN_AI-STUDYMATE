# GenAI StudyMate

GenAI StudyMate is a Streamlit dashboard for studying documents with the help of a generative AI. Upload PDFs, index their content into a FAISS vector store, and interact with the document to get answers, concise notes, quizzes, and key concepts.

## Features
- PDF Upload and text extraction
- Chat with PDF (question answering over the document)
- Notes Generation (structured study notes)
- Quiz Generation (MCQs from document content)
- Key Concepts Extraction (concise bullets)

## Tech Stack
- Python
- Streamlit (UI)
- Google Gemini / Generative AI SDK (via `google-genai` / REST fallback)
- FAISS (vector store)

## Installation
1. Create a virtual environment and install dependencies:

```bash
# Recommended: Python 3.11
python -m venv .venv
.venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and set your `GEMINI_API_KEY` (and optionally `GEMINI_API_URL`).

3. Run the Streamlit app:

```bash
streamlit run app.py
```

Notes: If no API key is provided the app falls back to deterministic local embeddings and simple local generation for offline testing.

## Usage
1. Open the app in your browser (Streamlit will print the local URL).
2. Use the sidebar to upload a PDF under **Upload**.
3. Once ingested, use **Chat** to ask questions about the document, **Notes** to generate study notes, **Quiz** to create MCQs, and **Dashboard** / **Key Concepts** to see summaries and metadata.

## Screenshots
Replace these with actual screenshots from your run (images/ directory recommended):

![Dashboard header](screenshots/dashboard-header.png)
![Chat with PDF](screenshots/chat-page.png)
![Notes page](screenshots/notes-page.png)

## Future Enhancements
- Persist FAISS index to disk and load between runs
- Add user authentication and per-user workspaces
- Improve UI styling and add icons/illustrations
- Support additional generative models and batching
- Add unit tests and CI for E2E flows

## File overview
- `app.py` — Streamlit UI and orchestration
- `pdf_processor.py` — PDF text extraction and chunking
- `vector_store.py` — FAISS vector store wrapper
- `chatbot.py` — Gemini client wrapper and QA engine
- `e2e_run.py` — end-to-end test script that prints raw SDK responses
- `requirements.txt` — Python dependencies

---

If you reuse this project, do not commit your `.env` file (it contains secrets).

License: See the `LICENSE` file for full license text (MIT).
