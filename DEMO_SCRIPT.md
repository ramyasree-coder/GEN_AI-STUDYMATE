# Demo Script — GenAI StudyMate (3-minute demo)

Goal: Show core functionality quickly: login, upload+ingest, chat, generate notes, show charts, save/load index.

Environment prep (start before recording)
- Activate virtualenv and install requirements: `pip install -r requirements.txt`
- Ensure `.env` contains `GEMINI_API_KEY` or omit to use fallback behavior.
- Start Streamlit:

```powershell
.venv\Scripts\streamlit.exe run app.py
```

Demo timeline (3 minutes)
- 0:00–0:20 — Intro (project name, purpose in one sentence).
- 0:20–0:50 — Show Login/Register (create a test user, log in).
- 0:50–1:20 — Upload a sample PDF (e.g., sample_resume.pdf) and click ingest; mention embedding dim detection and FAISS.
- 1:20–1:45 — Chat with the document: ask a content question and show the answer.
- 1:45–2:10 — Generate Notes and download as PDF; open downloaded file to show content.
- 2:10–2:30 — Explain page: paste topic or use extracted text, select WordCloud/Bar chart, show visualization.
- 2:30–2:50 — Save the index under a name in the sidebar, then Load it (demonstrate persistence).
- 2:50–3:00 — Closing: limitations and next steps (image-gen fallback, multi-user project features).

Recording tips
- Keep a stopwatch visible to track time.
- Speak clearly and narrate what the app does, not implementation details.
- If a GenAI call is slow, explain it briefly and continue with the saved index demo.

Suggested sample commands to run separately (for reproducibility)

```powershell
# Run E2E script (non-interactive) to validate pipeline
.venv\Scripts\python.exe e2e_run.py

# List saved indices for a user
python - <<'PY'
import os
print(os.listdir('user_data'))
PY
```