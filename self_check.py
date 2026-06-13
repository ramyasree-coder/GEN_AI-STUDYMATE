import os
import tempfile
import json

from users import init_db, create_user, verify_user, user_exists
from vector_store import FaissVectorStore

print("Running self-check for GenAI StudyMate...\n")

# DB check
init_db()
test_user = "selfcheck_user"
test_pass = "pass123"
if not user_exists(test_user):
    ok = create_user(test_user, test_pass)
    print(f"Created test user: {ok}")
else:
    print("Test user already exists")
ver = verify_user(test_user, test_pass)
print(f"Verify user login: {ver}")

# Faiss save/load check
tmpdir = tempfile.mkdtemp(prefix="study_faiss_")
path = os.path.join(tmpdir, "index_test")
print(f"Testing FAISS save/load in: {path}")
store = FaissVectorStore(embedding_dim=8)
docs = ["alpha beta gamma", "delta epsilon zeta"]
embs = [[0.1 * i for i in range(8)], [0.2 * i for i in range(8)]]
store.add_documents(docs, embs)
store.save(path)
print("Saved index")
store2 = FaissVectorStore.load(path)
print(f"Loaded index, documents: {len(store2._metadatas)}")

# Optional SDK check
gemini_key = os.environ.get("GEMINI_API_KEY")
if gemini_key:
    try:
        from chatbot import GeminiClient

        c = GeminiClient(api_key=gemini_key)
        print("Testing generate_text (SDK/REST)")
        out = c.generate_text("Hello test", max_tokens=10)
        print("Generate OK. Sample:" , out[:200])
        print("Testing embed_texts")
        emb = c.embed_texts(["test text"])
        print("Embedding length:", len(emb[0]) if emb and emb[0] else None)
    except Exception as e:
        print("Gemini tests failed:", e)
else:
    print("GEMINI_API_KEY not set; skipping API checks")

print('\nSelf-check completed.')
