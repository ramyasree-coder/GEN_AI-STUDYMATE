import sys
import importlib

required = [
    ("streamlit", "streamlit"),
    ("dotenv", "dotenv"),
    ("PyPDF2", "PyPDF2"),
    ("numpy", "numpy"),
    ("requests", "requests"),
]

missing = []
for name, mod in required:
    try:
        importlib.import_module(mod)
        print(f"OK: {name}")
    except Exception as e:
        print(f"MISSING: {name} ({e})")
        missing.append(name)

# faiss is optional but check if available
faiss_ok = True
try:
    import faiss
    print("OK: faiss")
    # quick minimal sanity test
    idx = faiss.IndexFlatL2(8)
    import numpy as np
    vec = np.random.rand(1, 8).astype('float32')
    idx.add(vec)
    D, I = idx.search(vec, 1)
    print("faiss index test passed")
except Exception as e:
    print(f"faiss not available or failed: {e}")
    faiss_ok = False

if missing:
    print("\nOne or more required packages are missing. Re-run setup and check errors.")
    sys.exit(2)

if not faiss_ok:
    print("\nNote: FAISS is optional but recommended for semantic search. See README for conda installation instructions.")

print("\nAll core imports succeeded. You can run the Streamlit app with: streamlit run app.py")
sys.exit(0)
