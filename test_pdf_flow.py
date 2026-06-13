import os
from dotenv import load_dotenv
load_dotenv('.env')

from chatbot import GeminiClient, QAEngine
from vector_store import FaissVectorStore
from pdf_processor import chunk_text
import traceback

client = GeminiClient()
print('Client initialized')
# avoid SDK blocking behavior in tests; prefer REST + deterministic fallback
client._genai = None

try:
    sample_text = '\n\n'.join([f"Paragraph {i}: This is a sample sentence about topic {i}." for i in range(1,6)])
    chunks = chunk_text(sample_text, chunk_size=200, overlap=40)
    if not chunks:
        print('No chunks produced (chunk_text returned empty). Aborting PDF flow test.')
        raise SystemExit(0)
    print('Chunks:', chunks)
    embs = client.embed_texts(chunks)
    print('Embeddings count:', len(embs), 'first dim:', len(embs[0]) if embs else None)

    store = FaissVectorStore(embedding_dim=len(embs[0]))
    store.add_documents(chunks, embs)
    print('Added documents to Faiss index; index nlist placeholder')

    # search test
    res = store.search(embs[0], top_k=3)
    print('Search results:', res)

    engine = QAEngine(client, store)
    notes = engine.generate_notes()
    print('Notes:', notes)
    mcqs = engine.generate_mcqs()
    print('MCQs:', mcqs)

except Exception as e:
    print('Error during PDF flow:', repr(e))
    print(traceback.format_exc())
