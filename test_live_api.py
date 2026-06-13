import os
from dotenv import load_dotenv
load_dotenv('.env')
print('Loaded .env')
print('GEMINI_API_KEY present:', 'GEMINI_API_KEY' in os.environ)

from chatbot import GeminiClient, QAEngine
from vector_store import FaissVectorStore
from pdf_processor import chunk_text

client = GeminiClient()  # reads env vars
print('Using API URL:', client.api_url)

# 1) Test embeddings
try:
    texts = ['Hello world', 'This is a test embedding']
    embs = client.embed_texts(texts)
    print('Embeddings returned:', len(embs), 'vectors; first dim:', len(embs[0]) if embs else None)
except Exception as e:
    import traceback
    print('Embeddings error:', repr(e))
    print(traceback.format_exc())

# 2) Test content generation
try:
    out = client.generate_text('Write a concise 1-sentence definition of machine learning.', max_tokens=60)
    print('Generation output:', out[:400])
except Exception as e:
    import traceback
    print('Generation error:', repr(e))
    print(traceback.format_exc())

# 3/4) Simulate PDF upload -> notes and MCQ generation
try:
    # simulate extracted PDF text by creating a long sample
    sample_text = '\n\n'.join([f"Paragraph {i}: This is a sample sentence about topic {i}." for i in range(1,6)])
    chunks = chunk_text(sample_text, chunk_size=200, overlap=40)
    print('Generated', len(chunks), 'chunks from sample text')

    # embed chunks
    chunk_embs = client.embed_texts(chunks)
    print('Chunk embeddings:', len(chunk_embs))

    # build store
    store = FaissVectorStore(embedding_dim=len(chunk_embs[0]))
    store.add_documents(chunks, chunk_embs)

    # QA engine
    engine = QAEngine(client, store)
    notes = engine.generate_notes()
    print('Notes output (truncated):', notes[:600])

    mcqs = engine.generate_mcqs(n=3)
    print('MCQs output sample:', mcqs[:2])
except Exception as e:
    import traceback
    print('PDF/Notes/MCQ flow error:', repr(e))
    print(traceback.format_exc())

print('\nLive API test script finished')
