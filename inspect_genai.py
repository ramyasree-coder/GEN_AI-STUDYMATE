import inspect
import google
import google.genai as genai

print('google.genai module:', genai)
print('\nTop-level attributes:')
print([a for a in dir(genai) if not a.startswith('_')])

# print inspect of known names
for name in ['Client', 'configure', 'generate', 'Embeddings', 'embeddings', 'Embedding', 'TextGenerationModel', 'GenerativeModel', 'Generation']:
    if hasattr(genai, name):
        print('\nFOUND:', name)
        obj = getattr(genai, name)
        try:
            print(inspect.getsource(obj))
        except Exception:
            print(repr(obj))
