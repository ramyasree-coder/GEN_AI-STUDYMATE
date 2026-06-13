import os
from dotenv import load_dotenv
load_dotenv('.env')
import google.genai as genai

API_KEY = os.environ.get('GEMINI_API_KEY')
client = genai.Client(api_key=API_KEY)
print('models methods:', [m for m in dir(client.models) if not m.startswith('_')])
try:
    import inspect
    if hasattr(client.models, 'generate_content'):
        print('\ngenerate_content sig:')
        print(inspect.signature(client.models.generate_content))
    if hasattr(client.models, 'embed_content'):
        print('\nembed_content sig:')
        print(inspect.signature(client.models.embed_content))
except Exception as e:
    print('inspect error', e)
