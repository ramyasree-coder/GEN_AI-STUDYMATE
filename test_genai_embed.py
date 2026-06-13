import os
from dotenv import load_dotenv
load_dotenv('.env')
API_KEY = os.environ.get('GEMINI_API_KEY')
import google.genai as genai
client = genai.Client(api_key=API_KEY)
print('Calling embed_content...')
resp = client.models.embed_content(model='gemini-embedding-2', contents=["Hello world", "Test embedding"])
print('Type:', type(resp))
try:
    d = resp.dict()
    print('Keys:', d.keys())
    print(d)
except Exception as e:
    print('Could not dict response:', e)
    try:
        print(resp)
    except Exception:
        pass
