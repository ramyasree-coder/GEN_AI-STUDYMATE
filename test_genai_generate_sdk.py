import os
from dotenv import load_dotenv
load_dotenv('.env')
API_KEY = os.environ.get('GEMINI_API_KEY')
import google.genai as genai
client = genai.Client(api_key=API_KEY)
print('Calling generate_content via SDK...')
resp = client.models.generate_content(model='gemini-2.5-flash', contents='Hello from SDK test')
print('Type:', type(resp))
try:
    print('As dict keys:', resp.dict().keys())
    d = resp.dict()
    print(d.get('text'))
    print('Parts len:', len(d.get('parts', [])))
    print('Full dict:', d)
except Exception as e:
    print('Could not parse response:', e)
