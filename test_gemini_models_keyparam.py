import os
from dotenv import load_dotenv
import requests

load_dotenv('.env')
API_KEY = os.environ.get('GEMINI_API_KEY')
API_URL = os.environ.get('GEMINI_API_URL') or 'https://generativelanguage.googleapis.com'
models_url = API_URL.rstrip('/') + '/v1/models'

print('Trying models list with ?key= fallback')
if API_KEY:
    url_k = models_url + f'?key={API_KEY}'
else:
    url_k = models_url
try:
    r = requests.get(url_k, timeout=15)
    print('Status:', r.status_code)
    try:
        print(r.json())
    except Exception:
        print(r.text[:1000])
except Exception as e:
    print('Error:', repr(e))
