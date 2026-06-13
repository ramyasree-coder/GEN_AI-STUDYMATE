import os
from dotenv import load_dotenv
import requests

load_dotenv('.env')
API_KEY = os.environ.get('GEMINI_API_KEY')
API_URL = os.environ.get('GEMINI_API_URL') or 'https://generativelanguage.googleapis.com'
MODEL = 'gemini-2.5-flash'
base_url = f"{API_URL.rstrip('/')}/v1/models/{MODEL}:generateContent"
headers = {'Content-Type': 'application/json'}
if API_KEY:
    headers_auth = {'Content-Type': 'application/json', 'Authorization': f'Bearer {API_KEY}'}
else:
    headers_auth = headers

payloads = [
    {"prompt": {"text": "Hello from Gemini"}, "maxOutputTokens": 32},
    {"input": {"text": "Hello from Gemini"}},
    {"text": "Hello from Gemini"},
    {"instances": [{"content": "Hello from Gemini"}]},
    {"input": [{"text": "Hello from Gemini"}]},
    {"content": [{"type": "text", "text": "Hello from Gemini"}]},
    {"messages": [{"role": "user", "content": "Hello from Gemini"}]},
    {"inputs": "Hello from Gemini"},
]

for i, p in enumerate(payloads, 1):
    print('\n--- Payload', i, '---')
    print(p)
    try:
        r = requests.post(base_url, headers=headers_auth, json=p, timeout=15)
        print('Status:', r.status_code)
        try:
            print('JSON:', r.json())
        except Exception:
            print('Text:', r.text[:500])
    except Exception as e:
        print('Error:', repr(e))
    # try with ?key fallback
    if API_KEY:
        sep = '&' if '?' in base_url else '?'
        url_k = base_url + f"{sep}key={API_KEY}"
        try:
            r2 = requests.post(url_k, headers={'Content-Type': 'application/json'}, json=p, timeout=15)
            print('Fallback Status:', r2.status_code)
            try:
                print('Fallback JSON:', r2.json())
            except Exception:
                print('Fallback Text:', r2.text[:500])
        except Exception as e:
            print('Fallback Error:', repr(e))
