import os
from dotenv import load_dotenv
load_dotenv('.env')
import requests

API_KEY = os.environ.get('GEMINI_API_KEY')
API_URL = os.environ.get('GEMINI_API_URL') or 'https://generativelanguage.googleapis.com'
MODEL = 'gemini-2.5-flash'

url = f"{API_URL.rstrip('/')}/v1/models/{MODEL}:generate"
headers = {'Content-Type': 'application/json'}
if API_KEY:
    headers['Authorization'] = f'Bearer {API_KEY}'

payload = {"prompt": {"text": "Hello from Gemini"}, "maxOutputTokens": 32}

print('REQUEST:')
print('POST', url)
print('HEADERS:')
for k, v in headers.items():
    if k.lower() == 'authorization':
        print(k + ':', 'Bearer ***REDACTED***')
    else:
        print(k + ':', v)
print('PAYLOAD:', payload)

try:
    r = requests.post(url, headers=headers, json=payload, timeout=15)
    print('\nRESPONSE STATUS:', r.status_code)
    try:
        print('RESPONSE JSON:', r.json())
    except Exception:
        print('RESPONSE TEXT:', r.text[:1000])
    # if unauthorized, try ?key= fallback
    if r.status_code != 200 and API_KEY:
        print('\nRetrying with ?key= fallback')
        sep = '&' if '?' in url else '?'
        url_k = f"{url}{sep}key={API_KEY}"
        r2 = requests.post(url_k, headers={'Content-Type': 'application/json'}, json=payload, timeout=15)
        print('RETRY STATUS:', r2.status_code)
        try:
            print('RETRY JSON:', r2.json())
        except Exception:
            print('RETRY TEXT:', r2.text[:1000])
        # If both header and ?key fallback returned 404, try alternative RPC name
        if r2.status_code == 404:
            alt_url = url.replace(':generate', ':generateContent') if ':generate' in url else url + ':generateContent'
            print('\nTrying alternative RPC:', alt_url)
            r3 = requests.post(alt_url, headers=headers, json=payload, timeout=15)
            print('ALT STATUS:', r3.status_code)
            try:
                print('ALT JSON:', r3.json())
            except Exception:
                print('ALT TEXT:', r3.text[:1000])
            if r3.status_code != 200 and API_KEY:
                sep = '&' if '?' in alt_url else '?'
                alt_k = f"{alt_url}{sep}key={API_KEY}"
                r4 = requests.post(alt_k, headers={'Content-Type': 'application/json'}, json=payload, timeout=15)
                print('ALT RETRY STATUS:', r4.status_code)
                try:
                    print('ALT RETRY JSON:', r4.json())
                except Exception:
                    print('ALT RETRY TEXT:', r4.text[:1000])
except Exception as e:
    print('REQUEST ERROR:', repr(e))
