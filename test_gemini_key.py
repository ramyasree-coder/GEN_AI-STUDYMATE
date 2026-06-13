import os
from dotenv import load_dotenv
import requests

# Load dotenv
load_dotenv('.env')
print('Loaded .env')

# Read raw .env and validate format
env_path = '.env'
with open(env_path, 'r', encoding='utf-8') as f:
    raw = f.read()

lines = [ln for ln in raw.splitlines() if ln.strip() and not ln.strip().startswith('#')]
fixed_lines = []
modified = False
for ln in lines:
    if '=' not in ln:
        # Try to auto-fix by ignoring spaces
        parts = ln.split()
        if len(parts) >= 2:
            k = parts[0].strip()
            v = parts[1].strip().strip('"').strip("'")
            fixed_lines.append(f"{k}={v}")
            modified = True
        else:
            fixed_lines.append(ln)
    else:
        k, v = ln.split('=', 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        new = f"{k}={v}"
        if new != ln:
            modified = True
        fixed_lines.append(new)

if modified:
    # write back normalized .env
    with open(env_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(fixed_lines) + '\n')
    print('Normalized .env formatting (trimmed quotes/spaces).')
else:
    print('.env format looks OK.')

# Ensure loaded into environment
load_dotenv('.env', override=True)
api_key = os.environ.get('GEMINI_API_KEY')
api_url = os.environ.get('GEMINI_API_URL') or 'https://generativelanguage.googleapis.com'
print('GEMINI_API_KEY present in env:', bool(api_key))
print('GEMINI_API_URL:', api_url)

headers = {}
if api_key:
    headers['Authorization'] = f'Bearer {api_key}'

# 1) Validate key by calling models list
models_url = api_url.rstrip('/') + '/v1/models'
try:
    r = requests.get(models_url, headers=headers, timeout=15)
    print('Models list response:', r.status_code)
    try:
        print(r.json())
    except Exception:
        print(r.text[:400])
    if r.status_code == 200:
        print('GEMINI key validated: models endpoint accessible.')
    elif r.status_code in (401, 403):
        print('GEMINI key appears invalid or lacks permission (status', r.status_code, ').')
    else:
        print('Models endpoint returned status', r.status_code)
except Exception as e:
    print('Error while validating key via models endpoint:', repr(e))

# 2) Try a direct generate call
model = 'text-bison-001'
gen_url = f"{api_url.rstrip('/')}/v1/models/{model}:generate"
payload = {"prompt": {"text": "Hello from Gemini"}, "maxOutputTokens": 32}
try:
    r2 = requests.post(gen_url, headers=headers, json=payload, timeout=15)
    print('Generate response status:', r2.status_code)
    try:
        data = r2.json()
        print('Generate response JSON keys:', list(data.keys()))
    except Exception:
        print('Generate response text:', r2.text[:400])
    if r2.status_code == 200:
        # Attempt to extract text
        out = None
        if isinstance(data, dict):
            if 'candidates' in data:
                out = data['candidates'][0].get('content')
            elif 'outputs' in data and data['outputs']:
                out = data['outputs'][0].get('content') or data['outputs'][0].get('text')
            elif 'text' in data:
                out = data.get('text')
        print('Generate output (extracted):', out)
        print('Generate succeeded')
    elif r2.status_code in (401, 403):
        print('Generate failed: invalid key or permission error (status', r2.status_code, ')')
    elif r2.status_code == 404:
        print('Generate failed: model not found (404). Key may be valid but model unavailable.')
    else:
        print('Generate returned status', r2.status_code)
except Exception as e:
    print('Error during generate call:', repr(e))

print('\nFinal .env format expected:')
print('GEMINI_API_KEY=YOUR_BEARER_TOKEN_OR_API_KEY')
print('GEMINI_API_URL=https://generativelanguage.googleapis.com')
