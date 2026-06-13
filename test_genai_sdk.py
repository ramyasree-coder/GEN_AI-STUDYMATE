import os
from dotenv import load_dotenv
load_dotenv('.env')
API_KEY = os.environ.get('GEMINI_API_KEY')
try:
    import google.genai as genai
except Exception as e:
    print('google.genai not installed or import failed:', e)
    raise

print('Configuring SDK with API key')
try:
    genai.configure(api_key=API_KEY)
    print('Configured')
except Exception as e:
    print('Configure error:', e)

MODEL = 'gemini-2.5-flash'
print('Calling genai.generate...')
try:
    resp = genai.generate(model=MODEL, text='Hello from SDK test', max_output_tokens=50)
    print('SDK response type:', type(resp))
    try:
        print('As dict:', dict(resp))
    except Exception:
        print('Response repr:', repr(resp))
except Exception as e:
    print('SDK generate error:', repr(e))
