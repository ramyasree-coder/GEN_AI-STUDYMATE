import os
from dotenv import load_dotenv
load_dotenv('.env')

from chatbot import GeminiClient

client = GeminiClient(api_key=os.getenv('GEMINI_API_KEY'))
print('Using SDK:', getattr(client, '_genai_name', None))
try:
    out = client.generate_text('Hello from Gemini', max_tokens=64)
    print('Generation result (truncated):', (out or '')[:800])
    print('Success' if out else 'Empty output')
except Exception as e:
    print('Generation error:', repr(e))
    # if requests Response attached, try to print more info
    try:
        import traceback
        traceback.print_exc()
    except Exception:
        pass
