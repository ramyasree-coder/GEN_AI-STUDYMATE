from chatbot import GeminiClient
import os

api = os.getenv('GEMINI_API_KEY')
print('Using API key present:', bool(api))
client = GeminiClient(api_key=api)
try:
    res = client.generate_text('Create a one-sentence summary: Python developer with ML experience.', max_tokens=60)
    print('Result:', res[:500])
except Exception as e:
    print('Error:', e)
