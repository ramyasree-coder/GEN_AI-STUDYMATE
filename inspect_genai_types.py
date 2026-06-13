import google.genai as genai
import inspect
print('GenerateContentConfig attrs:')
print([a for a in dir(genai.types.GenerateContentConfig) if not a.startswith('_')])
print('\nEmbedContentResponse attrs:')
print([a for a in dir(genai.types.EmbedContentResponse) if not a.startswith('_')])
print('\nGenerateContentResponse attrs:')
print([a for a in dir(genai.types.GenerateContentResponse) if not a.startswith('_')])
print('\nEmbedContentResponse.__doc__:\n', genai.types.EmbedContentResponse.__doc__)
print('\nGenerateContentResponse.__doc__:\n', genai.types.GenerateContentResponse.__doc__)
