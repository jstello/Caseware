from google import genai
from google.genai import types

with open('receipts/20260210_181013.jpg', 'rb') as f:
    image_bytes = f.read()

client = genai.Client()
response = client.models.generate_content(
model='gemini-2.5-flash',
contents=[
    types.Part.from_bytes(
    data=image_bytes,
    mime_type='image/jpeg',
    ),
    'Extract the invoice information from this image. Respodn in JSON format'
]
)

print(response.text)