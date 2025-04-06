from dotenv import load_dotenv
import os
from mistralai import Mistral
import openai
import groq
# Cargar variables de entorno
load_dotenv()

# Obtener y verificar la API key
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("No se encontró la API key. Asegúrate de definir MISTRAL_API_KEY en tu entorno.")

def analizar_documento_mistral(prompt, contexto=""):
    client = Mistral(api_key=MISTRAL_API_KEY)
    prompt_completo = f"{contexto}\n\n{prompt}"
    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[
            {"role": "user", "content": prompt_completo},
        ]
    )
    return response.choices[0].message.content, "mistral-large-latest"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("No se encontró la API key. Asegúrate de definir OPENAI_API_KEY en tu entorno.")

def analizar_documento_openai(prompt, contexto=""):
    prompt_completo = f"{contexto}\n\n{prompt}"
    openai.api_key = OPENAI_API_KEY
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt_completo},
        ]
    )
    return response.choices[0].message.content, "gpt-4o-mini"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("No se encontró la API key. Asegúrate de definir GROQ_API_KEY en tu entorno.")

def analizar_documento_groq(prompt, contexto=""):
    client = groq.Groq(api_key=GROQ_API_KEY)
    prompt_completo = f"{contexto}\n\n{prompt}"
    groq.api_key = GROQ_API_KEY
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "user", "content": prompt_completo},
        ]
    )
    return response.choices[0].message.content, "llama3-8b-8192"

