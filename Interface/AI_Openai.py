# AI_model.py
import openai
from dotenv import load_dotenv
import os


# Cargar variables de entorno
load_dotenv()

def analizar_documento_openai_gpt4o_mini(prompt, contexto=""):
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("No se encontró la API key. Asegúrate de definir OPENAI_API_KEY en tu entorno.")
    prompt_completo = f"{contexto}\n\n{prompt}"
    openai.api_key = OPENAI_API_KEY
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt_completo},
        ]
    )
    return response.choices[0].message.content, "gpt-4o-mini"
