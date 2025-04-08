# AI_Gemini.py
import os
from google import genai
import google.generativeai as genai

def analizar_documento_gemini_2_5_pro_preview_03_25(prompt, contexto=""):
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("No se encontró la API key. Asegúrate de definir GEMINI_API_KEY en tu entorno.")

    prompt_completo = f"{contexto}\n\n{prompt}"
    
    # Configuramos la API key para PaLM / Google Generative AI
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    # Utilizamos generate_chat (o generate_text) con un modelo existente
    response = client.models.generate_chat(
        model="gemini-2.5-pro-preview-03-25",  # Reemplaza por el modelo de tu preferencia
        contents= prompt_completo
    )
    
    # Extraemos la respuesta
    # Dependiendo de la versión de la librería, la estructura del response puede variar
    return response.last, "gemini-2.5-pro-preview-03-25"

def analizar_documento_gemini_2_0_flash(prompt, contexto=""):
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("No se encontró la API key. Asegúrate de definir GEMINI_API_KEY en tu entorno.")

    prompt_completo = f"{contexto}\n\n{prompt}"
    
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    response = client.models.generate_chat(
        model="gemini-2.0-flash",  # O "models/text-bison-001"
        contents= prompt_completo
    )

    return response.last, "gemini-2.0-flash"
