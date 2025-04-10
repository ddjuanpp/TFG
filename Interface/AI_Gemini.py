# AI_Gemini.py
import os
import sys
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def analizar_documento_gemini_2_0_flash(prompt, contexto=""):
    """
    Analiza un documento utilizando el modelo "gemini-2.0-flash".
    
    Args:
        prompt (str): El contenido principal del documento.
        contexto (str, optional): Texto adicional que aporta contexto. Defaults to "".
        
    Returns:
        tuple: Una tupla (respuesta, nombre_del_modelo) con el texto generado y el identificador del modelo.
    """
    if not GEMINI_API_KEY:
        raise ValueError("No se encontró la API key. Asegúrate de definir GEMINI_API_KEY en tu entorno.")

    prompt_completo = f"{contexto}\n\n{prompt}" if contexto else prompt

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(
        prompt_completo
    )

    return response.text, "gemini-2.0-flash"

def analizar_documento_gemini_1_5_flash_8b(prompt, contexto=""):
    """
    Analiza un documento utilizando el modelo "gemini-1.5-flash-8b".
    
    Args:
        prompt (str): El contenido principal del documento.
        contexto (str, optional): Texto adicional que aporta contexto. Defaults to "".
    """
    if not GEMINI_API_KEY:
        raise ValueError("No se encontró la API key. Asegúrate de definir GEMINI_API_KEY en tu entorno.")

    prompt_completo = f"{contexto}\n\n{prompt}" if contexto else prompt

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash-8b")
    response = model.generate_content(
        prompt_completo
    )

    return response.text, "gemini-1.5-flash-8b"
