# AI_model.py
from mistralai import Mistral
import openai
from dotenv import load_dotenv
import os


# Cargar variables de entorno
load_dotenv()

def analizar_documento_mistral_large_latest(prompt, contexto=""):
    """
    Analiza el documento usando el modelo Mistral Large Latest.
    Nota: Se asume que el identificador del modelo es "mistral-large-latest".
    """
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
    if not MISTRAL_API_KEY:
        raise ValueError("No se encontró la API key. Asegúrate de definir MISTRAL_API_KEY en tu entorno.")

    client_mistral = Mistral(api_key=MISTRAL_API_KEY)
    prompt_completo = f"{contexto}\n\n{prompt}"
    response = client_mistral.chat.complete(
        model="mistral-large-latest",
        messages=[
            {"role": "user", "content": prompt_completo},
        ]
    )
    return response.choices[0].message.content, "mistral-large-latest"

def analizar_documento_mistral_small_latest(prompt, contexto=""):
    """
    Analiza el documento usando el modelo Mistral 7B.
    Nota: Se asume que el identificador del modelo es "mistral-small-latest".
    """
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
    if not MISTRAL_API_KEY:
        raise ValueError("No se encontró la API key. Asegúrate de definir MISTRAL_API_KEY en tu entorno.")
    
    client_mistral = Mistral(api_key=MISTRAL_API_KEY)
    prompt_completo = f"{contexto}\n\n{prompt}"
    
    response = client_mistral.chat.complete(
        model="mistral-small-latest",  # Verifica el nombre exacto del modelo según la documentación oficial.
        messages=[
            {"role": "user", "content": prompt_completo},
        ]
    )
    return response.choices[0].message.content, "mistral-small-latest"

def analizar_documento_mistral_pixtral12b2409(prompt, contexto=""):
    """
    Analiza el documento usando el modelo Mistral 7B Instruct.
    Nota: Se asume que el identificador del modelo es "pixtral-12b-2409".
    """
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
    if not MISTRAL_API_KEY:
        raise ValueError("No se encontró la API key. Asegúrate de definir MISTRAL_API_KEY en tu entorno.")
    
    client_mistral = Mistral(api_key=MISTRAL_API_KEY)
    prompt_completo = f"{contexto}\n\n{prompt}"
    
    response = client_mistral.chat.complete(
        model="pixtral-12b-2409",  # Verifica el nombre exacto del modelo en la documentación oficial.
        messages=[
            {"role": "user", "content": prompt_completo},
        ]
    )
    return response.choices[0].message.content, "pixtral-12b-2409"