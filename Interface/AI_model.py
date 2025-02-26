# AI_model.py

import requests
import os
from dotenv import load_dotenv


COHERE_MODELS = {
    "baseline": [
        "c4ai-aya-expanse-32b", # Modelo grande con 32 mil millones de parámetros, optimizado para generación de texto con alto rendimiento. Requiere muchos recursos.
        "c4ai-aya-expanse-8b", # Versión más pequeña (8B parámetros). Menos costoso en computación, pero aún potente. Buena opción intermedia.
        "command", # Modelo estándar de Cohere para generación de texto. Balance entre rendimiento y costo.
        "command-light" # Versión más ligera del modelo Command. Menos consumo de recursos, ideal para tareas rápidas.

    ],
    "command_r": [
        "command-r", # Primera versión optimizada para razonamiento complejo y contextos largos. Mejor que el Command estándar.
        "command-r-08-2024", # Versión mejorada de command-r con actualizaciones recientes.
        "command-r-plus", # Versión avanzada de command-r, con mayor capacidad de razonamiento y generación más precisa. Mayor consumo de recursos.
        "command-r-plus-08-2024", # Última versión de command-r-plus, optimizada para tareas más complejas.
        "command-r7b-12-2024" # Modelo más nuevo y eficiente de la serie Command-R. Mejor balance entre rendimiento y costo computacional.
    ],
    "nightly": [
        "command-light-nightly", # Versión experimental de command-light. Más rápido pero menos probado.
        "command-nightly" # Versión experimental del modelo command, con mejoras en rendimiento, pero posible inestabilidad.
    ]
}

# Cargar variables de entorno (.env)
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Verificar si la API Key está configurada
if not COHERE_API_KEY:
    raise ValueError("No se encontró la API Key de Cohere. Asegúrate de agregar COHERE_API_KEY en las variables de entorno.")

def analizar_documento_solo_texto(prompt):
    """
    Genera texto usando el modelo Command de Cohere.
    """
    url = "https://api.cohere.com/v1/chat"
    headers = {
        "Authorization": f"Bearer {COHERE_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "command-r7b-12-2024",
        "message": prompt,
        "temperature": 0.7
    }

    try:
        resp = requests.post(url, json=data, headers=headers)
        print("DEBUG - Status code:", resp.status_code)

        # Parse JSON response
        result = resp.json()
        print("DEBUG - Response JSON:", result)

        # Extraer texto de la respuesta de Cohere
        content = result.get("text", "⚠️ No se recibió respuesta válida de Cohere.")
        
        print(content)
        return content

    except Exception as e:
        return f"❌ Error al analizar el documento: {e}"
