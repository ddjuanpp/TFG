from dotenv import load_dotenv
import os
from mistralai import Mistral

# Cargar variables de entorno
load_dotenv()

# Obtener y verificar la API key
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("No se encontró la API key. Asegúrate de definir MISTRAL_API_KEY en tu entorno.")

client = Mistral(api_key=MISTRAL_API_KEY)

def analizar_documento_solo_texto(prompt, contexto=""):
    prompt_completo = f"{contexto}\n\n{prompt}"
    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[
            {"role": "user", "content": prompt_completo},
        ]
    )
    return response.choices[0].message.content
