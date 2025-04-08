from dotenv import load_dotenv
import os

# Cargar variables de entorno
load_dotenv()

def clear_proxy_env():
    # Elimina todas las variables de entorno relacionadas con proxies
    proxy_keys = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy',
                  'ALL_PROXY', 'all_proxy', 'NO_PROXY', 'no_proxy']
    for key in proxy_keys:
        os.environ.pop(key, None)
    # Además, eliminar cualquier clave que contenga 'proxy' (en minúsculas)
    for key in list(os.environ.keys()):
        if 'proxy' in key.lower():
            os.environ.pop(key, None)
    # Verifica que no queden variables de proxy
    print("Proxy variables:", {k: v for k, v in os.environ.items() if 'proxy' in k.lower()})

def analizar_documento_groq(prompt, contexto=""):
    # Limpiar las variables de proxy antes de inicializar el cliente
    clear_proxy_env()
    
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in environment. Please set it in your .env file.")

    try:
        from groq import Groq
    except ImportError as e:
        raise ImportError("Failed to import groq library. Please install it with 'pip install groq'") from e

    # Inicializar el cliente de Groq
    try:
        client_groq = Groq(api_key=GROQ_API_KEY)
        print("Successfully connected to Groq!")
    except Exception as e:
        print("Error initializing Groq client:", e)
        exit(1)

    # Realizar la llamada a la API de chat
    try:
        prompt_completo = f"{contexto}\n\n{prompt}"
        response = client_groq.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt_completo}]
        )
        print("Received response from Groq:")
        print(response.choices[0].message.content)
    except Exception as e:
        print("Error during Groq API call:", e)
        raise e  # Lanza la excepción para manejarla en la capa superior si es necesario
    
    return response.choices[0].message.content, "llama3-8b-8192"

if __name__ == "__main__":
    prompt = input("Introduce el prompt para analizar el documento: ")
    contexto = input("Introduce el contexto (opcional): ")
    
    try:
        respuesta, modelo_usado = analizar_documento_groq(prompt, contexto)
        print("Respuesta del modelo:", respuesta)
        print("Modelo utilizado:", modelo_usado)
    except Exception as e:
        print("Ocurrió un error:", e)
