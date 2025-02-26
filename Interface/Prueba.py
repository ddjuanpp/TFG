import cohere
import os

from dotenv import load_dotenv

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

API_KEY = os.getenv("API_KEY")
# Inicializa el cliente de Cohere con tu API Key
co = cohere.Client(API_KEY)

# Obtén la lista de modelos disponibles
modelos = co.list_models()

# Muestra los nombres de los modelos
for modelo in modelos:
    print(modelo.name)