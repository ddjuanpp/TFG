# faiss_manager.py

import faiss
import numpy as np
from mistralai import Mistral
import os
from dotenv import load_dotenv
import openai
import time
import groq

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class FAISSManager:
    def __init__(self, api_key):
        """
        Inicializa el FAISS Manager con la API Key de Mistral.
        """
        self.api_key = api_key
        self.mistral_client = Mistral(api_key=api_key)  # Inicializa el cliente de Mistral
        self.index = None
        self.chunks = []  # Guardamos el texto de cada chunk
        self.dim = None   # Dimensión de embeddings (se define tras la primera llamada)

    def chunk_text(self, text, max_length=2048):
        chunks = []
        start = 0
        while start < len(text):
            end = start + max_length
            chunk = text[start:end]
            chunks.append(chunk)
            start = end
        return chunks

    def generate_embeddings_mistral(self, texts):
        """
        Genera embeddings usando Mistral (modelo mistral-embed).
        Parámetros:
        - texts: lista de strings.
        Retorna:
        - np.array de forma (len(texts), embedding_dim)
        """
        retries = 0
        max_retries = 5
        delay = 2  # Pausa inicial de 2 segundos

        while retries < max_retries:
            try:
                response = self.mistral_client.embeddings.create(
                    model="mistral-embed",
                    inputs=texts
                )

                if hasattr(response, "data") and response.data:
                    embeddings_list = [item.embedding for item in response.data]
                else:
                    raise Exception(f"Error al obtener embeddings: {response}")

                return np.array(embeddings_list, dtype=np.float32)

            except Exception as e:
                if "401" in str(e):
                    raise Exception("Error de autenticación: Verifica tu API Key de Mistral.")
                elif "429" in str(e):
                    print(f"Rate limit exceeded. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    retries += 1
                    delay *= 2
                else:
                    raise  # Re-lanzar la excepción si no es un error de límite de tasa

        raise Exception("Se alcanzó el máximo de reintentos debido a la tasa de solicitudes.")

    def generate_embeddings_openai(self, texts):
        """
        Genera embeddings usando OpenAI.
        Parámetros:
        - texts: lista de strings.
        Retorna:
        - np.array de forma (len(texts), embedding_dim)
        """
        openai.api_key = self.api_key
        response = openai.Embedding.create(
            model="text-embedding-ada-002",  # Modelo de OpenAI para embeddings
            input=texts
        )
        
        embeddings_list = [item['embedding'] for item in response['data']]
        return np.array(embeddings_list, dtype=np.float32)

    def generate_embeddings(self, texts):
        """
        Genera embeddings usando el modelo seleccionado (Mistral o OpenAI).
        Si se selecciona Groq, se usarán los embeddings de Mistral.
        """
        if self.api_key == MISTRAL_API_KEY or self.api_key == GROQ_API_KEY:
            return self.generate_embeddings_mistral(texts)
        elif self.api_key == OPENAI_API_KEY:
            return self.generate_embeddings_openai(texts)
        else:
            raise Exception("API Key no válida")

    def create_faiss_index(self, docs):
        """
        1) Divide todos los documentos en chunks.
        2) Genera embeddings para cada chunk (en lotes) usando Mistral.
        3) Crea y entrena el índice FAISS.
        """
        # 1) Crear chunks
        all_chunks = []
        for doc in docs:
            doc_chunks = self.chunk_text(doc, max_length=2048)
            all_chunks.extend(doc_chunks)
        self.chunks = all_chunks

        # 2) Generar embeddings por lotes
        embeddings = []
        batch_size = 16
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i+batch_size]
            emb_batch = self.generate_embeddings(batch)
            embeddings.append(emb_batch)
        
        if not embeddings:
            return
        
        embeddings = np.concatenate(embeddings, axis=0)
        self.dim = embeddings.shape[1]

        # 3) Crear el índice FAISS
        self.index = faiss.IndexFlatIP(self.dim)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

    def get_random_chunk(self):
        """
        Devuelve un chunk aleatorio del índice FAISS.
        """
        if not self.index or not self.chunks:
            return None
        import numpy as np
        idx = np.random.randint(0, len(self.chunks))
        return self.chunks[idx]
