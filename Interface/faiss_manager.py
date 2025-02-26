# faiss_manager.py

import faiss
import numpy as np
import cohere
import os
from dotenv import load_dotenv

class FAISSManager:
    def __init__(self, api_key):
        """
        Inicializa el FAISS Manager con la API Key de Cohere.
        """
        self.api_key = api_key
        self.cohere_client = cohere.Client(api_key)  # Inicializa el cliente de Cohere
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

    def generate_embeddings(self, texts):
        """
        Genera embeddings usando Cohere (modelo embed-multilingual-v3.0).
        Parámetros:
        - texts: lista de strings.
        Retorna: np.array de forma (len(texts), embedding_dim)
        """
        response = self.cohere_client.embed(
            texts=texts,
            model="embed-multilingual-v3.0",
            input_type="search_document"  # Se requiere para embeddings de documentos
        )

        if not response.embeddings:
            raise Exception(f"Error al obtener embeddings: {response}")

        # Convertimos a array de floats
        return np.array(response.embeddings, dtype=np.float32)


    def create_faiss_index(self, docs):
        """
        1) Divide todos los documentos en chunks.
        2) Genera embeddings para cada chunk (en lotes) usando Cohere.
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
