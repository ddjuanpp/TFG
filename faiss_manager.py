# faiss_manager.py

import faiss
import numpy as np
import json
import boto3

class FAISSManager:
    def __init__(self):
        self.bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')
        self.index = None
        self.chunks = []  # guardamos el texto de cada chunk
        self.dim = None   # dimensión de embeddings (la definiremos tras la primera llamada)

    def chunk_text(self, text, max_length=2048):
        """
        Divide un string `text` en una lista de trozos (chunks),
        donde cada chunk tiene como máximo `max_length` caracteres.
        """
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
        Genera embeddings usando 'cohere.embed-multilingual-v3' a través de Bedrock.
        texts: lista de strings
        Retorna: np.array de forma (len(texts), embedding_dim)
        """
        # Construimos el body JSON con la lista "texts".
        # Importante: 'texts' NO debe contener cadenas de más de 2048 caracteres,
        # por eso hemos hecho el chunking antes.
        body = json.dumps({
            "texts": texts,
            "input_type": "search_document",
            "truncate": "END"
        })

        response = self.bedrock_client.invoke_model(
            modelId="cohere.embed-multilingual-v3",
            accept="application/json",
            contentType="application/json",
            body=body
        )
        response_body = json.loads(response['body'].read())
        embeddings = response_body['embeddings']
        return np.array(embeddings, dtype=np.float32)

    def create_faiss_index(self, docs):
        """
        1) Divide todos los documentos en chunks de máximo 2048 caracteres,
        2) Genera embeddings para cada chunk (en lotes),
        3) Crea un índice FAISS y almacena los vectores.
        """
        # 1) Crear chunks de todos los docs
        all_chunks = []
        for doc in docs:
            doc_chunks = self.chunk_text(doc, max_length=2048)
            all_chunks.extend(doc_chunks)

        self.chunks = all_chunks

        # 2) Generar embeddings por lotes
        embeddings = []
        batch_size = 16  # Ajusta según tu límite o conveniencia
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i+batch_size]
            emb_batch = self.generate_embeddings(batch)
            embeddings.append(emb_batch)

        if not embeddings:
            return  # No se agregan embeddings si la lista está vacía

        embeddings = np.concatenate(embeddings, axis=0)  # (N, embedding_dim)
        self.dim = embeddings.shape[1]

        # 3) Crear y entrenar el índice FAISS
        self.index = faiss.IndexFlatIP(self.dim)  
        # Normalizamos para que sea más similar a coseno
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)  # Agregar todos los vectores

    def get_random_chunk(self):
        """
        Devuelve un chunk aleatorio del índice FAISS.
        """
        if not self.index or not self.chunks:
            return None
        idx = np.random.randint(0, len(self.chunks))
        return self.chunks[idx]
