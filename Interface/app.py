# app.py

import streamlit as st
from documentos import DocumentUploader
from AI_model import analizar_documento_solo_texto
from faiss_manager import FAISSManager
import faiss
import cohere
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Ensure API key is loaded
if not COHERE_API_KEY:
    raise ValueError("⚠️ No se encontró la API Key de Cohere. Asegúrate de agregar COHERE_API_KEY en el archivo .env.")

# Pass API key to FAISSManager
faiss_manager = FAISSManager(api_key=COHERE_API_KEY)


st.set_page_config(page_title="Análisis de Incidentes Marítimos", page_icon="🌊", layout="wide")

# Preguntas clave para analizar incidentes marítimos
INCIDENT_QUESTIONS = [
    "At what time did the incident occur?",
    "Who is the vessel’s owner?",
    "Where was the vessel heading?",
    "Was there any routine activity, such as a crew member preparing food or drinks, at the time of the incident?",
    "What was the vessel’s speed when it began to experience difficulty (e.g., grounding, collision, etc.)?",
    "When was the vessel built?",
    "Did the accident have any fatalities?",
    "Was the vessel operating under normal conditions when the incident occurred?",
    "Were there any notable mechanical failures reported before the incident?",
    "Was there any distress signal or emergency call sent out prior to the incident?",
    "In which sea did the accident occur?",
    "What were the weather conditions at the time of the incident?",
    "What was the crew member's activity around the time of the accident?",
    "What did the crew member consume before heading to their post?",
    "What type or model of rescue boat was involved in the response to the accident?",
    "Where did the crew member retrieve safety equipment (e.g., lifejackets, knives, flotation devices) from onboard the vessel?",
    "Was there any communication from the vessel’s crew to nearby ships or maritime authorities before the incident?",
    "Were there any previous accidents or incidents involving this vessel?",
    "Was the vessel’s cargo secure at the time of the incident?",
    "Was the crew properly trained for handling emergency situations?",
    "What was the time in the GMT+1 time zone when the incident occurred?",
    "How many people were onboard at the time of the accident, and who were they?",
    "Was the vessel following the recommended navigational routes? (Provide the deviation in nautical miles from the planned route)",
    "How long did it take for the emergency response team to reach the vessel from the time of distress signal reception? (Provide the response time in hours and minutes)",
    "Were there any other vessels nearby at the time of the incident? (Provide the distance in nautical miles between the involved vessel and the nearest other vessel)",
    "What was the visibility like at the time of the accident? (Provide the distance in meters or miles)",
    "Did the vessel experience a reduction in speed before the incident? (If so, calculate the percentage decrease in speed from the normal operational speed)",
    "How many hours had the vessel been at sea before the incident occurred?",
    "What was the total cargo weight onboard at the time of the incident? (Provide the weight in tons or kilograms)",
    "Was there a significant change in the vessel's position before and after the incident? (Calculate the distance traveled in nautical miles, or the change in latitude/longitude)"
]

# ========== Instancias ==========

doc_uploader = DocumentUploader()

col1, col2 = st.columns([1.5, 2])

with col1:
    st.header("📂 Subir Reporte de Incidente Marítimo")
    uploaded_files = st.file_uploader(
        "Selecciona un archivo (PDF)",
        accept_multiple_files=True,
        type=['pdf']
    )
    if uploaded_files:
        for file in uploaded_files:
            try:
                doc_uploader.add_document(file)
                st.success(f"✅ {file.name} cargado correctamente.")
            except Exception as e:
                st.error(f"❌ Error al procesar {file.name}: {e}")

# Si hay documentos, crea el índice FAISS
if doc_uploader.get_documents():
    documents = doc_uploader.get_documents()
    faiss_manager.create_faiss_index(documents)

with col2:
    st.title("🔎 Análisis del Incidente")
    if st.button("🚀 Analizar Reporte"):
        if doc_uploader.get_documents():
            st.info("Procesando el documento y generando respuestas...")
            try:
                # Dividir las preguntas en 3 lotes de 10 preguntas cada uno
                question_batches = [
                    INCIDENT_QUESTIONS[:10],  # Primera tanda
                    INCIDENT_QUESTIONS[10:20],  # Segunda tanda
                    INCIDENT_QUESTIONS[20:]  # Tercera tanda
                ]

                responses = {}

                for batch_index, batch_questions in enumerate(question_batches, start=1):
                    st.write(f"### 🔍 Procesando Tanda {batch_index}...")

                    # Obtener contexto relevante de FAISS para las preguntas de la tanda actual
                    all_context_chunks = []
                    for question in batch_questions:
                        question_embedding = faiss_manager.generate_embeddings([question])
                        faiss.normalize_L2(question_embedding)

                        # Obtener los 5 chunks más relevantes por pregunta
                        k = 5
                        distances, indices = faiss_manager.index.search(question_embedding, k)
                        relevant_chunks = [faiss_manager.chunks[idx] for idx in indices[0] if idx >= 0]
                        all_context_chunks.extend(relevant_chunks)

                    # Unir los chunks de contexto en un solo bloque de texto
                    context_text = "\n".join(set(all_context_chunks))  # Eliminar duplicados

                    # Crear un único prompt con las preguntas de la tanda
                    prompt = f"Por favor, contesta en español cada una de las siguientes preguntas basándote en el CONTEXTO proporcionado para la Tanda {batch_index}.\n\n"
                    prompt += f"CONTEXTO:\n{context_text}\n\n"

                    for idx, question in enumerate(batch_questions, start=1):
                        prompt += f"{idx}. {question}\n"

                    prompt += "\nProporcione cada respuesta en el mismo formato numérico de las preguntas.\n"

                    # Llamar a Cohere con la tanda actual
                    answer_str = analizar_documento_solo_texto(prompt)

                    # Procesar la respuesta de Cohere
                    answer_lines = answer_str.strip().split("\n")

                    for line in answer_lines:
                        parts = line.split(". ", 1)
                        if len(parts) == 2 and parts[0].isdigit():
                            num, answer = parts
                            question_index = (batch_index - 1) * 10 + int(num) - 1
                            if 0 <= question_index < len(INCIDENT_QUESTIONS):
                                responses[INCIDENT_QUESTIONS[question_index]] = answer.strip()

                # Mostrar resultados en Streamlit
                st.subheader("📋 Resumen del Incidente")
                for i, (question, answer) in enumerate(responses.items()):
                    st.write(f"**{i+1}. {question}**")
                    st.write(f"➡️ **Respuesta:** {answer}")

            except Exception as e:
                st.error(f"❌ Error al analizar el documento: {e}")
        else:
            st.warning("⚠️ Primero sube un archivo PDF.")
