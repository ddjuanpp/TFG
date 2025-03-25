import streamlit as st
from documentos import DocumentUploader
from AI_model import analizar_documento_solo_texto  # Versión adaptada a Mistral
from faiss_manager import FAISSManager              # Versión adaptada a Mistral
import faiss
import os
from dotenv import load_dotenv
import time

# Importa la nueva función para guardar todos los resultados en un Excel
from excel import append_answers_to_excel  

# Cargar variables de entorno
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    raise ValueError("⚠️ No se encontró la API Key de Mistral.")

st.set_page_config(
    page_title="Análisis de Incidentes Marítimos",
    page_icon="🌊",
    layout="wide"
)

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

with col2:
    st.title("🔎 Análisis del Incidente")
    
    if st.button("🚀 Analizar Reporte"):
        if doc_uploader.get_documents():
            st.info("Procesando los documentos y generando respuestas...")
            results = []
            documents = doc_uploader.get_documents()
            
            for idx, file in enumerate(uploaded_files):
                document_text = documents[idx]
                # Crear un FAISS index local para el documento actual
                local_faiss_manager = FAISSManager(api_key=MISTRAL_API_KEY)
                local_faiss_manager.create_faiss_index([document_text])
                
                local_responses = {}
                # Generar embeddings para las preguntas
                question_embeddings = local_faiss_manager.generate_embeddings(INCIDENT_QUESTIONS)
                faiss.normalize_L2(question_embeddings)
                
                batch_size = 5
                for batch_index in range(0, len(INCIDENT_QUESTIONS), batch_size):
                    current_questions = INCIDENT_QUESTIONS[batch_index: batch_index + batch_size]
                    current_embeddings = question_embeddings[batch_index: batch_index + batch_size]
                    
                    all_context_chunks = []
                    for emb in current_embeddings:
                        k = 5
                        distances, indices = local_faiss_manager.index.search(emb.reshape(1, -1), k)
                        relevant_chunks = [local_faiss_manager.chunks[i] for i in indices[0] if i >= 0]
                        all_context_chunks.extend(relevant_chunks)
                    
                    unique_chunks = list(set(all_context_chunks))
                    context_text = "\n".join(unique_chunks)
                    
                    prompt = (
                        "Por favor, contesta en **español** cada una de las siguientes preguntas "
                        "basándote en el CONTEXTO, **de forma breve y concisa**. "
                        "Si la información **no se menciona** en el CONTEXTO, responde con **'-'**.\n\n"
                        f"CONTEXTO:\n{context_text}\n\n"
                    )

                    for i, question in enumerate(current_questions, start=batch_index+1):
                        prompt += f"{i}. {question}\n"

                    prompt += (
                        "\nProporcione cada respuesta en el mismo formato numérico de las preguntas, "
                        "sin añadir explicaciones adicionales.\n"
                    )
                    
                    retries = 0
                    max_retries = 7
                    delay = 5
                    while retries < max_retries:
                        try:
                            answer_str = analizar_documento_solo_texto(prompt)
                            break
                        except Exception as e:
                            if "429" in str(e):
                                st.warning(
                                    f"Rate limit excedido. Reintentando en {delay} segundos... "
                                    f"(Intento {retries+1}/{max_retries})"
                                )
                                time.sleep(delay)
                                delay *= 2
                                retries += 1
                            else:
                                raise Exception(e)
                    else:
                        raise Exception("Se alcanzó el máximo de reintentos debido a la tasa de solicitudes.")
                    
                    answer_lines = answer_str.strip().split("\n")
                    for line in answer_lines:
                        parts = line.split(". ", 1)
                        if len(parts) == 2 and parts[0].isdigit():
                            num = int(parts[0])
                            respuesta = parts[1].strip()
                            if 1 <= num <= len(INCIDENT_QUESTIONS):
                                local_responses[INCIDENT_QUESTIONS[num - 1]] = respuesta
                    time.sleep(2)
                
                results.append({
                    "name": file.name,
                    "IA_model": "mistral-large-latest",
                    "responses": local_responses
                })
            
            st.subheader("📋 Resumen de los Incidentes")
            for result in results:
                st.write(f"**Documento: {result['name']}**")
                for i, question in enumerate(INCIDENT_QUESTIONS, start=1):
                    respuesta = result["responses"].get(question, "-")
                    st.write(f"**{i}. {question}**")
                    st.write(f"➡️ **Respuesta:** {respuesta}")
                st.write("---")

            # Aquí usamos append_answers_to_excel en lugar de sobrescribir
            excel_filename = append_answers_to_excel(results, INCIDENT_QUESTIONS, save_path="../resultados.xlsx")
            st.success(f"Excel actualizado: {excel_filename}")
            
            # Botón de descarga
            with open(excel_filename, "rb") as f:
                st.download_button(
                    label="Descargar Excel",
                    data=f,
                    file_name=os.path.basename(excel_filename),
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.warning("⚠️ Primero sube un archivo PDF.")
