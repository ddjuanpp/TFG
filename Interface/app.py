import streamlit as st
from documentos import DocumentUploader

from AI_Mistral import (
    analizar_documento_mistral_large_latest, 
    analizar_documento_mistral_small_latest, 
    analizar_documento_mistral_pixtral12b2409
)
from AI_Openai import analizar_documento_openai_gpt4o_mini
from AI_Gemini import (
    analizar_documento_gemini_2_0_flash, 
    analizar_documento_gemini_1_5_flash_8b
)
from faiss_manager import FAISSManager
import faiss
import os
from dotenv import load_dotenv
import time
from excel import append_answers_to_excel  
import zipfile
import io

# Cargar variables de entorno
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not MISTRAL_API_KEY and not OPENAI_API_KEY and not GROQ_API_KEY and not GEMINI_API_KEY:
    raise ValueError("⚠️ No se encontró la API Key de Mistral ni OpenAI ni Groq ni Gemini.")

st.set_page_config(
    page_title="Análisis de Incidentes Marítimos",
    page_icon="🌊",
    layout="wide"
)

INCIDENT_QUESTIONS = [
    "At what time did the incident occur?",
    "Who is the vessel's owner?",
    "Where was the vessel heading?",
    "Was there any routine activity, such as a crew member preparing food or drinks, at the time of the incident?",
    "What was the vessel's speed when it began to experience difficulty (e.g., grounding, collision, etc.)?",
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
    "Was there any communication from the vessel's crew to nearby ships or maritime authorities before the incident?",
    "Were there any previous accidents or incidents involving this vessel?",
    "Was the vessel's cargo secure at the time of the incident?",
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
    
    modelo_seleccionado = st.selectbox(
        "Selecciona el modelo de IA:", 
        [
            "mistral-large-latest", 
            "mistral-small-latest", 
            "pixtral-12b-2409",
            "gpt-4o-mini", 
            "gemini-2.0-flash",
            "gemini-1.5-flash-8b"
        ]
    )
    
    if st.button("🚀 Analizar Reporte"):
        if doc_uploader.get_documents():
            st.info("Procesando los documentos y generando respuestas...")
            results = []
            documents = doc_uploader.get_documents()
            
            for idx, file in enumerate(uploaded_files):
                document_text = documents[idx]

                # Decidir qué API key usar, en tu caso quizás no sea tan relevante
                if modelo_seleccionado in [
                    "mistral-large-latest", 
                    "mistral-small-latest", 
                    "pixtral-12b-2409",
                    "gemini-2.0-flash",
                    "gemini-1.5-flash-8b"
                ]:
                    api_key_used = MISTRAL_API_KEY
                elif modelo_seleccionado in [
                    "gpt-4o-mini",
                ]:
                    api_key_used = OPENAI_API_KEY


                # Diccionario para almacenar las respuestas de este documento
                local_responses = {}


                local_faiss_manager = FAISSManager(api_key=api_key_used)
                local_faiss_manager.create_faiss_index([document_text])

                # Generar embeddings de las preguntas
                question_embeddings = local_faiss_manager.generate_embeddings(INCIDENT_QUESTIONS)
                faiss.normalize_L2(question_embeddings)

                time.sleep(2)
                batch_size = 5
                for batch_index in range(0, len(INCIDENT_QUESTIONS), batch_size):
                    current_questions = INCIDENT_QUESTIONS[batch_index: batch_index + batch_size]
                    current_embeddings = question_embeddings[batch_index: batch_index + batch_size]

                    # Reunir los K chunks más relevantes para cada embedding
                    all_context_chunks = []
                    for emb in current_embeddings:
                        k = 5
                        distances, indices = local_faiss_manager.index.search(emb.reshape(1, -1), k)
                        relevant_chunks = [
                            local_faiss_manager.chunks[i] 
                            for i in indices[0] 
                            if i >= 0
                        ]
                        all_context_chunks.extend(relevant_chunks)

                    # Eliminar duplicados y combinarlos en un solo texto
                    unique_chunks = list(set(all_context_chunks))
                    context_text = "\n".join(unique_chunks)

                    # Construir prompt con contexto + preguntas
                    prompt_text = (
                        "Please answer each of the following questions based on the CONTEXT provided.\n"
                        "Whenever possible, extract the **full expression or complete phrase** exactly as it appears in the CONTEXT.\n"
                        "If the information is not mentioned, respond with '-'.\n"
                        "Be precise and preserve details such as approximate times, dates, measurements, and any qualifiers.\n\n"
                        f"CONTEXT:\n{context_text}\n\n"
                    )
                    
                    for i, question in enumerate(current_questions, start=batch_index + 1):
                        prompt_text += f"{i}. {question}\n"

                    prompt_text += (
                        "\nProvide each answer with the same numeric format as the questions, "
                        "without adding additional explanations.\n"
                    )

                    # Llamada al modelo
                    
                    retries = 0
                    max_retries = 7
                    delay = 5
                    while retries < max_retries:
                        try:
                            if modelo_seleccionado == "mistral-large-latest":
                                answer_str, model_used = analizar_documento_mistral_large_latest(prompt_text)
                            elif modelo_seleccionado == "mistral-small-latest":
                                answer_str, model_used = analizar_documento_mistral_small_latest(prompt_text)
                            elif modelo_seleccionado == "pixtral-12b-2409":
                                answer_str, model_used = analizar_documento_mistral_pixtral12b2409(prompt_text)
                            elif modelo_seleccionado == "gpt-4o-mini":
                                answer_str, model_used = analizar_documento_openai_gpt4o_mini(prompt_text)
                            elif modelo_seleccionado == "gemini-2.0-flash":
                                answer_str, model_used = analizar_documento_gemini_2_0_flash(prompt_text)
                            elif modelo_seleccionado == "gemini-1.5-flash-8b":
                                answer_str, model_used = analizar_documento_gemini_1_5_flash_8b(prompt_text)
                            if answer_str is None:
                                raise ValueError("La respuesta del modelo es None. Verifica la configuración del modelo o la API.")
                            break
                        except Exception as e:
                            if "429" in str(e):
                                st.warning(
                                    f"Rate limit exceeded. Retrying in {delay} seconds... "
                                    f"(Attempt {retries + 1}/{max_retries})"
                                )
                                time.sleep(delay)
                                delay = 2
                                retries += 1
                            else:
                                raise Exception(e)
                    else:
                        raise Exception("Se alcanzó el máximo de reintentos debido a la tasa de solicitudes.")

                    # Parsear las respuestas
                    answer_lines = answer_str.strip().split("\n")
                    for line in answer_lines:
                        parts = line.split(". ", 1)
                        if len(parts) == 2 and parts[0].isdigit():
                            num = int(parts[0])
                            respuesta = parts[1].strip()
                            if 1 <= num <= len(INCIDENT_QUESTIONS):
                                local_responses[INCIDENT_QUESTIONS[num - 1]] = respuesta

                    time.sleep(2)

                # Guardamos la info final de este documento en results
                results.append({
                    "name": file.name,
                    "IA_model": model_used,
                    "responses": local_responses
                })

            # Mostrar resultados en pantalla
            st.subheader("📋 Resumen de los Incidentes")
            for result in results:
                st.write(f"**Documento: {result['name']}**")
                for i, question in enumerate(INCIDENT_QUESTIONS, start=1):
                    respuesta = result["responses"].get(question, "-")
                    st.write(f"**{i}. {question}**")
                    st.write(f"➡️ **Respuesta:** {respuesta}")
                st.write("---")

            # Guardar en Excel
            excel_filename = append_answers_to_excel(results, INCIDENT_QUESTIONS, save_path="../resultados.xlsx")
            st.success(f"Excel actualizado")

            # AGREGAR ESTE BLOQUE PARA CREAR UN EXCEL POR CADA MODELO:
            from collections import defaultdict

            # Agrupamos los resultados por modelo (IA_model)
            model_results = defaultdict(list)
            for r in results:
                model_name = r["IA_model"]
                model_results[model_name].append(r)

            # Para cada modelo, guardamos o actualizamos un Excel específico
            for model_name, model_data in model_results.items():
                model_excel_path = f"../{model_name}.xlsx"
                model_excel_filename = append_answers_to_excel(model_data, INCIDENT_QUESTIONS, save_path=model_excel_path)
                st.success(f"Excel del modelo '{model_name}' actualizado")
            
            excel_files = [excel_filename]
            for model_name in model_results:
                model_excel_path = f"../{model_name}.xlsx"
                excel_files.append(model_excel_path)

            # Crear un buffer en memoria para el archivo ZIP
            zip_buffer = io.BytesIO()

            # Abrir el zip y agregar cada archivo Excel, usando solo su nombre (sin ruta completa)
            with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
                for file_path in excel_files:
                    arcname = os.path.basename(file_path)
                    zip_file.write(file_path, arcname=arcname)

            # Regresar el cursor al inicio del buffer para la descarga
            zip_buffer.seek(0)

            # Botón para descargar el archivo ZIP que contiene todos los Excels
            st.download_button(
                label="Descargar todos los Excels en un ZIP",
                data=zip_buffer,
                file_name="excels.zip",
                mime="application/zip"
            )
        else:
            st.warning("⚠️ Primero sube un archivo PDF.")
