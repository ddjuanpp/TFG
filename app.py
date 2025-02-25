import streamlit as st
from documentos import DocumentUploader
from busqueda import HuggingFaceAPI, AVAILABLE_MODELS

# Configurar la página
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

# Agregar selección de modelo en la interfaz
selected_model = st.sidebar.selectbox(
    "🧠 Selecciona el modelo de IA",
    list(AVAILABLE_MODELS.keys()),
    index=0  # Mistral 7B por defecto
)

# Inicializar API con el modelo seleccionado
huggingface_api = HuggingFaceAPI(model_name=selected_model)

# ========== Layout en dos columnas ==========
col1, col2 = st.columns([1.5, 2])

# ---------- Columna Izquierda ----------
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

# ---------- Columna Derecha ----------
with col2:
    st.title("🔎 Análisis del Incidente")

    if st.button("🚀 Analizar Reporte"):
        if doc_uploader.get_documents():
            full_text = doc_uploader.get_concatenated_text()

            st.info("Procesando el documento y generando respuestas...")
            try:
                responses = {}
                for question in INCIDENT_QUESTIONS:
                    # Generar respuesta para cada pregunta
                    answer = huggingface_api.generar_pregunta(full_text, question)
                    responses[question] = answer

                # Mostrar respuestas en la interfaz
                st.subheader("📋 Resumen del Incidente")
                for i, (question, answer) in enumerate(responses.items()):
                    st.write(f"**{i+1}. {question}**")
                    st.write(f"➡️ **Respuesta:** {answer}")

            except Exception as e:
                st.error(f"❌ Error al analizar el documento: {e}")

        else:
            st.warning("⚠️ Primero sube un archivo PDF.")

