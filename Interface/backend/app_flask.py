import os
import time
import faiss
from flask import Flask, request, jsonify, send_file
from documentos import DocumentUploader
from AI_model import analizar_documento_solo_texto
from faiss_manager import FAISSManager
from excel import save_answers_to_excel
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("No se encontró MISTRAL_API_KEY en el entorno.")

app = Flask(__name__)

# Instanciar las clases de backend
doc_uploader = DocumentUploader()
faiss_manager = FAISSManager(api_key=MISTRAL_API_KEY)

# Lista global de nombres de archivos subidos
uploaded_files_info = []

# Lista de preguntas (idéntica a la usada en tu app Streamlit)
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

@app.route('/upload', methods=['POST'])
def upload():
    """
    Endpoint para subir uno o varios archivos PDF.
    """
    files = request.files.getlist('files')
    if not files:
        return jsonify({"error": "No se proporcionó ningún archivo."}), 400

    file_names = []
    for file in files:
        try:
            # Agregar documento (se extrae el texto internamente)
            doc_uploader.add_document(file)
            file_names.append(file.filename)
        except Exception as e:
            return jsonify({"error": f"Error al procesar {file.filename}: {str(e)}"}), 400

    global uploaded_files_info
    uploaded_files_info = file_names

    return jsonify({"message": "Archivos subidos correctamente.", "files": file_names}), 200

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Endpoint para analizar los documentos subidos.
    Se crea el índice FAISS, se generan embeddings para las preguntas,
    se consulta el índice para obtener contexto y se llama al modelo de análisis.
    Finalmente, se genera un Excel para cada PDF subido.
    """
    documents = doc_uploader.get_documents()
    if not documents:
        return jsonify({"error": "No hay documentos subidos."}), 400

    try:
        # Crear el índice FAISS a partir de los documentos
        faiss_manager.create_faiss_index(documents)
    except Exception as e:
        return jsonify({"error": f"Error al crear el índice FAISS: {str(e)}"}), 500

    responses = {}
    batch_size = 5

    try:
        # Generar embeddings para las preguntas
        question_embeddings = faiss_manager.generate_embeddings(INCIDENT_QUESTIONS)
    except Exception as e:
        return jsonify({"error": f"Error al generar embeddings: {str(e)}"}), 500

    # Normalizar embeddings
    faiss.normalize_L2(question_embeddings)

    # Procesar preguntas en lotes
    for batch_index in range(0, len(INCIDENT_QUESTIONS), batch_size):
        current_questions = INCIDENT_QUESTIONS[batch_index: batch_index + batch_size]
        current_embeddings = question_embeddings[batch_index: batch_index + batch_size]

        all_context_chunks = []
        for emb in current_embeddings:
            k = 5  # Número de chunks relevantes
            distances, indices = faiss_manager.index.search(emb.reshape(1, -1), k)
            relevant_chunks = [faiss_manager.chunks[idx] for idx in indices[0] if idx >= 0]
            all_context_chunks.extend(relevant_chunks)

        unique_chunks = list(set(all_context_chunks))
        context_text = "\n".join(unique_chunks)

        # Construir prompt para el modelo
        prompt = (
            "Por favor, contesta en español cada una de las siguientes preguntas "
            "basándote en el CONTEXTO.\n\n"
            f"CONTEXTO:\n{context_text}\n\n"
        )
        for i, question in enumerate(current_questions, start=batch_index+1):
            prompt += f"{i}. {question}\n"
        prompt += "\nProporcione cada respuesta en el mismo formato numérico de las preguntas.\n"

        # Lógica de reintentos en caso de error (por ejemplo, rate limit)
        retries = 0
        max_retries = 7
        delay = 5
        answer_str = ""
        while retries < max_retries:
            try:
                answer_str = analizar_documento_solo_texto(prompt)
                break
            except Exception as e:
                if "429" in str(e):
                    time.sleep(delay)
                    delay *= 2
                    retries += 1
                else:
                    return jsonify({"error": f"Error en análisis: {str(e)}"}), 500

        if not answer_str:
            return jsonify({"error": "No se pudo obtener respuesta después de varios intentos."}), 500

        # Parsear la respuesta y mapearla a las preguntas
        answer_lines = answer_str.strip().split("\n")
        for line in answer_lines:
            parts = line.split(". ", 1)
            if len(parts) == 2 and parts[0].isdigit():
                num = int(parts[0])
                respuesta = parts[1].strip()
                if 1 <= num <= len(INCIDENT_QUESTIONS):
                    responses[INCIDENT_QUESTIONS[num - 1]] = respuesta

        time.sleep(2)

    # Generar archivo Excel para cada PDF subido
    excel_files = {}
    for pdf_filename in uploaded_files_info:
        try:
            excel_filename = save_answers_to_excel(pdf_filename, INCIDENT_QUESTIONS, responses)
            excel_files[pdf_filename] = excel_filename
        except Exception as e:
            excel_files[pdf_filename] = f"Error: {str(e)}"

    return jsonify({"responses": responses, "excel_files": excel_files}), 200

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """
    Endpoint para descargar un archivo Excel generado.
    """
    try:
        return send_file(filename, as_attachment=True)
    except Exception as e:
        return jsonify({"error": f"Error al descargar el archivo: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
