import os
import torch
from dotenv import load_dotenv
from pathlib import Path
from huggingface_hub import snapshot_download
import sentencepiece as spm  # 🔹 Se usa en lugar de MistralTokenizer

# 🔹 IMPORTANTE: Importa Transformer y generate
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate

# Mistral imports
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

# HF Transformers imports
from transformers import AutoModelForCausalLM, AutoTokenizer


load_dotenv()
HF_TOKEN = os.getenv("HuggingTFG")  # Hugging Face token

# Ruta donde se descarga el modelo de Mistral
mistral_models_path = Path.home() / "mistral_models" / "7B-Instruct-v0.3"
mistral_models_path.mkdir(parents=True, exist_ok=True)

# Diccionario de modelos disponibles
AVAILABLE_MODELS = {
    "Mistral 7B": "mistralai/Mistral-7B-Instruct-v0.3",
    "Llama2 7B": "meta-llama/Llama-2-7b-chat-hf",
}

class HuggingFaceAPI:
    def __init__(self, model_name="Mistral 7B"):
        self.model_name = AVAILABLE_MODELS[model_name]
        self.is_mistral = (model_name == "Mistral 7B")

        if self.is_mistral:
            #
            # -------------- CARGAR MISTRAL CON SENTENCEPIECE --------------
            #
            # 1) Cargar el tokenizer desde el archivo `tokenizer.model`
            tokenizer_file = mistral_models_path / "tokenizer.model"
            self.tokenizer = spm.SentencePieceProcessor()
            if not self.tokenizer.Load(str(tokenizer_file)):
                raise ValueError("❌ Error al cargar el tokenizer de SentencePiece")

            # 2) Cargar los pesos del modelo desde la carpeta local
            self.model = Transformer.from_folder(str(mistral_models_path))  # Elimina `.to(device="cpu")`


        else:
            #
            # -------------- CARGAR MODELOS ESTÁNDAR DE HUGGING FACE --------------
            #
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_auth_token=HF_TOKEN
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                use_auth_token=HF_TOKEN
            )

    def generar_pregunta(self, full_text, question):
        """
        Genera una respuesta a una pregunta específica basada en el texto dado.
        """
        prompt = (
            f"Based on the following maritime incident report:\n\n"
            f"{full_text[:8000]}\n\n"
            f"Question: {question}\n\nAnswer:"
        )

        if self.is_mistral:
            #
            # -------------- GENERACIÓN CON MISTRAL --------------
            #
            completion_request = ChatCompletionRequest(
                messages=[UserMessage(content=prompt)]
            )

            # 🔹 Tokenizar usando SentencePiece en lugar de `MistralTokenizer`
            tokens = self.tokenizer.EncodeAsIds(prompt)

            # 🔹 Generar respuesta
            out_tokens, _ = generate(
                [tokens], 
                self.model,
                max_tokens=300,  
                temperature=0.5,
                eos_id=self.tokenizer.eos_id()
            )

            # 🔹 Decodificar tokens de salida
            result = self.tokenizer.DecodeIds(out_tokens[0])
            return result

        else:
            #
            # -------------- GENERACIÓN CON MODELOS HF --------------
            #
            inputs = self.tokenizer(prompt, return_tensors="pt")
            output = self.model.generate(**inputs, max_length=300, temperature=0.5)
            return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def generar_preguntas(self, full_text, tema, tipo):
        """
        Genera preguntas y respuestas sobre un texto dado.
        """
        prompt = (
            f"Genera preguntas del tipo '{tipo}' sobre el tema '{tema}' "
            f"usando el siguiente contenido:\n\n{full_text}\n\n"
            "Formato de salida:\n"
            "Pregunta: [Aquí va la pregunta]\n"
            "Respuesta: [Aquí va la respuesta]"
        )

        if self.is_mistral:
            #
            # -------------- GENERACIÓN MULTI-QA CON MISTRAL --------------
            #
            completion_request = ChatCompletionRequest(
                messages=[UserMessage(content=prompt)]
            )

            tokens = self.tokenizer.EncodeAsIds(prompt)

            out_tokens, _ = generate(
                [tokens],
                self.model,
                max_tokens=800,  
                temperature=0.7,
                eos_id=self.tokenizer.eos_id()
            )

            content = self.tokenizer.DecodeIds(out_tokens[0])

        else:
            #
            # -------------- GENERACIÓN MULTI-QA CON HUGGING FACE --------------
            #
            inputs = self.tokenizer(prompt, return_tensors="pt")
            output = self.model.generate(**inputs, max_length=1000, temperature=0.7)
            content = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return self._parse_questions_and_answers(content)

    def _parse_questions_and_answers(self, content):
        """
        Separa el texto generado en listas de preguntas y respuestas.
        """
        questions, answers = [], []
        current_answer = []
        capturing_answer = False

        for line in content.split("\n"):
            if line.startswith("Pregunta:"):
                if capturing_answer and current_answer:
                    answers.append(" ".join(current_answer).strip())
                    current_answer = []
                questions.append(line.replace("Pregunta:", "").strip())
                capturing_answer = True
            elif line.startswith("Respuesta:"):
                current_answer = [line.replace("Respuesta:", "").strip()]
            elif capturing_answer:
                current_answer.append(line.strip())

        if capturing_answer and current_answer:
            answers.append(" ".join(current_answer).strip())

        return questions, answers
