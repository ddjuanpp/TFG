# manager.py

import random

class ExamManager:
    def __init__(self, agent_rag):
        self.agent_rag = agent_rag
        self.questions = []
        self.answers = []

    def resolver_ticket(self, description):
        """
        Resuelve un ticket utilizando Claude y procesa la solución.
        """
        # Obtener datos de solución desde Claude
        solution_data = self.agent_rag.resolver_ticket(description)

        if isinstance(solution_data, str):
            return f"Error resolving ticket: {solution_data}"

        # Procesar los datos de solución y asegurarse de que sea un string único
        solution_text = ""
        if isinstance(solution_data.get('content'), list):
            # Concatenar todos los textos en un único string si es una lista
            solution_text = " ".join(
                item.get('text', '') for item in solution_data['content'] if item.get('type') == 'text'
            )
        else:
            solution_text = solution_data.get('content', 'No resolution found')

        return solution_text

    def create_development_questions(self, full_text, count=5):
        """
        Genera preguntas de desarrollo a partir de frases o temáticas detectadas en el texto.
        """
        sentences = [s.strip() for s in full_text.split('.') if len(s.strip()) > 30]
        selected = random.sample(sentences, min(count, len(sentences)))
        
        for s in selected:
            question = f"Explica en detalle: '{s}'"
            answer = f"Respuesta elaborada sobre: {s}"
            self.questions.append(question)
            self.answers.append(answer)
        return self.questions, self.answers

    def create_true_false_questions(self, full_text, count=5):
        """
        Genera preguntas de verdadero/falso, donde la 'respuesta' podría ser 'Verdadero' o 'Falso'.
        """
        sentences = [s.strip() for s in full_text.split('.') if len(s.strip()) > 30]
        selected = random.sample(sentences, min(count, len(sentences)))
        
        for s in selected:
            question = f"'{s}'. ¿Verdadero o Falso?"
            # De manera simplificada, asignamos al azar la respuesta como 'Verdadero' o 'Falso'.
            # En la práctica, deberías analizar la oración para determinar su veracidad.
            answer = random.choice(["Verdadero", "Falso"])
            self.questions.append(question)
            self.answers.append(answer)
        return self.questions, self.answers

    def create_short_questions(self, full_text, count=5):
        """
        Genera preguntas cortas (con respuestas breves) a partir del texto.
        """
        sentences = [s.strip() for s in full_text.split('.') if len(s.strip()) > 30]
        selected = random.sample(sentences, min(count, len(sentences)))
        
        for s in selected:
            # Separamos por comas a modo de ejemplo; se puede refinar la lógica
            parts = s.split(',')
            if len(parts) >= 2:
                question = f"¿{parts[0]}?"
                answer = ','.join(parts[1:]).strip()
            else:
                question = f"Describe brevemente: '{s[:50]}'..."
                answer = f"Respuesta corta sobre: {s[:50]}..."
            
            self.questions.append(question)
            self.answers.append(answer)
        return self.questions, self.answers

    def combine_claude_and_local(self, description, question_type="desarrollo"):
        """
        Combina la extracción de texto desde Claude con la generación local de preguntas.
        """
        # 1. Resolvemos ticket con Claude para obtener el texto (full_text).
        full_text = self.resolver_ticket(description)

        if "Error" in full_text:
            return [], [], full_text  # Devuelve el error si algo falla

        # 2. Dependiendo del tipo de pregunta, generamos de forma local:
        if question_type.lower() == "desarrollo":
            qs, ans = self.create_development_questions(full_text)
        elif question_type.lower() in ["verdadero/falso", "verdadero falso"]:
            qs, ans = self.create_true_false_questions(full_text)
        elif question_type.lower() == "preguntas cortas":
            qs, ans = self.create_short_questions(full_text)
        else:
            qs, ans = [], []
        
        return qs, ans, "Preguntas generadas exitosamente."
