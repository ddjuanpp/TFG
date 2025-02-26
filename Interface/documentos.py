# documentos.py

from PyPDF2 import PdfReader

class DocumentUploader:
    def __init__(self):
        self.documents = []

    def add_document(self, file):
        """Leer documento y guardar su texto (solo PDF)"""
        try:
            if file.name.endswith(".pdf"):
                text = self._extract_text_from_pdf(file)
            else:
                raise ValueError("Formato de archivo no soportado. Solo se permiten archivos PDF.")
            self.documents.append(text)
        except Exception as e:
            raise ValueError(f"Error al procesar el archivo {file.name}: {e}")

    def _extract_text_from_pdf(self, file):
        """Extraer texto de un archivo PDF"""
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    def get_documents(self):
        """Retornar lista de documentos cargados"""
        return self.documents

    def get_concatenated_text(self):
        """Concatenar texto de todos los documentos"""
        return " ".join(self.documents)
