import React, { useState } from 'react';
import './App.css';

function App() {
  const [selectedFiles, setSelectedFiles] = useState(null);
  const [uploadMessage, setUploadMessage] = useState('');
  const [analysisResults, setAnalysisResults] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Maneja la selección de archivos
  const handleFileChange = (e) => {
    setSelectedFiles(e.target.files);
  };

  // Envía los archivos PDF al endpoint /upload
  const handleUpload = async () => {
    if (!selectedFiles || selectedFiles.length === 0) {
      setUploadMessage("Por favor, selecciona al menos un archivo PDF.");
      return;
    }
    setIsUploading(true);
    setUploadMessage("Subiendo archivos...");

    const formData = new FormData();
    for (let i = 0; i < selectedFiles.length; i++) {
      formData.append("files", selectedFiles[i]);
    }

    try {
      const response = await fetch("http://localhost:5000/upload", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (response.ok) {
        setUploadMessage("Archivos subidos correctamente.");
      } else {
        setUploadMessage("Error: " + data.error);
      }
    } catch (error) {
      console.error("Error al subir archivos:", error);
      setUploadMessage("Error al subir archivos.");
    }
    setIsUploading(false);
  };

  // Llama al endpoint /analyze para procesar los archivos y obtener resultados
  const handleAnalyze = async () => {
    setIsAnalyzing(true);
    try {
      const response = await fetch("http://localhost:5000/analyze", {
        method: "POST",
      });
      const data = await response.json();
      if (response.ok) {
        setAnalysisResults(data);
      } else {
        setUploadMessage("Error en el análisis: " + data.error);
      }
    } catch (error) {
      console.error("Error al analizar:", error);
      setUploadMessage("Error al analizar.");
    }
    setIsAnalyzing(false);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Integración React con Flask</h1>
      </header>
      <main>
        <section>
          <h2>Subir archivos PDF</h2>
          <input type="file" multiple accept="application/pdf" onChange={handleFileChange} />
          <button onClick={handleUpload} disabled={isUploading}>
            {isUploading ? "Subiendo..." : "Subir"}
          </button>
          <p>{uploadMessage}</p>
        </section>
        <section>
          <h2>Analizar Documentos</h2>
          <button onClick={handleAnalyze} disabled={isAnalyzing}>
            {isAnalyzing ? "Analizando..." : "Analizar"}
          </button>
        </section>
        {analysisResults && (
          <section>
            <h2>Resultados del Análisis</h2>
            {analysisResults.responses && (
              <div>
                <h3>Respuestas:</h3>
                <ul>
                  {Object.entries(analysisResults.responses).map(([question, answer], index) => (
                    <li key={index}>
                      <strong>{question}:</strong> {answer}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {analysisResults.excel_files && (
              <div>
                <h3>Archivos Excel:</h3>
                <ul>
                  {Object.entries(analysisResults.excel_files).map(([pdfFile, excelFile], index) => (
                    <li key={index}>
                      {pdfFile}:{' '}
                      <a 
                        href={`http://localhost:5000/download/${excelFile}`} 
                        target="_blank" 
                        rel="noopener noreferrer">
                        Descargar {excelFile}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
