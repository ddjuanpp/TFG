# Generador y Corrector de Exámenes con IA

Esta herramienta permite **generar preguntas y corregir exámenes** utilizando **inteligencia artificial generativa (Claude en AWS Bedrock)**.  

El sistema procesa **documentos en formato PDF, DOCX y PPTX**, extrae su contenido y genera preguntas de distintos tipos. También ofrece corrección automática y feedback interactivo.

---

## **1. Instalación y configuración**

### **1.1. Requisitos previos**
Para ejecutar este programa, necesitas:

- **Python 3.9 o superior** → [Descargar aquí](https://www.python.org/downloads/)  
- **Acceso a Internet** (para conectarse a AWS Bedrock)  
- **Cuenta de AWS con Bedrock habilitado** → [Crear cuenta](https://aws.amazon.com/)  

### **1.2. Instalación de librerías necesarias**
Para instalar todas las dependencias, ejecuta en la terminal:

pip install -r requirements.txt

### **1.3. Conexión con AWS**
Para usar claude necesitas tener cuenta con AWS, ejecuta en la terminal:

aws configure

Introduce lo siguiente:

aws_access_key_id=TU_ACCESS_KEY
aws_secret_access_key=TU_SECRET_KEY
region=us-east-1

### **1.4. .Env**
En el .env debes tener las siguientes variables de entorno:

ANTHROPIC_API_URL
CLAUDE_MODEL_ARN
CLAUDE_MODEL_ID

---

## **2. Uso de la herramienta**
Para usar la herramienta ejecuta en la terminal:

streamlit run app.py

Una vez iniciado en el navegador, introduce tus documentos en la pestaña correspondiente
Espera a que aparezca "Documentos cargados exitosamente"

Elige en el panel de la derecha el ámbito de las preguntas, todo el temario o eligiendo tu de que tratarán
Elige también el tipo de pregunta que te gustaría ver.

Una vez hayas seleccionado todo a tu gusto, selecciona "Generar Preguntas"

Si ya tienes tus preguntas con sus soluciones, puedes volver a generar de otro tipo cuando quieras o subir más documentos