import streamlit as st
import google.generativeai as genai

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Chat con Texto",
    page_icon="🤖",
    layout="wide"
)

# --- CONFIGURACIÓN DE LA API DE GEMINI ---
# Para desplegar en Streamlit Community Cloud, debes configurar un "Secreto".
# 1. En la configuración de tu app en Streamlit, ve a "Settings" > "Secrets".
# 2. Pega tu clave de API de Gemini así:
#    GEMINI_API_KEY = "tu_clave_api_aqui"

try:
    # Intenta configurar la API key desde los secretos de Streamlit
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except (KeyError, FileNotFoundError):
    # Si falla, muestra un error claro en la interfaz
    st.error("🚨 ¡Error de configuración! No se encontró la clave de API de Gemini.")
    st.warning(
        "Por favor, asegúrate de haber configurado el 'Secreto' de Streamlit "
        "llamado `GEMINI_API_KEY` en los ajustes de tu aplicación."
    )
    # Detiene la ejecución del script si no hay API key
    st.stop()

# --- INTERFAZ DE LA APLICACIÓN ---

st.title("💬 Chatbot de Análisis de Texto")
st.markdown("Pega cualquier texto en el área de abajo y haz preguntas sobre él. La IA de Gemini te responderá basándose en el contenido.")

st.markdown("---")

# Columna 1: Entrada de Texto
st.subheader("Paso 1: Pega el contenido de tu documento aquí")
document_text = st.text_area(
    "Contenido del Documento",
    height=300,
    placeholder="Pega aquí el texto que quieres analizar...",
    label_visibility="collapsed"
)

st.markdown("---")

# Columna 2: Chat
st.subheader("Paso 2: Haz tu pregunta")
question = st.text_input(
    "Pregunta",
    placeholder="¿Qué quieres saber sobre el texto?",
    label_visibility="collapsed"
)

if st.button("Enviar Pregunta", type="primary", use_container_width=True):
    # Validaciones de entrada
    if not document_text.strip():
        st.warning("Por favor, pega el texto de un documento en el área de contenido.")
    elif not question.strip():
        st.warning("Por favor, escribe una pregunta.")
    else:
        # Si todo está bien, procede a llamar a la IA
        with st.spinner("🤖 Gemini está analizando el texto y pensando en tu respuesta..."):
            try:
                # Configura el modelo de IA
                model = genai.GenerativeModel('gemini-pro')

                # Crea el prompt con instrucciones claras
                prompt = f"""
                Analiza el siguiente texto y responde la pregunta del usuario.
                Tu respuesta debe basarse estricta y únicamente en la información contenida en el texto proporcionado.
                Si la respuesta no se puede encontrar en el texto, indícalo claramente.

                --- TEXTO PROPORCIONADO ---
                {document_text}
                --- FIN DEL TEXTO ---

                PREGUNTA DEL USUARIO:
                {question}
                """

                # Genera la respuesta
                response = model.generate_content(prompt)

                # Muestra la respuesta
                st.success("¡Respuesta generada!")
                st.markdown(response.text)

            except Exception as e:
                st.error(f"Ocurrió un error al contactar a la IA: {e}")