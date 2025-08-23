import streamlit as st
import google.generativeai as genai
import io
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="DocTalk",
    layout="wide"
)

# --- VALIDACIÓN DE SECRETOS ---
if "gcp_service_account" not in st.secrets or "GEMINI_API_KEY" not in st.secrets:
    st.error("🚨 ¡Error de configuración! Faltan secretos en tu aplicación de Streamlit.")
    st.warning(
        "Asegúrate de haber configurado `GEMINI_API_KEY` y la tabla `[gcp_service_account]` "
        "en los ajustes de tu aplicación."
    )
    st.stop()

# --- CONFIGURACIÓN DE APIS ---
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
    creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=SCOPES)
    drive_service = build("drive", "v3", credentials=creds)
except Exception as e:
    st.error(f"🚨 Error al configurar las APIs: {e}")
    st.stop()

# --- LÓGICA DE LA APLICACIÓN ---
@st.cache_data(ttl=300) # Cachea el contenido por 5 minutos
def get_google_doc_content(url):
    try:
        doc_id = url.split('/d/')[1].split('/')[0]
        request = drive_service.files().export_media(fileId=doc_id, mimeType="text/plain")
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        return fh.getvalue().decode('utf-8')
    except (IndexError, AttributeError):
        st.error("URL no válida. Asegúrate de que sea un enlace a un Google Doc.")
        return None
    except HttpError as error:
        st.error(
            f"Error al acceder al documento: {error.reason}. "
            "Verifica la URL y asegúrate de haber compartido el documento con el 'client_email' de la cuenta de servicio."
        )
        return None
    except Exception as e:
        st.error(f"Ocurrió un error inesperado: {e}")
        return None

# --- INTERFAZ DE LA APLICACIÓN ---
st.title("DocTalk")
st.markdown("Esta herramienta utiliza Gemini para analizar el contenido de un Google Doc y responder tus preguntas.")

# Contenedor para la entrada de datos
with st.container(border=True):
    st.subheader("1. Proporciona el Documento")
    url = st.text_input(
        "Pega la URL de tu Google Doc aquí",
        placeholder="https://docs.google.com/document/d/..."
    )

# Dos columnas para la pregunta y la respuesta
col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.subheader("2. Haz tu Pregunta")
        question = st.text_area(
            "¿Qué quieres saber sobre el documento?",
            height=150,
            placeholder="Ej: ¿Cuál es el estado actual del proyecto Hydra?"
        )
        submit_button = st.button("Analizar y Responder", type="primary", use_container_width=True)

with col2:
    with st.container(border=True):
        st.subheader("3. Respuesta de la IA")
        # Usamos el estado de la sesión para mantener la respuesta
        if 'response' not in st.session_state:
            st.session_state.response = "La respuesta aparecerá aquí..."
        
        if submit_button:
            if not url.strip() or not question.strip():
                st.warning("Por favor, introduce una URL y una pregunta.")
                st.session_state.response = "Esperando información..."
            else:
                with st.spinner("🔗 Accediendo al documento..."):
                    document_text = get_google_doc_content(url)
                
                if document_text:
                    st.success("📄 Documento leído.")
                    with st.spinner("🤖 Gemini está pensando..."):
                        try:
                            model = genai.GenerativeModel('gemini-1.5-flash-latest')
                            prompt = f"""
                            Tu tarea es actuar como un analista de inteligencia.
                            Analiza el siguiente documento, que está estructurado usando Markdown. Presta especial atención a los encabezados (#), listas (-) y texto en negrita (**) para entender la jerarquía y los datos clave.
                            Responde la pregunta del usuario de forma concisa y precisa, basándote únicamente en la información proporcionada.

                            --- DOCUMENTO ---
                            {document_text}
                            --- FIN DEL DOCUMENTO ---

                            PREGUNTA:
                            {question}
                            """
                            response = model.generate_content(prompt)
                            st.session_state.response = response.text
                        except Exception as e:
                            st.error(f"Ocurrió un error al contactar a Gemini: {e}")
                            st.session_state.response = "Error al generar la respuesta."
        
        st.markdown(st.session_state.response)
