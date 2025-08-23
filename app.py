import streamlit as st
import google.generativeai as genai
import io
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Chat con Google Docs",
    page_icon="🤖",
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
@st.cache_data(ttl=600)
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
st.title("💬 Chatbot para Google Docs")
st.markdown("Pega la URL de un documento de Google Drive y haz preguntas sobre su contenido.")
st.markdown("---")

url = st.text_input(
    "Paso 1: Pega la URL de tu Google Doc aquí",
    placeholder="https://docs.google.com/document/d/..."
)
question = st.text_area(
    "Paso 2: Haz tu pregunta sobre el documento",
    height=150,
    placeholder="¿Qué quieres saber?"
)

if st.button("Enviar Pregunta", type="primary", use_container_width=True):
    if not url.strip() or not question.strip():
        st.warning("Por favor, introduce una URL y una pregunta.")
    else:
        with st.spinner("🔗 Accediendo al documento..."):
            document_text = get_google_doc_content(url)
        
        if document_text:
            st.success("📄 Documento leído correctamente.")
            with st.spinner("🤖 Gemini está analizando y pensando..."):
                try:
                    # --- LÍNEA CORREGIDA ---
                    model = genai.GenerativeModel('gemini-1.5-flash-latest')
                    
                    prompt = f"""
                    Analiza el siguiente texto y responde la pregunta.
                    Tu respuesta debe basarse estricta y únicamente en la información del texto.
                    Si la respuesta no se encuentra en el texto, indícalo claramente.

                    --- TEXTO DEL DOCUMENTO ---
                    {document_text}
                    --- FIN DEL TEXTO ---

                    PREGUNTA:
                    {question}
                    """
                    response = model.generate_content(prompt)
                    st.markdown("---")
                    st.subheader("Respuesta de Gemini:")
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"Ocurrió un error al contactar a Gemini: {e}")
