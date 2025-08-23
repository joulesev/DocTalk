import streamlit as st
import google.generativeai as genai
import io
import time
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

# --- NUEVAS LIBRERÍAS PARA RAG ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="IA de Base de Conocimiento",
    page_icon="📚",
    layout="wide"
)

# --- VALIDACIÓN DE SECRETOS Y CONFIGURACIÓN DE APIS ---
if "gcp_service_account" not in st.secrets or "GEMINI_API_KEY" not in st.secrets:
    st.error("🚨 ¡Error de configuración! Faltan secretos en tu aplicación.")
    st.stop()

try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
    creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=SCOPES)
    drive_service = build("drive", "v3", credentials=creds)
except Exception as e:
    st.error(f"🚨 Error al configurar las APIs: {e}")
    st.stop()

# --- LÓGICA DE LA APLICACIÓN ---

@st.cache_resource
def get_all_docs_from_folder(folder_id):
    """
    Escanea recursivamente una carpeta de Drive y devuelve una lista de todos los
    Google Docs encontrados, con su nombre y ID.
    """
    docs = []
    query = f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.document'"
    try:
        results = drive_service.files().list(q=query, fields="nextPageToken, files(id, name)").execute()
        docs.extend(results.get('files', []))

        # Búsqueda recursiva en subcarpetas
        folder_query = f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.folder'"
        subfolders = drive_service.files().list(q=folder_query, fields="files(id, name)").execute()
        for subfolder in subfolders.get('files', []):
            docs.extend(get_all_docs_from_folder(subfolder.get('id')))
        return docs
    except HttpError as error:
        st.error(f"Error al listar archivos: {error}")
        return []

@st.cache_data(ttl=600)
def get_doc_content(_doc_id):
    """Descarga el contenido de un Google Doc como texto plano."""
    try:
        request = drive_service.files().export_media(fileId=_doc_id, mimeType="text/plain")
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        return fh.getvalue().decode('utf-8')
    except HttpError as error:
        # Silenciamos errores de conversión para no detener el proceso
        print(f"No se pudo convertir el doc {_doc_id}: {error}")
        return ""

def create_vector_db(docs):
    """
    Toma una lista de documentos, los divide en fragmentos y crea una base de datos
    vectorial (FAISS) para búsquedas de similitud.
    """
    if not docs:
        return None
    
    with st.status("Construyendo base de conocimiento...", expanded=True) as status:
        all_texts = ""
        for i, doc in enumerate(docs):
            status.write(f"📄 Leyendo documento {i+1}/{len(docs)}: {doc['name']}...")
            content = get_doc_content(doc['id'])
            all_texts += content + "\n\n"
            time.sleep(0.1) # Pequeña pausa para que la UI se actualice

        status.write("쪼 Dividiendo textos en fragmentos...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(all_texts)
        
        status.write("🧠 Creando 'embeddings' (representaciones numéricas)...")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        status.write("💾 Construyendo el índice de búsqueda...")
        vector_db = FAISS.from_texts(chunks, embedding=embeddings)
        
        status.update(label="¡Base de conocimiento lista!", state="complete")
    
    return vector_db

# --- INTERFAZ DE LA APLICACIÓN ---
st.title("📚 IA de Base de Conocimiento (Google Drive)")
st.markdown("Proporciona una URL de una carpeta de Google Drive para crear una base de conocimiento y luego haz preguntas sobre su contenido.")

# Inicializar el estado de la sesión
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None

with st.container(border=True):
    st.subheader("1. Construir la Base de Conocimiento")
    folder_url = st.text_input(
        "Pega la URL de la carpeta principal de Google Drive aquí",
        placeholder="https://drive.google.com/drive/folders/..."
    )

    if st.button("Indexar Carpeta", type="primary", use_container_width=True):
        if folder_url:
            try:
                folder_id = folder_url.split('/')[-1]
                all_docs = get_all_docs_from_folder(folder_id)
                if all_docs:
                    st.session_state.vector_db = create_vector_db(all_docs)
                else:
                    st.warning("No se encontraron documentos de Google en la carpeta o subcarpetas.")
            except (IndexError, AttributeError):
                st.error("URL de carpeta no válida.")
        else:
            st.warning("Por favor, introduce una URL de carpeta.")

st.markdown("---")

with st.container(border=True):
    st.subheader("2. Haz tu Pregunta")
    question = st.text_area(
        "¿Qué quieres saber sobre el contenido de la carpeta?",
        height=100,
        disabled=(st.session_state.vector_db is None)
    )

    if st.button("Obtener Respuesta", use_container_width=True, disabled=(st.session_state.vector_db is None)):
        if question:
            with st.spinner("🧠 Buscando en la base de conocimiento y generando respuesta..."):
                # Configura el modelo de lenguaje de Gemini
                llm = genai.GenerativeModel('gemini-1.5-flash-latest')
                
                # Realiza la búsqueda de similitud y obtén los fragmentos relevantes
                retriever = st.session_state.vector_db.as_retriever()
                relevant_docs = retriever.get_relevant_documents(question)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])

                # Crea el prompt final
                prompt = f"""
                Actúa como un analista experto. Tu única fuente de verdad es el siguiente CONTEXTO.
                Responde la PREGUNTA del usuario de forma clara y concisa basándote exclusivamente en la información del CONTEXTO.
                Si la respuesta no se encuentra en el CONTEXTO, indica que no tienes suficiente información.

                --- CONTEXTO ---
                {context}
                --- FIN DEL CONTEXTO ---

                PREGUNTA: {question}
                """
                
                response = llm.generate_content(prompt)
                st.success("Respuesta generada:")
                st.markdown(response.text)
        else:
            st.warning("Por favor, escribe una pregunta.")

if st.session_state.vector_db is None:
    st.info("La sección de preguntas se habilitará una vez que la base de conocimiento esté indexada.")
