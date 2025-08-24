import streamlit as st
import google.generativeai as genai
import io
import time
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

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

# --- LÓGICA DE LA APLICACIÓN (CORREGIDA) ---

@st.cache_resource
def get_all_docs_from_folder(folder_id):
    """
    Escanea recursivamente una carpeta de Drive y devuelve una lista de todos los
    Google Docs y archivos de texto (.md, .txt) encontrados.
    """
    docs = []
    # Query ampliado para incluir Google Docs Y archivos de texto plano/markdown
    query = f"'{folder_id}' in parents and (mimeType='application/vnd.google-apps.document' or mimeType='text/plain' or mimeType='text/markdown')"
    try:
        results = drive_service.files().list(q=query, fields="nextPageToken, files(id, name, mimeType)").execute()
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
def get_doc_content(doc_object):
    """
    Descarga el contenido de un archivo de Drive, usando el método correcto
    según su tipo (Google Doc vs. archivo de texto).
    """
    try:
        file_id = doc_object['id']
        mime_type = doc_object['mimeType']
        
        # Si es un Google Doc, se debe "exportar"
        if mime_type == 'application/vnd.google-apps.document':
            request = drive_service.files().export_media(fileId=file_id, mimeType="text/plain")
        # Si es un archivo de texto/markdown, se debe "descargar"
        else:
            request = drive_service.files().get_media(fileId=file_id)
        
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        return fh.getvalue().decode('utf-8')
    except HttpError as error:
        print(f"No se pudo procesar el archivo {doc_object.get('name', file_id)}: {error}")
        return ""

def create_vector_db(docs):
    """
    Toma una lista de documentos, los divide en fragmentos y crea una base de datos
    vectorial para búsquedas de similitud.
    """
    # Importaciones de LangChain aquí para mantener el código organizado
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS

    if not docs:
        return None
    
    with st.status("Construyendo base de conocimiento...", expanded=True) as status:
        all_texts_with_metadata = []
        for i, doc in enumerate(docs):
            status.write(f"📄 Leyendo documento {i+1}/{len(docs)}: {doc['name']}...")
            content = get_doc_content(doc)
            if content:
                # Añadimos el nombre del archivo como metadato
                # Esto es útil para saber de dónde viene la información
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = text_splitter.create_documents([content], metadatas=[{"source": doc['name']}])
                all_texts_with_metadata.extend(chunks)
            time.sleep(0.1)

        if not all_texts_with_metadata:
             st.warning("No se pudo leer contenido de ningún documento.")
             return None

        status.write("🧠 Creando 'embeddings' (representaciones numéricas)...")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        status.write("💾 Construyendo el índice de búsqueda...")
        vector_db = FAISS.from_documents(all_texts_with_metadata, embedding=embeddings)
        
        status.update(label="¡Base de conocimiento lista!", state="complete")
    
    return vector_db

# --- INTERFAZ DE LA APLICACIÓN ---
st.title("📚 IA de Base de Conocimiento (Google Drive)")
st.markdown("Proporciona una URL de una carpeta de Google Drive para crear una base de conocimiento y luego haz preguntas sobre su contenido.")

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
                # Limpia la URL para obtener solo el ID de la carpeta
                folder_id = folder_url.split('/')[-1].split('?')[0]
                all_docs = get_all_docs_from_folder(folder_id)
                if all_docs:
                    st.session_state.vector_db = create_vector_db(all_docs)
                else:
                    st.warning("No se encontraron documentos de Google o archivos de texto (.md, .txt) en la carpeta o subcarpetas.")
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
                from langchain_google_genai import ChatGoogleGenerativeAI
                
                llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
                retriever = st.session_state.vector_db.as_retriever()
                
                # Usamos una cadena de LangChain para manejar el flujo de Q&A
                from langchain.chains.question_answering import load_qa_chain
                chain = load_qa_chain(llm, chain_type="stuff")
                
                relevant_docs = retriever.get_relevant_documents(question)
                
                response = chain.invoke({"input_documents": relevant_docs, "question": question})
                
                st.success("Respuesta generada:")
                st.markdown(response['output_text'])

                with st.expander("Ver fuentes utilizadas"):
                    sources = {doc.metadata['source'] for doc in relevant_docs}
                    for source in sources:
                        st.write(f"- {source}")
        else:
            st.warning("Por favor, escribe una pregunta.")

if st.session_state.vector_db is None:
    st.info("La sección de preguntas se habilitará una vez que la base de conocimiento esté indexada.")
