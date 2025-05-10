import os
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from openai import AzureOpenAI
import uuid

# Disable Chroma telemetry and avoid SQLite usage for compatibility
os.environ["CHROMA_TELEMETRY"] = "False"

# Load environment variables
load_dotenv()
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# Initialize OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# Use HuggingFace for local embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Streamlit UI
st.set_page_config(page_title="Azure RAG with Chroma", layout="wide")
st.title("📚 Chat with your PDF (Chroma + GPT-4)")

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_files = st.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    docs = []
    for uploaded_file in uploaded_files:
        file_path = f"tmp_{uuid.uuid4().hex}.pdf"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        docs.extend(pages)
        os.remove(file_path)

    st.success("✅ All PDFs uploaded and loaded!")

    # Split and store
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Use in-memory Chroma to avoid sqlite3 issues on cloud platforms
    vectordb = Chroma.from_documents(chunks, embedding_model)
    st.success("✅ Content embedded and stored in Chroma (in-memory mode).")

    # User query input
    query = st.text_input("❓ Ask a question about the PDFs")
    if query:
        with st.spinner("Thinking..."):
            retrieved_docs = vectordb.similarity_search(query, k=3)
            context = "\n\n".join([f"- {doc.page_content[:300]}..." for doc in retrieved_docs])

            response = client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
                ],
                temperature=0.3,
                max_tokens=500
            )
            answer = response.choices[0].message.content

            # Update chat history
            st.session_state.chat_history.append((query, answer))

    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 💬 Chat History")
        for i, (q, a) in enumerate(reversed(st.session_state.chat_history)):
            st.markdown(f"**Q{i+1}:** {q}")
            st.markdown(f"_A{i+1}:_ {a}")
