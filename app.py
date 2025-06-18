import streamlit as st
from pathlib import Path
import time

# LangChain / LlamaCpp
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA

# ▼ 自分の環境に合わせて
PDF_PATH = "ハリー・ポッターシリーズの登場人物一覧.pdf"
MODEL_PATH = "models/swallow-13b-instruct.Q4_K_S.gguf"
INDEX_DIR = Path("store")
N_GPU_LAYERS = 35  # Apple Siliconなら35くらい、なければ0
N_THREADS = 8      # CPUの物理コア数

st.title("🧙‍♂️ ハリーポッター質問アプリ")
query = st.text_input("質問を入力してください")

# ベクトルDB（初回は生成）
@st.cache_resource(show_spinner="🔍 ベクトルDBを準備中…")
def load_vectorstore():
    if (INDEX_DIR / "index.faiss").exists():
        return FAISS.load_local(
            folder_path=str(INDEX_DIR),
            embeddings=HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                encode_kwargs={"normalize_embeddings": True}
            ),
            allow_dangerous_deserialization=True
        )
    
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )

    vs = FAISS.from_documents(splits, embeddings)
    vs.save_local(INDEX_DIR)
    return vs

# LLM（ローカル）
@st.cache_resource(show_spinner="🧠 モデル読み込み中…")
def build_llm():
    return LlamaCpp(
        model_path=MODEL_PATH,
        n_ctx=2048,
        max_tokens=256,
        temperature=0.2,
        n_threads=N_THREADS,
        n_batch=192,
        n_gpu_layers=N_GPU_LAYERS,
        verbose=False
    )

# 推論実行
if query:
    vs = load_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": 6})
    llm = build_llm()
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

    with st.spinner("🧙‍♀️ 考え中…"):
        t0 = time.time()
        result = qa.invoke({"query": query})
        st.success("🪄 回答：" + result["result"])
        st.caption(f"⏱️ {time.time() - t0:.1f} 秒で完了")