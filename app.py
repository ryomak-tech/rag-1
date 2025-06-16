from huggingface_hub import hf_hub_download
import streamlit as st
from pathlib import Path
import time

# ---- LangChain / llama-cpp imports ----
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA

# ---------- ここを自分の環境に合わせて ---------- #
PDF_PATH   = "ハリー・ポッターシリーズの登場人物一覧.pdf"
INDEX_DIR  = Path("store")                       # FAISS 保存先
MODEL_PATH = "models/swallow-13b-instruct.Q4_K_S.gguf"  # GGUF ファイル
N_GPU_LAYERS = 35    # GPU(またはApple Silicon) が無ければ 0
N_THREADS     = 8     # 物理コア数に合わせる
# -------------------------------------------------- #

st.title("📚 ハリポタ質問アプリ")
query = st.text_input("質問を入力してください")

# ----------- ベクトルストア（FAISS）を用意 ----------- #
@st.cache_resource(show_spinner="Vector DB を準備中…")
def load_vectorstore():
    """初回は PDF → 分割 → 埋め込み → FAISS 保存。2 回目以降はロードのみ。"""
    if (INDEX_DIR / "index.faiss").exists():
        return FAISS.load_local(
            folder_path=str(INDEX_DIR),
            embeddings=HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                encode_kwargs={"normalize_embeddings": True},
            ),
            allow_dangerous_deserialization=True       # ← これを忘れるとエラーになる
        )

    # ----- 初回だけ走る重い処理 -----
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=450, chunk_overlap=50
    )
    splits = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )
    vs = FAISS.from_documents(splits, embeddings)
    vs.save_local(INDEX_DIR)
    return vs

# ----------- LlamaCpp インスタンスを用意 ----------- #
@st.cache_resource(show_spinner="LLM をロード中…")
def build_llm():
    model_path = hf_hub_download(
    repo_id="itsryoma/swallow-13b-local",  # あなたのHFリポジトリ
    filename="swallow-13b-instruct.Q4_K_S.gguf",
    subfolder="models"
    )
    return LlamaCpp(
        model_path=MODEL_PATH,
        n_ctx=2048,
        max_tokens=256,
        temperature=0.2,
        n_threads=N_THREADS,
        n_batch=192,
        n_gpu_layers=N_GPU_LAYERS,   # CPU-only 環境なら 0
        verbose=False,
        stream=False                 # 逐次出力を使うなら True
    )

# ------------------- 推論 ------------------- #
if query:
    vs = load_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": 6})

    llm = build_llm()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    with st.spinner("🧙‍♀️ 考え中…"):
        t0 = time.time()
        result = qa_chain.invoke({"query": query})
        st.write("🧙 回答：", result["result"])
        st.caption(f"⏱️ {time.time()-t0:.1f} 秒で完了")