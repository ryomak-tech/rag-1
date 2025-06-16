import streamlit as st
from pathlib import Path
import time

# LangChain / Transformers
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

PDF_PATH = "ハリー・ポッターシリーズの登場人物一覧.pdf"
INDEX_DIR = Path("store")

st.title("📚 ハリポタ質問アプリ")
query = st.text_input("質問を入力してください")

# ベクトルストア準備
@st.cache_resource(show_spinner="Vector DB を準備中…")
def load_vectorstore():
    if (INDEX_DIR / "index.faiss").exists():
        return FAISS.load_local(
            folder_path=str(INDEX_DIR),
            embeddings=HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                encode_kwargs={"normalize_embeddings": True},
            ),
            allow_dangerous_deserialization=True
        )
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=50)
    splits = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )
    vs = FAISS.from_documents(splits, embeddings)
    vs.save_local(INDEX_DIR)
    return vs

# LLM（無料モデル）準備
@st.cache_resource(show_spinner="LLM をロード中…")
def build_llm():
    tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-rw-1b")
    model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-rw-1b")
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.7,
    )
    return HuggingFacePipeline(pipeline=pipe)

# 推論
if query:
    vs = load_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": 6})
    llm = build_llm()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

    with st.spinner("🧙‍♀️ 考え中…"):
        t0 = time.time()
        result = qa_chain.invoke({"query": query})
        st.write("🧙 回答：", result["result"])
        st.caption(f"⏱️ {time.time()-t0:.1f} 秒で完了")