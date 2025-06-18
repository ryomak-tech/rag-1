import streamlit as st
import time
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# タイトル
st.title("🧠 ハリポタ質問アプリ（クラウドLLM）")

# 入力欄
query = st.text_input("質問を入力してください")

# secrets.toml に保存した Hugging Face の API トークンを読み込み
HUGGINGFACE_API_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

# LLM を準備
llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",  # 無料で軽いモデル
    model_kwargs={"temperature": 0.7, "max_length": 256},
    huggingfacehub_api_token=HUGGINGFACE_API_TOKEN
)

# プロンプトテンプレートを定義
template = """
次の質問に子どもでもわかるように答えてください。

質問: {question}
答え:
"""
prompt = PromptTemplate(input_variables=["question"], template=template)

# LLMChain を作成
chain = LLMChain(llm=llm, prompt=prompt)

# 入力があれば処理実行
if query:
    with st.spinner("🧙‍♂️ 考え中..."):
        t0 = time.time()
        answer = chain.run(query)
        st.success("🧙 回答：" + answer.strip())
        st.caption(f"⏱️ {time.time() - t0:.1f} 秒で完了")