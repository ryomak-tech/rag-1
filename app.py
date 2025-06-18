import streamlit as st
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import time

st.title("🧠 ハリポタ質問アプリ（クラウドLLM）")

query = st.text_input("質問を入力してください")

# Hugging Face トークン（Streamlit CloudのSecretsに設定）
HUGGINGFACE_API_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

# プロンプトテンプレート
template = """
次の質問に、子どもにもわかるように答えてください。

質問: {question}
答え:
"""
prompt = PromptTemplate(input_variables=["question"], template=template)

# LLM設定
llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",  # flan-t5-base より安定
    model_kwargs={"temperature": 0.7, "max_length": 256},
    huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
    task="text2text-generation"  # ← これがないとクラッシュすることがあります
)

chain = LLMChain(llm=llm, prompt=prompt)

# 推論
if query:
    with st.spinner("🧙‍♂️ 考え中..."):
        t0 = time.time()
        try:
            answer = chain.run(query)
            st.success("🧙 回答：" + answer.strip())
            st.caption(f"⏱️ {time.time() - t0:.1f} 秒で完了")
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")