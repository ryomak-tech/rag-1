import streamlit as st
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

st.title("🧠 ハリポタ質問アプリ（クラウドLLM）")

# Hugging FaceのAPIトークンをsecretsから取得
HUGGINGFACE_API_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

# モデルの設定
llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",
    model_kwargs={"temperature": 0.5, "max_length": 512},
    huggingfacehub_api_token=HUGGINGFACE_API_TOKEN
)

# プロンプトテンプレート
prompt = PromptTemplate(
    input_variables=["question"],
    template="質問に答えてください：{question}"
)

chain = LLMChain(llm=llm, prompt=prompt)

query = st.text_input("質問を入力してください")
if query:
    with st.spinner("考え中..."):
        answer = chain.run(query)
        st.write("🧙 回答：", answer)