import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.set_page_config(page_title="动手学大模型应用开发", page_icon="🦜🔗")

# ---------- 1. 从本地 Markdown 文件获取文档内容 ----------
def fetch_document_from_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        st.error(f"文件未找到: {file_path}")
        st.stop()
    except Exception as e:
        st.error(f"无法读取文件: {e}")
        st.stop()
    return ""

# Markdown 文件路径（替换为实际的 Markdown 文件路径）
DOCUMENT_FILE_PATH = "测试.md"  # 使用测试.md文件

def build_retriever():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("请先设置环境变量 OPENAI_API_KEY")
        st.stop()

    # 从本地文件获取文档内容
    raw_docs = fetch_document_from_file(DOCUMENT_FILE_PATH)
    if not raw_docs:
        st.error("文档内容为空")
        st.stop()

    # 切分长文档
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents([raw_docs])

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever()

# ---------- 2. 构建问答链 ----------
def get_qa_chain():
    retriever = build_retriever()
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))

    system = (
        "你是一个乐于助人的 AI 助手。\n"
        "请使用下面的上下文回答问题，如果不知道就说“我不知道”。\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{question}")
    ])
    chain = (
        {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
         "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# ---------- 3. Streamlit 主界面 ----------
def main():
    st.markdown("### 🦜🔗 动手学大模型应用开发")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chain" not in st.session_state:
        st.session_state.chain = get_qa_chain()

    msgs = st.container(height=550)
    for role, text in st.session_state.messages:
        msgs.chat_message(role).write(text)

    if prompt := st.chat_input("请输入你的问题"):
        st.session_state.messages.append(("user", prompt))
        msgs.chat_message("user").write(prompt)

        with msgs.chat_message("assistant"):
            response = st.write_stream(st.session_state.chain.stream(prompt))
        st.session_state.messages.append(("assistant", response))

if __name__ == "__main__":
    main()
