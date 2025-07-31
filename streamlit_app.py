import streamlit as st
import os
import tempfile
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.document_loaders import (
    TextLoader, 
    PyPDFLoader, 
    Docx2txtLoader,
    CSVLoader
)
import pandas as pd
from typing import List, Optional

# 页面配置
st.set_page_config(
    page_title="智能知识问答助手", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px 0;
        background: linear-gradient(90deg, #f0f2f6, #ffffff);
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .status-box {
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
</style>
""", unsafe_allow_html=True)

class EnhancedRAGSystem:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.memory = ConversationBufferWindowMemory(
            k=10,  # 保留最近10轮对话
            return_messages=True,
            memory_key="chat_history"
        )
        
    def initialize_embeddings(self):
        """初始化嵌入模型"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("❌ 请先设置环境变量 OPENAI_API_KEY")
            st.stop()
        
        if self.embeddings is None:
            self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        return self.embeddings
    
    def load_document(self, file_path: str, file_type: str) -> List[Document]:
        """根据文件类型加载文档"""
        try:
            if file_type.lower() == 'txt' or file_type.lower() == 'md':
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_type.lower() == 'pdf':
                loader = PyPDFLoader(file_path)
            elif file_type.lower() in ['docx', 'doc']:
                loader = Docx2txtLoader(file_path)
            elif file_type.lower() == 'csv':
                loader = CSVLoader(file_path)
            else:
                # 默认尝试文本加载
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return [Document(page_content=content, metadata={"source": file_path})]
            
            return loader.load()
        except Exception as e:
            st.error(f"❌ 加载文件失败: {str(e)}")
            return []
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """切分文档"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", "。", ".", " ", ""]
        )
        return text_splitter.split_documents(documents)
    
    def create_or_update_vectorstore(self, documents: List[Document]):
        """创建或更新向量存储"""
        if not documents:
            return
        
        embeddings = self.initialize_embeddings()
        
        if self.vectorstore is None:
            # 创建新的向量存储
            self.vectorstore = FAISS.from_documents(documents, embeddings)
            st.success(f"✅ 创建了新的知识库，包含 {len(documents)} 个文档片段")
        else:
            # 增量更新现有向量存储
            new_vectorstore = FAISS.from_documents(documents, embeddings)
            self.vectorstore.merge_from(new_vectorstore)
            st.success(f"✅ 知识库已更新，新增 {len(documents)} 个文档片段")
    
    def get_retriever(self):
        """获取检索器"""
        if self.vectorstore is None:
            return None
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}  # 返回最相关的3个片段
        )
    
    def build_qa_chain(self):
        """构建问答链"""
        retriever = self.get_retriever()
        if retriever is None:
            return None
        
        llm = ChatOpenAI(
            model_name="gpt-4o", 
            temperature=0.1, 
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # 改进的prompt模板，支持上下文记忆和自身知识回答
        system_template = """你是一个智能的AI助手。请根据以下规则回答问题：

1. 首先查看提供的知识库内容，如果能从中找到相关信息，请基于这些信息回答
2. 如果知识库中没有相关信息，请结合你自身的知识和经验来回答问题
3. 请结合对话历史来理解用户的问题，保持对话的连贯性
4. 回答要准确、有用且友好

知识库内容：
{context}

对话历史：
{chat_history}

请回答用户的问题。如果使用了知识库信息，请说明；如果使用了自身知识，也请适当说明。"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", "{question}")
        ])
        
        def format_chat_history(messages):
            """格式化聊天历史"""
            if not messages:
                return "暂无历史对话"
            
            formatted = []
            for msg in messages[-6:]:  # 只取最近6条消息
                if hasattr(msg, 'content'):
                    role = "用户" if msg.type == "human" else "助手"
                    formatted.append(f"{role}: {msg.content}")
            return "\n".join(formatted)
        
        def get_relevant_docs(question):
            """获取相关文档"""
            if retriever:
                docs = retriever.get_relevant_documents(question)
                return "\n\n".join([doc.page_content for doc in docs]) if docs else "知识库中暂无相关信息"
            return "知识库为空"
        
        chain = (
            {
                "context": lambda x: get_relevant_docs(x["question"]),
                "chat_history": lambda x: format_chat_history(self.memory.chat_memory.messages),
                "question": lambda x: x["question"]
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return chain
    
    def add_to_memory(self, question: str, answer: str):
        """添加到对话记忆"""
        self.memory.chat_memory.add_user_message(question)
        self.memory.chat_memory.add_ai_message(answer)

def main():
    # 初始化RAG系统
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = EnhancedRAGSystem()
    
    # 初始化聊天记录
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 主标题
    st.markdown('<div class="main-header">🤖 智能知识问答助手</div>', unsafe_allow_html=True)
    
    # 侧边栏 - 知识库管理
    with st.sidebar:
        st.header("📚 知识库管理")
        
        # 显示当前知识库状态
        rag_system = st.session_state.rag_system
        if rag_system.vectorstore is not None:
            doc_count = rag_system.vectorstore.index.ntotal
            st.markdown(f'<div class="status-box success-box">✅ 知识库已准备就绪<br>包含 {doc_count} 个文档片段</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-box warning-box">⚠️ 知识库为空<br>请上传文档来构建知识库</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 文件上传区域
        st.subheader("📁 上传文档")
        uploaded_files = st.file_uploader(
            "选择文件", 
            type=['txt', 'md', 'pdf', 'docx', 'doc', 'csv'],
            accept_multiple_files=True,
            help="支持多种格式：TXT, MD, PDF, DOCX, CSV"
        )
        
        if uploaded_files:
            if st.button("🚀 添加到知识库", use_container_width=True):
                with st.spinner("正在处理文档..."):
                    all_documents = []
                    
                    for uploaded_file in uploaded_files:
                        # 保存上传的文件到临时目录
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name
                        
                        # 加载文档
                        file_type = uploaded_file.name.split('.')[-1]
                        documents = rag_system.load_document(tmp_file_path, file_type)
                        
                        if documents:
                            # 添加文件名到元数据
                            for doc in documents:
                                doc.metadata["filename"] = uploaded_file.name
                            all_documents.extend(documents)
                            st.success(f"✅ {uploaded_file.name} 加载成功")
                        else:
                            st.error(f"❌ {uploaded_file.name} 加载失败")
                        
                        # 删除临时文件
                        os.unlink(tmp_file_path)
                    
                    if all_documents:
                        # 切分文档
                        split_docs = rag_system.split_documents(all_documents)
                        # 更新向量存储
                        rag_system.create_or_update_vectorstore(split_docs)
                        st.rerun()
        
        st.markdown("---")
        
        # 清除功能
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ 清空知识库", use_container_width=True):
                st.session_state.rag_system.vectorstore = None
                st.success("知识库已清空")
                st.rerun()
        
        with col2:
            if st.button("💭 清除记忆", use_container_width=True):
                st.session_state.rag_system.memory.clear()
                st.session_state.messages = []
                st.success("对话记忆已清除")
                st.rerun()
        
        # 使用说明
        st.markdown("---")
        st.subheader("📖 使用说明")
        st.markdown("""
        **功能特点：**
        - 🔍 智能检索：从知识库中查找相关信息
        - 🧠 自主回答：知识库无答案时使用AI自身知识
        - 💭 上下文记忆：保持对话连贯性
        - 📁 多格式支持：TXT、PDF、Word、CSV等
        - ⚡ 增量更新：随时添加新文档
        
        **使用步骤：**
        1. 上传相关文档构建知识库
        2. 开始提问，系统会智能匹配答案
        3. 支持连续对话和追问
        """)
    
    # 主聊天区域
    st.subheader("💬 智能问答")
    
    # 创建聊天容器
    chat_container = st.container(height=500)
    
    # 显示聊天历史
    with chat_container:
        for role, content in st.session_state.messages:
            with st.chat_message(role):
                st.write(content)
    
    # 聊天输入
    if prompt := st.chat_input("请输入您的问题..."):
        # 添加用户消息
        st.session_state.messages.append(("user", prompt))
        
        with chat_container:
            with st.chat_message("user"):
                st.write(prompt)
            
            with st.chat_message("assistant"):
                # 构建问答链
                qa_chain = rag_system.build_qa_chain()
                
                if qa_chain is not None:
                    # 流式输出响应
                    with st.spinner("思考中..."):
                        try:
                            response = qa_chain.invoke({"question": prompt})
                            st.write(response)
                            
                            # 添加到记忆和聊天历史
                            rag_system.add_to_memory(prompt, response)
                            st.session_state.messages.append(("assistant", response))
                            
                        except Exception as e:
                            error_msg = f"抱歉，处理您的问题时出现了错误：{str(e)}"
                            st.error(error_msg)
                            st.session_state.messages.append(("assistant", error_msg))
                else:
                    # 没有知识库时使用纯LLM回答
                    no_kb_msg = "知识库为空，让我用自己的知识来回答您的问题："
                    st.info(no_kb_msg)
                    
                    try:
                        llm = ChatOpenAI(
                            model_name="gpt-4o", 
                            temperature=0.3, 
                            openai_api_key=os.getenv("OPENAI_API_KEY")
                        )
                        
                        # 格式化历史对话
                        history_context = ""
                        if st.session_state.messages:
                            recent_messages = st.session_state.messages[-6:]  # 最近3轮对话
                            history_context = "\n".join([f"{'用户' if role == 'user' else '助手'}: {content}" for role, content in recent_messages])
                        
                        enhanced_prompt = f"""基于以下对话历史，请回答用户的最新问题：

对话历史：
{history_context}

最新问题：{prompt}

请提供准确、有用的回答。"""
                        
                        response = llm.invoke(enhanced_prompt).content
                        st.write(response)
                        
                        # 添加到记忆和聊天历史
                        rag_system.add_to_memory(prompt, response)
                        st.session_state.messages.append(("assistant", response))
                        
                    except Exception as e:
                        error_msg = f"抱歉，处理您的问题时出现了错误：{str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append(("assistant", error_msg))

    # 页面底部信息
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("💡 **提示**：上传相关文档可获得更准确的答案")
    with col2:
        st.markdown("🔄 **记忆**：系统会记住对话上下文")
    with col3:
        st.markdown("🤖 **智能**：无答案时使用AI自身知识")

if __name__ == "__main__":
    main()
