import streamlit as st
import os
import tempfile
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage
import docx2txt
import PyPDF2
import io
import time
import json

# 页面配置 - 优化布局
st.set_page_config(
    page_title="重庆科技大学智能问答系统", 
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 优化的CSS样式 - 确保单页显示
st.markdown("""
<style>
    /* 重置默认样式，确保单页显示 */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
        overflow: hidden;
    }
    
    /* 紧凑的标题样式 */
    .custom-title {
        font-size: 28px;
        font-weight: 700;
        text-align: center;
        padding: 0.8rem;
        color: white;
        background: linear-gradient(to right, #667eea, #764ba2);
        border-radius: 8px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* 优化聊天容器 */
    .chat-container {
        height: 60vh;
        overflow-y: auto;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 0.5rem;
        background: #fafafa;
    }
    
    /* 优化消息气泡 */
    .message-bubble {
        background-color: #f0f2f6;
        padding: 0.8rem;
        border-radius: 10px;
        margin: 0.3rem 0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        line-height: 1.4;
    }
    
    .assistant-bubble {
        background-color: #e6f0ff;
    }
    
    /* 紧凑的按钮样式 */
    .action-buttons {
        display: flex;
        gap: 8px;
        margin-top: 0.5rem;
        align-items: center;
    }
    
    .copy-button {
        background: white;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 4px 8px;
        cursor: pointer;
        font-size: 14px;
        color: #666;
        transition: all 0.2s ease;
        min-width: 60px;
        height: 28px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .copy-button:hover {
        background: #f8f9fa;
        border-color: #adb5bd;
        color: #333;
    }
    
    /* 侧边栏优化 */
    .sidebar .stSelectbox {
        margin-bottom: 0.5rem;
    }
    
    /* 输入框样式 */
    .stChatInput {
        margin-top: 0.5rem;
    }
    
    /* 隐藏不必要的空白 */
    .element-container {
        margin-bottom: 0.5rem;
    }
    
    /* 优化分隔线 */
    hr {
        margin: 0.5rem 0;
        border: 0;
        height: 1px;
        background: #e0e0e0;
    }
    
    /* 确保内容不会被遮挡 */
    .main {
        overflow: visible;
    }
    
    /* 紧凑的指标显示 */
    .metric-container {
        padding: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------- 初始化会话状态 ----------
def initialize_session_state():
    """初始化会话状态变量"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    if "regenerate_question" not in st.session_state:
        st.session_state.regenerate_question = None
        
    if "regenerate_index" not in st.session_state:
        st.session_state.regenerate_index = None

# ---------- 创建紧凑的复制按钮 ----------
def create_compact_copy_button(message_index, message_text):
    """创建紧凑的复制按钮"""
    escaped_text = message_text.replace('\\', '\\\\').replace('`', '\\`').replace("'", "\\'").replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')

    copy_html = f'''
    <div style="display: flex; align-items: center; gap: 8px; margin: 3px 0;">
        <button onclick="copyText{message_index}()" class="copy-button" 
                style="background: white; border: 1px solid #ddd; border-radius: 4px; padding: 4px 8px; cursor: pointer; font-size: 12px; color: #666; min-width: 50px; height: 24px;">
            📋 复制
        </button>
        <span id="status-{message_index}" style="color: #28a745; font-size: 11px;"></span>
    </div>

    <script>
    function copyText{message_index}() {{
        const text = `{escaped_text}`;
        const status = document.getElementById('status-{message_index}');
        
        if (navigator.clipboard) {{
            navigator.clipboard.writeText(text).then(() => {{
                status.textContent = '✅ 已复制';
                setTimeout(() => status.textContent = '', 2000);
            }});
        }} else {{
            const textarea = document.createElement('textarea');
            textarea.value = text;
            document.body.appendChild(textarea);
            textarea.select();
            document.execCommand('copy');
            document.body.removeChild(textarea);
            status.textContent = '✅ 已复制';
            setTimeout(() => status.textContent = '', 2000);
        }}
    }}
    </script>
    '''
    return copy_html

# ---------- 处理重新生成请求 ----------
def handle_regenerate_request():
    """处理重新生成回答的请求"""
    if st.session_state.regenerate_question is not None and st.session_state.regenerate_index is not None:
        question = st.session_state.regenerate_question
        message_index = st.session_state.regenerate_index
                
        try:
            if message_index > 0 and message_index < len(st.session_state.messages):
                st.session_state.messages = st.session_state.messages[:message_index]
                pairs_to_keep = message_index // 2
                st.session_state.chat_history = st.session_state.chat_history[:pairs_to_keep * 2]
                                
                if not st.session_state.messages or st.session_state.messages[-1][0] != "user":
                    st.session_state.messages.append(("user", question))
                    st.session_state.chat_history.append(HumanMessage(content=question))
                                
                st.session_state.regenerate_question = None
                st.session_state.regenerate_index = None
                return question
                        
        except Exception as e:
            st.error(f"处理重新生成请求时出错: {e}")
            st.session_state.regenerate_question = None
            st.session_state.regenerate_index = None
        
    return None

# ---------- 从本地 Markdown 文件获取文档内容 ----------
def fetch_document_from_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return ""
    except Exception as e:
        st.error(f"无法读取文件: {e}")
        return ""

# ---------- 处理上传文件 ----------
def process_uploaded_file(uploaded_file):
    """处理上传的文件并提取文本内容"""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
                
        if file_extension == 'txt':
            content = str(uploaded_file.read(), "utf-8")
        elif file_extension == 'md':
            content = str(uploaded_file.read(), "utf-8")
        elif file_extension == 'pdf':
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            content = ""
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                content += f"\n[第{page_num+1}页]\n{page_text}"
        elif file_extension in ['docx', 'doc']:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
                        
            content = docx2txt.process(tmp_file_path)
            os.unlink(tmp_file_path)
        else:
            st.error(f"不支持的文件格式: {file_extension}")
            return ""
                
        if not content or len(content.strip()) == 0:
            st.error(f"文件 {uploaded_file.name} 内容为空")
            return ""
                
        return content
            
    except Exception as e:
        st.error(f"处理文件时出错: {str(e)}")
        return ""

# ---------- 构建检索器 ----------
def build_retriever():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("请先设置环境变量 OPENAI_API_KEY")
        st.stop()
        
    all_docs = []
        
    # 从本地文件获取文档内容
    DOCUMENT_FILE_PATH = "测试.md"
    if os.path.exists(DOCUMENT_FILE_PATH):
        raw_docs = fetch_document_from_file(DOCUMENT_FILE_PATH)
        if raw_docs:
            all_docs.append(raw_docs)
        
    # 添加上传的文档内容
    if 'uploaded_docs' in st.session_state and st.session_state.uploaded_docs:
        all_docs.extend(st.session_state.uploaded_docs)
        
    if not all_docs:
        return None
        
    # 切分长文档
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.create_documents(all_docs)
        
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})

# ---------- 构建问答链 ----------
def get_qa_chain_with_memory():
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
        
    system = (
        "你是一个乐于助人的 AI 助手。\n"
        "请首先基于下面提供的上下文信息回答问题。如果上下文中包含相关信息，请优先使用这些信息。"
        "如果上下文中没有相关信息，请使用你的知识和经验来回答问题，并在回答开头说明'基于我的一般知识'。\n"
        "请保持回答的准确性和有用性。\n\n"
        "上下文信息:\n{context}\n\n"
        "请结合对话历史和上下文信息来回答用户的问题。"
    )
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
        
    def format_docs(docs):
        if not docs:
            return "没有找到相关的上下文信息。"
        return "\n\n".join(d.page_content for d in docs)
        
    def get_context_and_question(inputs):
        retriever = build_retriever()
        if retriever:
            try:
                context_docs = retriever.invoke(inputs["question"])
                context = format_docs(context_docs)
            except Exception as e:
                context = "检索出错，没有找到相关的上下文信息。"
        else:
            context = "没有找到相关的上下文信息。"
                
        return {
            "context": context,
            "question": inputs["question"],
            "chat_history": inputs["chat_history"]
        }
        
    chain = (
        get_context_and_question
        | prompt
        | llm
        | StrOutputParser()
    )
        
    return chain

# ---------- 紧凑的侧边栏 ----------
def setup_compact_sidebar():
    with st.sidebar:
        st.markdown("### 📁 文件上传")
                
        uploaded_files = st.file_uploader(
            "上传文档",
            type=['txt', 'md', 'pdf', 'docx', 'doc'],
            accept_multiple_files=True
        )
                
        if uploaded_files:
            if 'uploaded_docs' not in st.session_state:
                st.session_state.uploaded_docs = []
            if 'uploaded_files_info' not in st.session_state:
                st.session_state.uploaded_files_info = []
                        
            existing_files = [info['name'] for info in st.session_state.uploaded_files_info]
            new_files_processed = 0
                        
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in existing_files:
                    content = process_uploaded_file(uploaded_file)
                    if content:
                        st.session_state.uploaded_docs.append(content)
                        st.session_state.uploaded_files_info.append({
                            'name': uploaded_file.name,
                            'size': uploaded_file.size,
                            'content_length': len(content)
                        })
                        new_files_processed += 1
                        
            if new_files_processed > 0:
                st.session_state.chain = get_qa_chain_with_memory()
                st.success(f"✅ 处理了 {new_files_processed} 个文件")
                st.rerun()
                
        # 紧凑的文件列表显示
        if 'uploaded_files_info' in st.session_state and st.session_state.uploaded_files_info:
            st.markdown("### 📋 已上传文件")
            for file_info in st.session_state.uploaded_files_info:
                st.text(f"📄 {file_info['name'][:20]}...")
                
        # 知识库状态
        if 'uploaded_docs' in st.session_state:
            total_chars = sum(len(doc) for doc in st.session_state.uploaded_docs)
            st.markdown("### 📊 知识库状态")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("文档", len(st.session_state.uploaded_docs))
            with col2:
                st.metric("字符", f"{total_chars//1000}K")
                
        st.markdown("---")
                
        # 操作按钮
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ 清除对话", use_container_width=True):
                st.session_state.messages = []
                st.session_state.chat_history = []
                st.rerun()
        
        with col2:
            if st.button("📁 清除文件", use_container_width=True):
                if 'uploaded_docs' in st.session_state:
                    del st.session_state.uploaded_docs
                if 'uploaded_files_info' in st.session_state:
                    del st.session_state.uploaded_files_info
                st.rerun()

# ---------- 生成AI回答 ----------
def generate_ai_response(prompt, container):
    """生成AI回答"""
    try:
        chain_input = {
            "question": prompt,
            "chat_history": st.session_state.chat_history
        }
                
        with st.spinner("思考中..."):
            response = st.write_stream(st.session_state.chain.stream(chain_input))
                
        st.session_state.messages.append(("assistant", response))
                
        st.session_state.chat_history.extend([
            HumanMessage(content=prompt),
            AIMessage(content=response)
        ])
                
        if len(st.session_state.chat_history) > 20:
            st.session_state.chat_history = st.session_state.chat_history[-20:]
                
        # 添加操作按钮
        message_index = len(st.session_state.messages) - 1
        
        col1, col2, col3 = st.columns([2, 2, 6])
        
        with col1:
            copy_html = create_compact_copy_button(message_index, response)
            st.components.v1.html(copy_html, height=35)
                
        with col2:
            if st.button("🔄 重新生成", key=f"regen_{message_index}"):
                st.session_state.regenerate_question = prompt
                st.session_state.regenerate_index = message_index
                st.rerun()
                    
    except Exception as e:
        st.error(f"生成回答时出错: {str(e)}")

# ---------- 主函数 ----------
def main():
    # 初始化
    initialize_session_state()
    
    # 紧凑的标题
    st.markdown('<div class="custom-title">🌐 重庆科技大学 · 智能问答系统</div>', unsafe_allow_html=True)

    # 设置侧边栏
    setup_compact_sidebar()

    # 初始化问答链
    if "chain" not in st.session_state:
        st.session_state.chain = get_qa_chain_with_memory()

    # 处理重新生成请求
    regenerate_question = handle_regenerate_request()

    # 聊天区域 - 使用固定高度容器
    st.markdown("### 💬 智能问答")
    
    # 创建聊天容器，设置合适的高度
    chat_container = st.container(height=400)

    # 显示聊天历史
    with chat_container:
        for i, (role, text) in enumerate(st.session_state.messages):
            avatar = "🧑‍💻" if role == "user" else "🚀"
            
            with st.chat_message(role, avatar=avatar):
                # 使用紧凑的样式显示消息
                bubble_class = "message-bubble" + (" assistant-bubble" if role == "assistant" else "")
                st.markdown(f'<div class="{bubble_class}">{text}</div>', unsafe_allow_html=True)

                # 为助手消息添加操作按钮
                if role == "assistant":
                    question = st.session_state.messages[i-1][1] if i > 0 and st.session_state.messages[i-1][0] == "user" else None
                    
                    col1, col2, _ = st.columns([2, 2, 6])
                    with col1:
                        copy_html = create_compact_copy_button(i, text)
                        st.components.v1.html(copy_html, height=30)
                    with col2:
                        if question and st.button("🔄", key=f"regen_history_{i}", help="重新生成"):
                            st.session_state.regenerate_question = question
                            st.session_state.regenerate_index = i
                            st.rerun()

    # 处理重新生成
    if regenerate_question:
        with chat_container:
            with st.chat_message("assistant", avatar="🚀"):
                generate_ai_response(regenerate_question, chat_container)
        st.rerun()

    # 用户输入
    if prompt := st.chat_input("请输入你的问题..."):
        st.session_state.messages.append(("user", prompt))
        
        with chat_container:
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(f'<div class="message-bubble">{prompt}</div>', unsafe_allow_html=True)
            
            with st.chat_message("assistant", avatar="🚀"):
                generate_ai_response(prompt, chat_container)

if __name__ == "__main__":
    main()
