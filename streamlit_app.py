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

# 页面配置
st.set_page_config(
    page_title="动手学大模型应用开发", 
    page_icon="🦜🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        text-align: center;
        margin: 0;
    }
    .upload-section {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .chat-container {
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        padding: 1rem;
    }
    .sidebar-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    /* 消息按钮组样式 - 左下角排列 */
    .message-actions {
        display: flex;
        gap: 8px;
        margin-top: 15px;
        margin-bottom: 5px;
        justify-content: flex-start;
        align-items: center;
    }
    
    /* 统一的白框按钮样式 */
    .action-button {
        background: white;
        border: 2px solid #dee2e6;
        border-radius: 8px;
        padding: 8px 16px;
        cursor: pointer;
        font-size: 18px;
        color: #495057;
        transition: all 0.2s ease;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 50px;
        height: 40px;
        text-decoration: none;
        font-family: inherit;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .action-button:hover {
        background: #f8f9fa;
        border-color: #adb5bd;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    .action-button:active {
        transform: translateY(0);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .action-button.copied {
        background: white;
        border-color: #28a745;
        color: #28a745;
    }
    
    .action-button.loading {
        background: white;
        border-color: #ffc107;
        color: #ffc107;
    }
    
    /* 状态提示样式 */
    .status-message {
        font-size: 14px;
        color: #28a745;
        margin-left: 8px;
        display: none;
        align-items: center;
    }
    
    .status-message.show {
        display: inline-flex;
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
    
    # 用于存储重新生成的请求
    if "regenerate_question" not in st.session_state:
        st.session_state.regenerate_question = None
    
    if "regenerate_index" not in st.session_state:
        st.session_state.regenerate_index = None

# ---------- 简化的HTML按钮组 ----------
def create_action_buttons_html(message_index, message_text):
    """创建操作按钮组HTML"""
    # 转义文本中的特殊字符
    escaped_text = message_text.replace('\\', '\\\\').replace('`', '\\`').replace("'", "\\'").replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')
    
    buttons_html = f'''
    <div class="message-actions">
        <button onclick="copyToClipboard{message_index}()" 
                class="action-button"
                title="Copy to clipboard"
                id="copy-btn-{message_index}">
            📋
        </button>
        <button onclick="showRegenerateHint{message_index}()" 
                class="action-button"
                title="Click regenerate button below"
                id="regen-btn-{message_index}">
            🔄
        </button>
        <span id="status-{message_index}" class="status-message"></span>
    </div>
    
    <script>
    function copyToClipboard{message_index}() {{
        const text = `{escaped_text}`;
        const statusElement = document.getElementById('status-{message_index}');
        const button = document.getElementById('copy-btn-{message_index}');
        
        if (navigator.clipboard && window.isSecureContext) {{
            navigator.clipboard.writeText(text).then(function() {{
                button.classList.add('copied');
                statusElement.textContent = '✅ Copied';
                statusElement.classList.add('show');
                setTimeout(() => {{
                    button.classList.remove('copied');
                    statusElement.classList.remove('show');
                    statusElement.textContent = '';
                }}, 2000);
            }}).catch(function(err) {{
                fallbackCopy{message_index}(text, statusElement, button);
            }});
        }} else {{
            fallbackCopy{message_index}(text, statusElement, button);
        }}
    }}
    
    function fallbackCopy{message_index}(text, statusElement, button) {{
        const textArea = document.createElement('textarea');
        textArea.value = text;
        textArea.style.position = 'fixed';
        textArea.style.left = '-999999px';
        textArea.style.top = '-999999px';
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        
        try {{
            const successful = document.execCommand('copy');
            if (successful) {{
                button.classList.add('copied');
                statusElement.textContent = '✅ Copied';
                statusElement.classList.add('show');
            }} else {{
                statusElement.textContent = '❌ Failed';
                statusElement.classList.add('show');
            }}
        }} catch (err) {{
            statusElement.textContent = '❌ Failed';
            statusElement.classList.add('show');
        }}
        
        document.body.removeChild(textArea);
        setTimeout(() => {{
            button.classList.remove('copied');
            statusElement.classList.remove('show');
            statusElement.textContent = '';
        }}, 2000);
    }}
    
    function showRegenerateHint{message_index}() {{
        const statusElement = document.getElementById('status-{message_index}');
        const button = document.getElementById('regen-btn-{message_index}');
        
        button.classList.add('loading');
        statusElement.textContent = '👇 Click regenerate below';
        statusElement.classList.add('show');
        
        setTimeout(() => {{
            button.classList.remove('loading');
            statusElement.classList.remove('show');
            statusElement.textContent = '';
        }}, 3000);
    }}
    </script>
    '''
    
    return buttons_html

# ---------- 处理重新生成请求 ----------
def handle_regenerate_request():
    """处理重新生成回答的请求"""
    if st.session_state.regenerate_question is not None and st.session_state.regenerate_index is not None:
        question = st.session_state.regenerate_question
        message_index = st.session_state.regenerate_index
        
        try:
            # 找到对应的问题在messages中的位置
            if message_index > 0 and message_index < len(st.session_state.messages):
                # 移除要重新生成的AI回答
                st.session_state.messages = st.session_state.messages[:message_index]
                
                # 同样调整对话历史
                pairs_to_keep = message_index // 2
                st.session_state.chat_history = st.session_state.chat_history[:pairs_to_keep * 2]
                
                # 添加问题（如果还没有）
                if not st.session_state.messages or st.session_state.messages[-1][0] != "user":
                    st.session_state.messages.append(("user", question))
                    st.session_state.chat_history.append(HumanMessage(content=question))
                
                # 清除重新生成请求
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
        st.error(f"文件未找到: {file_path}")
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
            st.error(f"文件 {uploaded_file.name} 内容为空或无法提取")
            return ""
        
        st.success(f"成功提取 {uploaded_file.name}，内容长度: {len(content)} 字符")
        return content
        
    except Exception as e:
        st.error(f"处理文件 {uploaded_file.name} 时出错: {str(e)}")
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
        st.info(f"已加载 {len(st.session_state.uploaded_docs)} 个上传文档到知识库")
    
    if not all_docs:
        st.warning("没有找到任何文档内容，AI将仅使用自身知识回答问题")
        return None
    
    # 切分长文档
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.create_documents(all_docs)
    
    st.info(f"知识库已构建，包含 {len(docs)} 个文档片段")
    
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})

# ---------- 构建问答链（带记忆功能） ----------
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
        formatted = "\n\n".join(d.page_content for d in docs)
        return formatted
    
    def get_context_and_question(inputs):
        retriever = build_retriever()
        if retriever:
            try:
                context_docs = retriever.invoke(inputs["question"])
                context = format_docs(context_docs)
                if context != "没有找到相关的上下文信息。":
                    st.info(f"从知识库中找到 {len(context_docs)} 个相关文档片段")
                else:
                    st.info("未在知识库中找到相关信息，将使用AI一般知识回答")
            except Exception as e:
                st.warning(f"检索时出错: {e}")
                context = "检索出错，没有找到相关的上下文信息。"
        else:
            context = "没有找到相关的上下文信息。"
            st.info("知识库为空，将使用AI一般知识回答")
        
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

# ---------- 侧边栏功能 ----------
def setup_sidebar():
    with st.sidebar:
        st.markdown("### 📁 文件上传")
        
        uploaded_files = st.file_uploader(
            "上传文档文件",
            type=['txt', 'md', 'pdf', 'docx', 'doc'],
            accept_multiple_files=True,
            help="支持的格式：TXT, MD, PDF, DOCX, DOC"
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
                    with st.spinner(f"正在处理文件: {uploaded_file.name}"):
                        content = process_uploaded_file(uploaded_file)
                        if content:
                            st.session_state.uploaded_docs.append(content)
                            st.session_state.uploaded_files_info.append({
                                'name': uploaded_file.name,
                                'size': uploaded_file.size,
                                'content_length': len(content)
                            })
                            new_files_processed += 1
                            st.success(f"✅ {uploaded_file.name} 处理成功！")
                        else:
                            st.error(f"❌ {uploaded_file.name} 处理失败！")
            
            if new_files_processed > 0:
                st.session_state.chain = get_qa_chain_with_memory()
                st.success(f"🎉 成功处理 {new_files_processed} 个新文件！知识库已更新。")
                st.rerun()
        
        if 'uploaded_files_info' in st.session_state and st.session_state.uploaded_files_info:
            st.markdown("### 📋 已上传文件")
            for i, file_info in enumerate(st.session_state.uploaded_files_info):
                with st.expander(f"📄 {file_info['name']}", expanded=False):
                    st.write(f"**文件大小:** {file_info['size']} bytes")
                    st.write(f"**内容长度:** {file_info['content_length']} 字符")
                    
                    if 'uploaded_docs' in st.session_state and i < len(st.session_state.uploaded_docs):
                        preview = st.session_state.uploaded_docs[i][:200] + "..." if len(st.session_state.uploaded_docs[i]) > 200 else st.session_state.uploaded_docs[i]
                        st.text_area("内容预览:", preview, height=100, disabled=True)
        
        st.markdown("---")
        
        if 'uploaded_docs' in st.session_state:
            total_chars = sum(len(doc) for doc in st.session_state.uploaded_docs)
            st.markdown("### 📊 知识库状态")
            st.metric("文档数量", len(st.session_state.uploaded_docs))
            st.metric("总字符数", f"{total_chars:,}")
        
        st.markdown("---")
        
        if st.button("🗑️ 清除对话历史", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.regenerate_question = None
            st.session_state.regenerate_index = None
            st.success("对话历史已清除！")
            st.rerun()
        
        if st.button("📁 清除上传文件", use_container_width=True):
            if 'uploaded_docs' in st.session_state:
                del st.session_state.uploaded_docs
            if 'uploaded_files_info' in st.session_state:
                del st.session_state.uploaded_files_info
            st.session_state.chain = get_qa_chain_with_memory()
            st.success("所有上传文件已清除！")
            st.rerun()

# ---------- 生成AI回答的函数 ----------
def generate_ai_response(prompt, msgs):
    """生成AI回答"""
    try:
        chain_input = {
            "question": prompt,
            "chat_history": st.session_state.chat_history
        }
        
        with st.spinner("正在思考中..."):
            response = st.write_stream(st.session_state.chain.stream(chain_input))
        
        st.session_state.messages.append(("assistant", response))
        
        st.session_state.chat_history.extend([
            HumanMessage(content=prompt),
            AIMessage(content=response)
        ])
        
        if len(st.session_state.chat_history) > 20:
            st.session_state.chat_history = st.session_state.chat_history[-20:]
        
        # 添加按钮组
        message_index = len(st.session_state.messages) - 1
        
        # 使用列布局
        col1, col2 = st.columns([3, 7])
        
        with col1:
            # HTML按钮组
            buttons_html = create_action_buttons_html(message_index, response)
            st.components.v1.html(buttons_html, height=60)
        
        with col2:
            # Streamlit重新生成按钮
            if st.button("🔄 重新生成", key=f"regen_new_{message_index}", help="重新生成回答"):
                st.session_state.regenerate_question = prompt
                st.session_state.regenerate_index = message_index
                st.rerun()
                
    except Exception as e:
        error_msg = f"生成回答时出错: {str(e)}"
        st.error(error_msg)
        st.session_state.messages.append(("assistant", "抱歉，生成回答时出现了错误。请检查网络连接和API密钥。"))

# ---------- Streamlit 主界面 ----------
def main():
    initialize_session_state()
    
    st.markdown("""
    <div class="main-header">
        <h1>🦜🔗 重庆科技大学</h1>
    </div>
    """, unsafe_allow_html=True)
    
    setup_sidebar()
    
    if "chain" not in st.session_state:
        st.session_state.chain = get_qa_chain_with_memory()
    
    st.markdown("### 💬 智能问答")
    
    regenerate_question = handle_regenerate_request()
    
    msgs = st.container(height=500)
    
    # 显示聊天历史
    for i, (role, text) in enumerate(st.session_state.messages):
        with msgs.chat_message(role):
            st.write(text)
            
            if role == "assistant":
                question = None
                if i > 0 and st.session_state.messages[i-1][0] == "user":
                    question = st.session_state.messages[i-1][1]
                
                col1, col2 = st.columns([3, 7])
                
                with col1:
                    buttons_html = create_action_buttons_html(i, text)
                    st.components.v1.html(buttons_html, height=60)
                
                with col2:
                    if question is not None:
                        if st.button("🔄 重新生成", key=f"regen_history_{i}", help="重新生成回答"):
                            st.session_state.regenerate_question = question
                            st.session_state.regenerate_index = i
                            st.rerun()
    
    if regenerate_question:
        with msgs.chat_message("assistant"):
            st.info("🔄 正在重新生成回答...")
            generate_ai_response(regenerate_question, msgs)
        st.rerun()
    
    if prompt := st.chat_input("请输入你的问题..."):
        st.session_state.messages.append(("user", prompt))
        with msgs.chat_message("user"):
            st.write(prompt)
        
        with msgs.chat_message("assistant"):
            generate_ai_response(prompt, msgs)
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("对话轮数", len(st.session_state.messages) // 2)
    with col2:
        uploaded_count = len(st.session_state.get('uploaded_files_info', []))
        st.metric("已上传文件", uploaded_count)
    with col3:
        memory_count = len(st.session_state.chat_history) // 2
        st.metric("记忆对话数", memory_count)

if __name__ == "__main__":
    main()
