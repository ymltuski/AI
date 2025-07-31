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
    
    /* 消息按钮组样式 */
    .message-actions {
        display: flex;
        gap: 8px;
        margin-top: 10px;
        margin-bottom: 10px;
        align-items: center;
    }
    
    /* 统一的按钮样式 */
    .action-button {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 6px;
        padding: 6px 12px;
        cursor: pointer;
        font-size: 14px;
        color: #495057;
        transition: all 0.2s ease;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 4px;
        min-width: 80px;
        text-decoration: none;
        font-family: inherit;
    }
    
    .action-button:hover {
        background: #e9ecef;
        border-color: #adb5bd;
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* 复制按钮样式 */
    .copy-button {
        background: #007bff;
        color: white;
        border-color: #007bff;
    }
    
    .copy-button:hover {
        background: #0056b3;
        border-color: #0056b3;
        color: white;
    }
    
    .copy-button.copied {
        background: #28a745;
        border-color: #28a745;
        color: white;
    }
    
    /* 点赞按钮样式 */
    .like-button {
        color: #6c757d;
    }
    
    .like-button:hover {
        color: #28a745;
        border-color: #28a745;
    }
    
    .like-button.liked {
        background: #d4edda;
        color: #155724;
        border-color: #c3e6cb;
    }
    
    /* 踩按钮样式 */
    .dislike-button {
        color: #6c757d;
    }
    
    .dislike-button:hover {
        color: #dc3545;
        border-color: #dc3545;
    }
    
    .dislike-button.disliked {
        background: #f8d7da;
        color: #721c24;
        border-color: #f5c6cb;
    }
    
    /* 重新生成按钮样式 */
    .regenerate-button {
        color: #6c757d;
    }
    
    .regenerate-button:hover {
        color: #17a2b8;
        border-color: #17a2b8;
    }
    
    /* 状态提示样式 */
    .status-message {
        font-size: 12px;
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
    
    if "message_ratings" not in st.session_state:
        st.session_state.message_ratings = {}  # 存储消息评分
    
    if "regenerating" not in st.session_state:
        st.session_state.regenerating = False
    
    if "last_question" not in st.session_state:
        st.session_state.last_question = ""

# ---------- HTML按钮组件 ----------
def create_message_actions_html(message_index, message_text, question=None):
    """创建消息操作按钮组的HTML"""
    # 转义文本中的特殊字符
    escaped_text = message_text.replace('\\', '\\\\').replace('`', '\\`').replace("'", "\\'").replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')
    
    # 获取当前评分状态
    current_rating = st.session_state.message_ratings.get(message_index, None)
    like_class = "liked" if current_rating == "like" else ""
    dislike_class = "disliked" if current_rating == "dislike" else ""
    
    # 构建按钮组HTML
    buttons_html = f'''
    <div class="message-actions">
        <!-- 复制按钮 -->
        <button id="copy-btn-{message_index}" class="action-button copy-button" onclick="copyMessage{message_index}()">
            📋 复制
        </button>
        
        <!-- 点赞按钮 -->
        <button id="like-btn-{message_index}" class="action-button like-button {like_class}" onclick="likeMessage{message_index}()">
            👍 点赞
        </button>
        
        <!-- 踩按钮 -->
        <button id="dislike-btn-{message_index}" class="action-button dislike-button {dislike_class}" onclick="dislikeMessage{message_index}()">
            👎 踩
        </button>
        
        <!-- 重新生成按钮（仅对AI回答显示） -->
        {"" if question is None else f'''
        <button id="regen-btn-{message_index}" class="action-button regenerate-button" onclick="regenerateMessage{message_index}()">
            🔄 重新回答
        </button>
        '''}
        
        <!-- 状态提示 -->
        <span id="status-{message_index}" class="status-message"></span>
    </div>
    
    <script>
    // 复制功能
    function copyMessage{message_index}() {{
        const text = `{escaped_text}`;
        const button = document.getElementById('copy-btn-{message_index}');
        const status = document.getElementById('status-{message_index}');
        
        if (navigator.clipboard && window.isSecureContext) {{
            navigator.clipboard.writeText(text).then(function() {{
                showCopySuccess{message_index}(button, status);
            }}).catch(function(err) {{
                fallbackCopy{message_index}(text, button, status);
            }});
        }} else {{
            fallbackCopy{message_index}(text, button, status);
        }}
    }}
    
    function fallbackCopy{message_index}(text, button, status) {{
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
                showCopySuccess{message_index}(button, status);
            }} else {{
                showError{message_index}(button, '复制失败');
            }}
        }} catch (err) {{
            showError{message_index}(button, '复制失败');
        }}
        
        document.body.removeChild(textArea);
    }}
    
    function showCopySuccess{message_index}(button, status) {{
        button.classList.add('copied');
        button.innerHTML = '✅ 已复制';
        
        setTimeout(function() {{
            button.classList.remove('copied');
            button.innerHTML = '📋 复制';
        }}, 2000);
    }}
    
    function showError{message_index}(button, message) {{
        button.innerHTML = '❌ ' + message;
        setTimeout(function() {{
            button.innerHTML = '📋 复制';
        }}, 2000);
    }}
    
    // 点赞功能
    function likeMessage{message_index}() {{
        // 通过Streamlit的方式触发Python函数
        const event = new CustomEvent('streamlit:likeMessage', {{
            detail: {{
                messageIndex: {message_index},
                action: 'like'
            }}
        }});
        window.dispatchEvent(event);
        
        // 更新UI状态
        const likeBtn = document.getElementById('like-btn-{message_index}');
        const dislikeBtn = document.getElementById('dislike-btn-{message_index}');
        
        if (likeBtn.classList.contains('liked')) {{
            likeBtn.classList.remove('liked');
        }} else {{
            likeBtn.classList.add('liked');
            dislikeBtn.classList.remove('disliked');
        }}
    }}
    
    // 踩功能
    function dislikeMessage{message_index}() {{
        const event = new CustomEvent('streamlit:dislikeMessage', {{
            detail: {{
                messageIndex: {message_index},
                action: 'dislike'
            }}
        }});
        window.dispatchEvent(event);
        
        // 更新UI状态
        const likeBtn = document.getElementById('like-btn-{message_index}');
        const dislikeBtn = document.getElementById('dislike-btn-{message_index}');
        
        if (dislikeBtn.classList.contains('disliked')) {{
            dislikeBtn.classList.remove('disliked');
        }} else {{
            dislikeBtn.classList.add('disliked');
            likeBtn.classList.remove('liked');
        }}
    }}
    
    {"" if question is None else f'''
    // 重新生成功能
    function regenerateMessage{message_index}() {{
        const event = new CustomEvent('streamlit:regenerateMessage', {{
            detail: {{
                messageIndex: {message_index},
                question: `{question.replace('`', '\\`').replace("'", "\\'").replace('"', '\\"') if question else ""}`
            }}
        }});
        window.dispatchEvent(event);
        
        // 显示加载状态
        const button = document.getElementById('regen-btn-{message_index}');
        button.innerHTML = '⏳ 生成中...';
        button.disabled = true;
    }}
    '''}
    </script>
    '''
    
    return buttons_html

# ---------- 消息评分功能 ----------
def handle_message_rating(message_index, rating):
    """处理消息评分"""
    current_rating = st.session_state.message_ratings.get(message_index, None)
    
    if current_rating == rating:
        # 如果已经是相同评分，则取消评分
        del st.session_state.message_ratings[message_index]
    else:
        # 设置新评分
        st.session_state.message_ratings[message_index] = rating
    
    # 这里可以添加日志记录或数据收集逻辑
    print(f"Message {message_index} rated: {st.session_state.message_ratings.get(message_index, 'none')}")

# ---------- 重新生成回答功能 ----------
def regenerate_answer(question):
    """重新生成回答"""
    st.session_state.regenerating = True
    st.session_state.last_question = question
    
    # 移除最后一条AI回答
    if st.session_state.messages and st.session_state.messages[-1][0] == "assistant":
        st.session_state.messages.pop()
        
    # 移除对话历史中的最后一条AI消息
    if st.session_state.chat_history and isinstance(st.session_state.chat_history[-1], AIMessage):
        st.session_state.chat_history.pop()

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
            # 保存临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            
            content = docx2txt.process(tmp_file_path)
            os.unlink(tmp_file_path)  # 删除临时文件
        else:
            st.error(f"不支持的文件格式: {file_extension}")
            return ""
        
        # 验证内容是否成功提取
        if not content or len(content.strip()) == 0:
            st.error(f"文件 {uploaded_file.name} 内容为空或无法提取")
            return ""
        
        st.success(f"成功提取 {uploaded_file.name}，内容长度: {len(content)} 字符")
        return content
        
    except Exception as e:
        st.error(f"处理文件 {uploaded_file.name} 时出错: {str(e)}")
        return ""

# ---------- 测试检索器功能 ----------
def test_retriever(question="测试"):
    """测试检索器是否正常工作"""
    try:
        retriever = build_retriever()
        if retriever:
            docs = retriever.invoke(question)
            return docs
        return []
    except Exception as e:
        st.error(f"测试检索器时出错: {e}")
        return []

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
    
    # 改进的系统提示，允许模型在找不到相关信息时使用自身知识
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
        # 每次调用时重新构建检索器，确保使用最新的文档
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
        
        # 文件上传器
        uploaded_files = st.file_uploader(
            "上传文档文件",
            type=['txt', 'md', 'pdf', 'docx', 'doc'],
            accept_multiple_files=True,
            help="支持的格式：TXT, MD, PDF, DOCX, DOC"
        )
        
        if uploaded_files:
            # 初始化会话状态
            if 'uploaded_docs' not in st.session_state:
                st.session_state.uploaded_docs = []
            if 'uploaded_files_info' not in st.session_state:
                st.session_state.uploaded_files_info = []
            
            # 处理新上传的文件
            existing_files = [info['name'] for info in st.session_state.uploaded_files_info]
            new_files_processed = 0
            
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in existing_files:
                    with st.spinner(f"正在处理文件: {uploaded_file.name}"):
                        content = process_uploaded_file(uploaded_file)
                        if content:
                            # 添加文档内容
                            st.session_state.uploaded_docs.append(content)
                            
                            # 保存文件信息
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
                # 重新构建chain（每次都会重新构建检索器）
                st.session_state.chain = get_qa_chain_with_memory()
                st.success(f"🎉 成功处理 {new_files_processed} 个新文件！知识库已更新。")
                st.rerun()
        
        # 显示已上传的文件
        if 'uploaded_files_info' in st.session_state and st.session_state.uploaded_files_info:
            st.markdown("### 📋 已上传文件")
            for i, file_info in enumerate(st.session_state.uploaded_files_info):
                with st.expander(f"📄 {file_info['name']}", expanded=False):
                    st.write(f"**文件大小:** {file_info['size']} bytes")
                    st.write(f"**内容长度:** {file_info['content_length']} 字符")
                    
                    # 显示文档内容预览
                    if 'uploaded_docs' in st.session_state and i < len(st.session_state.uploaded_docs):
                        preview = st.session_state.uploaded_docs[i][:200] + "..." if len(st.session_state.uploaded_docs[i]) > 200 else st.session_state.uploaded_docs[i]
                        st.text_area("内容预览:", preview, height=100, disabled=True)
        
        st.markdown("---")
        
        # 显示知识库状态
        if 'uploaded_docs' in st.session_state:
            total_chars = sum(len(doc) for doc in st.session_state.uploaded_docs)
            st.markdown("### 📊 知识库状态")
            st.metric("文档数量", len(st.session_state.uploaded_docs))
            st.metric("总字符数", f"{total_chars:,}")
            
            # 测试检索功能
            if st.button("🔍 测试知识库检索", use_container_width=True):
                test_query = st.text_input("输入测试问题:", value="学生手册", key="test_query")
                if test_query:
                    with st.spinner("正在测试检索..."):
                        docs = test_retriever(test_query)
                        if docs:
                            st.success(f"✅ 检索成功！找到 {len(docs)} 个相关片段")
                            with st.expander("查看检索结果"):
                                for i, doc in enumerate(docs):
                                    st.write(f"**片段 {i+1}:**")
                                    st.write(doc.page_content[:300] + "...")
                        else:
                            st.warning("⚠️ 未找到相关内容")
        
        st.markdown("---")
        
        # 清除对话历史按钮
        if st.button("🗑️ 清除对话历史", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.message_ratings = {}
            st.success("对话历史已清除！")
            st.rerun()
        
        # 清除上传文件按钮
        if st.button("📁 清除上传文件", use_container_width=True):
            if 'uploaded_docs' in st.session_state:
                del st.session_state.uploaded_docs
            if 'uploaded_files_info' in st.session_state:
                del st.session_state.uploaded_files_info
            st.session_state.chain = get_qa_chain_with_memory()
            st.success("所有上传文件已清除！")
            st.rerun()
        
        st.markdown("---")
        
        # 使用说明
        with st.expander("📖 使用说明"):
            st.markdown("""
            **功能特点：**
            - 🔍 智能检索：从知识库中查找相关信息  
            - 🧠 知识融合：找不到时使用AI自身知识回答
            - 💭 对话记忆：记住之前的对话内容
            - 📁 文件上传：支持多种格式文档
            - 📋 一键复制：直接点击复制AI回答到剪贴板
            - 👍👎 评分系统：对回答进行点赞或踩
            - 🔄 重新回答：不满意可重新生成回答
            
            **使用方法：**
            1. 上传相关文档文件（会自动处理并加入知识库）
            2. 在下方输入框中提问
            3. AI会结合文档内容和对话历史回答
            4. 使用底部按钮进行复制、评分或重新生成
            
            **注意事项：**
            - 文件上传后会自动构建知识库
            - 大文件处理可能需要几秒钟时间
            - 支持同时上传多个文件
            - 评分数据会用于改进服务质量
            - 复制功能支持现代浏览器的一键复制
            """)

# ---------- 处理JavaScript事件 ----------
def handle_javascript_events():
    """处理来自JavaScript的事件"""
    # 这里使用查询参数来模拟JavaScript事件处理
    # 在实际应用中，可能需要使用更复杂的方式来处理JavaScript和Python之间的通信
    
    query_params = st.query_params
    
    # 处理点赞事件
    if 'like_msg' in query_params:
        try:
            message_index = int(query_params['like_msg'])
            handle_message_rating(message_index, 'like')
            # 清除查询参数
            del st.query_params['like_msg']
            st.rerun()
        except:
            pass
    
    # 处理踩事件
    if 'dislike_msg' in query_params:
        try:
            message_index = int(query_params['dislike_msg'])
            handle_message_rating(message_index, 'dislike')
            # 清除查询参数
            del st.query_params['dislike_msg']
            st.rerun()
        except:
            pass
    
    # 处理重新生成事件
    if 'regen_msg' in query_params and 'regen_question' in query_params:
        try:
            question = query_params['regen_question']
            regenerate_answer(question)
            # 清除查询参数
            del st.query_params['regen_msg']
            del st.query_params['regen_question']
            st.rerun()
        except:
            pass

# ---------- Streamlit 主界面 ----------
def main():
    # 初始化会话状态
    initialize_session_state()
    
    # 处理JavaScript事件
    handle_javascript_events()
    
    # 页面标题
    st.markdown("""
    <div class="main-header">
        <h1>🦜🔗 重庆科技大学</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # 设置侧边栏
    setup_sidebar()
    
    # 初始化链
    if "chain" not in st.session_state:
        st.session_state.chain = get_qa_chain_with_memory()
    
    # 主聊天区域
    st.markdown("### 💬 智能问答")
    
    # 聊天消息容器
    msgs = st.container(height=500)
    
    # 显示聊天历史
    for i, (role, text) in enumerate(st.session_state.messages):
        with msgs.chat_message(role):
            st.write(text)
            
            # 为AI回答添加HTML按钮组
            if role == "assistant":
                # 寻找对应的用户问题
                question = None
                if i > 0 and st.session_state.messages[i-1][0] == "user":
                    question = st.session_state.messages[i-1][1]
                
                # 渲染HTML按钮组
                buttons_html = create_message_actions_html(i, text, question)
                st.components.v1.html(buttons_html, height=60)
    
    # 处理重新生成回答
    if st.session_state.regenerating:
        with msgs.chat_message("assistant"):
            try:
                # 准备输入数据
                chain_input = {
                    "question": st.session_state.last_question,
                    "chat_history": st.session_state.chat_history
                }
                
                # 显示处理状态
                with st.spinner("正在重新生成回答..."):
                    # 流式输出回答
                    response = st.write_stream(st.session_state.chain.stream(chain_input))
                
                # 保存新消息到历史记录
                st.session_state.messages.append(("assistant", response))
                
                # 更新对话历史
                st.session_state.chat_history.append(AIMessage(content=response))
                
                # 为新回答添加HTML按钮组
                new_message_index = len(st.session_state.messages) - 1
                buttons_html = create_message_actions_html(new_message_index, response, st.session_state.last_question)
                st.components.v1.html(buttons_html, height=60)
                
                # 重置重新生成状态
                st.session_state.regenerating = False
                st.session_state.last_question = ""
                
            except Exception as e:
                error_msg = f"重新生成回答时出错: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append(("assistant", "抱歉，重新生成回答时出现了错误。"))
                st.session_state.regenerating = False
                st.rerun()
    
    # 用户输入
    if prompt := st.chat_input("请输入你的问题..."):
        # 添加用户消息
        st.session_state.messages.append(("user", prompt))
        with msgs.chat_message("user"):
            st.write(prompt)
        
        # 生成AI回答
        with msgs.chat_message("assistant"):
            try:
                # 准备输入数据，包含对话历史
                chain_input = {
                    "question": prompt,
                    "chat_history": st.session_state.chat_history
                }
                
                # 显示处理状态
                with st.spinner("正在思考中..."):
                    # 流式输出回答
                    response = st.write_stream(st.session_state.chain.stream(chain_input))
                
                # 保存消息到历史记录
                st.session_state.messages.append(("assistant", response))
                
                # 更新对话历史（用于记忆功能）
                st.session_state.chat_history.extend([
                    HumanMessage(content=prompt),
                    AIMessage(content=response)
                ])
                
                # 限制对话历史长度，避免token过多
                if len(st.session_state.chat_history) > 20:
                    st.session_state.chat_history = st.session_state.chat_history[-20:]
                
                # 为新回答添加HTML按钮组
                message_index = len(st.session_state.messages) - 1
                buttons_html = create_message_actions_html(message_index, response, prompt)
                st.components.v1.html(buttons_html, height=60)
                    
            except Exception as e:
                error_msg = f"生成回答时出错: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append(("assistant", "抱歉，生成回答时出现了错误。请检查网络连接和API密钥。"))
                # 显示详细错误信息供调试
                with st.expander("错误详情"):
                    st.code(str(e))
    
    # 底部信息
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
