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
    
    /* AI回答容器样式 */
    .ai-message-container {
        position: relative;
        padding-bottom: 40px; /* 为按钮预留空间 */
    }
    
    /* 按钮容器样式 - 放在右下角 */
    .message-actions {
        position: absolute;
        bottom: 8px;
        right: 8px;
        display: flex;
        gap: 8px;
        z-index: 10;
    }
    
    /* 复制按钮样式 */
    .action-icon-button {
        background: none;
        border: none;
        cursor: pointer;
        font-size: 18px;
        padding: 6px;
        border-radius: 4px;
        transition: all 0.2s;
        opacity: 0.6;
        display: flex;
        align-items: center;
        justify-content: center;
        min-width: 32px;
        min-height: 32px;
    }
    
    .action-icon-button:hover {
        opacity: 1;
        background-color: rgba(0, 0, 0, 0.05);
        transform: scale(1.1);
    }
    
    .action-icon-button.copied {
        opacity: 1;
        background-color: rgba(76, 175, 80, 0.1);
        color: #4CAF50;
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

# ---------- 消息评分功能 ----------
def handle_message_rating(message_index, rating):
    """处理消息评分"""
    st.session_state.message_ratings[message_index] = rating
    # 这里可以添加日志记录或数据收集逻辑
    print(f"Message {message_index} rated: {rating}")

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
            4. 使用右下角按钮进行复制、评分或重新生成
            
            **注意事项：**
            - 文件上传后会自动构建知识库
            - 大文件处理可能需要几秒钟时间
            - 支持同时上传多个文件
            - 评分数据会用于改进服务质量
            - 复制功能支持现代浏览器的一键复制
            """)

# ---------- 简化的消息交互组件 ----------
def render_message_actions_v2(message_index, message_text, question=None):
    """渲染消息交互按钮 - 右下角复制，下方Streamlit按钮"""
    
    # 只保留复制按钮的HTML版本
    escaped_text = message_text.replace('\\', '\\\\').replace('`', '\\`').replace("'", "\\'").replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')
    
    copy_html = f'''
    <div class="message-actions">
        <button id="copy-btn-{message_index}" class="action-icon-button" onclick="copyText{message_index}()" title="复制">
            📋
        </button>
    </div>
    
    <script>
    function copyText{message_index}() {{
        const text = `{escaped_text}`;
        
        if (navigator.clipboard && window.isSecureContext) {{
            navigator.clipboard.writeText(text).then(function() {{
                showCopySuccess{message_index}();
            }}).catch(function(err) {{
                fallbackCopy{message_index}(text);
            }});
        }} else {{
            fallbackCopy{message_index}(text);
        }}
    }}
    
    function fallbackCopy{message_index}(text) {{
        const textArea = document.createElement('textarea');
        textArea.value = text;
        textArea.style.position = 'fixed';
        textArea.style.left = '-999999px';
        document.body.appendChild(textArea);
        textArea.select();
        
        try {{
            document.execCommand('copy');
            showCopySuccess{message_index}();
        }} catch (err) {{
            console.error('Copy failed:', err);
        }}
        
        document.body.removeChild(textArea);
    }}
    
    function showCopySuccess{message_index}() {{
        const button = document.getElementById('copy-btn-{message_index}');
        const originalText = button.innerHTML;
        button.classList.add('copied');
        button.innerHTML = '✅';
        
        setTimeout(function() {{
            button.classList.remove('copied');
            button.innerHTML = originalText;
        }}, 2000);
    }}
    </script>
    '''
    
    # 渲染复制按钮
    st.components.v1.html(copy_html, height=50)
    
    # 使用正常的Streamlit按钮进行评分和重新生成
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        # 点赞按钮
        current_rating = st.session_state.message_ratings.get(message_index, None)
        like_pressed = current_rating == "like"
        
        if st.button("👍", key=f"like_{message_index}", help="点赞", 
                    type="primary" if like_pressed else "secondary"):
            if like_pressed:
                if message_index in st.session_state.message_ratings:
                    del st.session_state.message_ratings[message_index]
            else:
                st.session_state.message_ratings[message_index] = "like"
            st.rerun()
    
    with col2:
        # 踩按钮
        dislike_pressed = current_rating == "dislike"
        
        if st.button("👎", key=f"dislike_{message_index}", help="踩",
                    type="primary" if dislike_pressed else "secondary"):
            if dislike_pressed:
                if message_index in st.session_state.message_ratings:
                    del st.session_state.message_ratings[message_index]
            else:
                st.session_state.message_ratings[message_index] = "dislike"
            st.rerun()
    
    with col3:
        # 重新回答按钮（仅对AI回答显示）
        if question:
            if st.button("🔄", key=f"regenerate_{message_index}", help="重新生成回答"):
                regenerate_answer(question)
                st.rerun()

# ---------- Streamlit 主界面 ----------
def main():
    # 初始化会话状态
    initialize_session_state()
    
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
            if role == "assistant":
                # AI回答使用特殊容器
                st.markdown(f'<div class="ai-message-container">', unsafe_allow_html=True)
                st.write(text)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # 寻找对应的用户问题
                question = None
                if i > 0 and st.session_state.messages[i-1][0] == "user":
                    question = st.session_state.messages[i-1][1]
                
                # 渲染交互按钮
                render_message_actions_v2(i, text, question)
            else:
                # 用户消息正常显示
                st.write(text)
    
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
                
                # AI回答使用特殊容器
                st.markdown(f'<div class="ai-message-container">', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # 为新回答添加交互按钮
                new_message_index = len(st.session_state.messages) - 1
                render_message_actions_v2(new_message_index, response, st.session_state.last_question)
                
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
                    # AI回答使用特殊容器
                    st.markdown(f'<div class="ai-message-container">', unsafe_allow_html=True)
                    # 流式输出回答
                    response = st.write_stream(st.session_state.chain.stream(chain_input))
                    st.markdown('</div>', unsafe_allow_html=True)
                
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
                
                # 为新回答添加交互按钮
                message_index = len(st.session_state.messages) - 1
                render_message_actions_v2(message_index, response, prompt)
                    
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
