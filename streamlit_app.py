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
    /* 全局样式 */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    /* 主容器 */
    .main .block-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* 标题样式 */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="rgba(255,255,255,0.1)"/><circle cx="75" cy="75" r="1" fill="rgba(255,255,255,0.1)"/><circle cx="50" cy="10" r="0.5" fill="rgba(255,255,255,0.1)"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
        opacity: 0.3;
    }
    
    .main-header h1 {
        color: white;
        text-align: center;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        position: relative;
        z-index: 1;
    }
    
    /* 聊天界面美化 */
    .stChatMessage {
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(0, 0, 0, 0.05);
    }
    
    .stChatMessage[data-testid="user-message"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .stChatMessage[data-testid="assistant-message"] {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-left: 4px solid #667eea;
    }
    
    /* 输入框美化 */
    .stChatInputContainer {
        border-radius: 25px;
        border: 2px solid rgba(102, 126, 234, 0.3);
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }
    
    .stChatInputContainer:focus-within {
        border-color: #667eea;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
    }
    
    /* 侧边栏美化 */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* 按钮美化 */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* 文件上传区域 */
    .uploadedFile {
        border-radius: 12px;
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid rgba(102, 126, 234, 0.2);
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* 指标卡片 */
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    /* 响应式设计 */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 1.8rem;
        }
        
        .main .block-container {
            margin: 0.5rem;
            padding: 1rem;
        }
    }
    
    /* 滚动条美化 */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* 消息操作按钮样式 */
    .message-actions {
        display: flex;
        gap: 0.5rem;
        margin-top: 0.5rem;
        justify-content: flex-end;
        opacity: 0.7;
        transition: opacity 0.3s ease;
    }
    
    .message-actions:hover {
        opacity: 1;
    }
    
    .action-btn {
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 8px;
        padding: 0.25rem 0.5rem;
        font-size: 0.8rem;
        cursor: pointer;
        transition: all 0.3s ease;
        color: #667eea;
    }
    
    .action-btn:hover {
        background: rgba(102, 126, 234, 0.2);
        transform: translateY(-1px);
    }
    
    .action-btn.liked {
        background: #10b981;
        color: white;
        border-color: #10b981;
    }
    
    .action-btn.disliked {
        background: #ef4444;
        color: white;
        border-color: #ef4444;
    }
</style>
""", unsafe_allow_html=True)

# ---------- 1. 从本地 Markdown 文件获取文档内容 ----------
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

# ---------- 2. 处理上传文件 ----------
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

# ---------- 3. 测试检索器功能 ----------
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

# ---------- 3. 构建检索器 ----------
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

# ---------- 4. 构建问答链（带记忆功能） ----------
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

# ---------- 5. 侧边栏功能 ----------
def setup_sidebar():
    with st.sidebar:
        # 侧边栏标题
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 1.5rem;">
            <h2 style="color: white; margin: 0;">⚙️ 控制面板</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📁 文件上传")
        
        # 文件上传器
        uploaded_files = st.file_uploader(
            "拖拽文件或点击上传",
            type=['txt', 'md', 'pdf', 'docx', 'doc'],
            accept_multiple_files=True,
            help="📋 支持格式：TXT, MD, PDF, DOCX, DOC"
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
                    with st.spinner(f"🔄 正在处理: {uploaded_file.name}"):
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
                        else:
                            st.error(f"❌ {uploaded_file.name} 处理失败！")
            
            if new_files_processed > 0:
                # 重新构建chain（每次都会重新构建检索器）
                st.session_state.chain = get_qa_chain_with_memory()
                st.balloons()  # 庆祝效果
                st.success(f"🎉 成功处理 {new_files_processed} 个新文件！知识库已更新。")
                st.rerun()
        
        # 显示已上传的文件
        if 'uploaded_files_info' in st.session_state and st.session_state.uploaded_files_info:
            st.markdown("### 📋 文件库")
            for i, file_info in enumerate(st.session_state.uploaded_files_info):
                with st.expander(f"📄 {file_info['name']}", expanded=False):
                    # 文件信息展示
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("文件大小", f"{file_info['size']:,} B")
                    with col2:
                        st.metric("内容长度", f"{file_info['content_length']:,} 字符")
                    
                    # 显示文档内容预览
                    if 'uploaded_docs' in st.session_state and i < len(st.session_state.uploaded_docs):
                        preview = st.session_state.uploaded_docs[i][:300] + "..." if len(st.session_state.uploaded_docs[i]) > 300 else st.session_state.uploaded_docs[i]
                        st.text_area("📖 内容预览:", preview, height=120, disabled=True)
        
        st.markdown("---")
        
        # 显示知识库状态
        if 'uploaded_docs' in st.session_state:
            total_chars = sum(len(doc) for doc in st.session_state.uploaded_docs)
            st.markdown("### 📊 知识库状态")
            
            # 美化的状态卡片
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); padding: 1rem; border-radius: 12px; border: 1px solid rgba(102, 126, 234, 0.2);">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="color: #667eea; font-weight: 600;">📚 文档数量</span>
                    <span style="color: #1e293b; font-weight: 700;">{len(st.session_state.uploaded_docs)}</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #667eea; font-weight: 600;">📝 总字符数</span>
                    <span style="color: #1e293b; font-weight: 700;">{total_chars:,}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("")  # 添加间距
            
            # 测试检索功能
            if st.button("🔍 测试知识库检索", use_container_width=True):
                test_query = st.text_input("🔎 输入测试问题:", value="学生手册", key="test_query")
                if test_query:
                    with st.spinner("🔄 正在测试检索..."):
                        docs = test_retriever(test_query)
                        if docs:
                            st.success(f"✅ 检索成功！找到 {len(docs)} 个相关片段")
                            with st.expander("📋 查看检索结果"):
                                for i, doc in enumerate(docs):
                                    st.markdown(f"**📄 片段 {i+1}:**")
                                    st.markdown(f"```\n{doc.page_content[:300]}...\n```")
                        else:
                            st.warning("⚠️ 未找到相关内容")
        
        st.markdown("---")
        
        # 操作按钮区域
        st.markdown("### 🛠️ 操作中心")
        
        # 清除对话历史按钮
        if st.button("🗑️ 清除对话历史", use_container_width=True, type="secondary"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.success("✅ 对话历史已清除！")
            st.rerun()
        
        # 清除上传文件按钮
        if st.button("📁 清除上传文件", use_container_width=True, type="secondary"):
            if 'uploaded_docs' in st.session_state:
                del st.session_state.uploaded_docs
            if 'uploaded_files_info' in st.session_state:
                del st.session_state.uploaded_files_info
            st.session_state.chain = get_qa_chain_with_memory()
            st.success("✅ 所有上传文件已清除！")
            st.rerun()
        
        st.markdown("---")
        
        # 使用说明
        with st.expander("📖 使用指南", expanded=False):
            st.markdown("""
            ### ✨ 功能特点
            
            **🔍 智能检索**  
            从知识库中查找相关信息
            
            **🧠 知识融合**  
            找不到时使用AI自身知识回答
            
            **💭 对话记忆**  
            记住之前的对话内容
            
            **📁 多格式支持**  
            支持TXT、MD、PDF、DOCX、DOC
            
            ### 🚀 使用步骤
            
            1. **📤 上传文档** - 拖拽或点击上传相关文档
            2. **⏳ 等待处理** - 系统自动解析并构建知识库  
            3. **💬 开始对话** - 在聊天框中输入问题
            4. **🎯 获得回答** - AI结合文档内容智能回答
            
            ### 💡 小贴士
            
            - 文件上传后会自动构建知识库
            - 大文件处理可能需要几秒钟
            - 支持同时上传多个文件
            - 可以使用测试功能验证知识库
            """)
        
        # 调试信息（可选）
        if st.checkbox("🔧 显示调试信息"):
            st.markdown("### 🐛 调试面板")
            st.json({
                "上传文档数量": len(st.session_state.get('uploaded_docs', [])),
                "对话历史长度": len(st.session_state.get('chat_history', [])),
                "消息总数": len(st.session_state.get('messages', [])),
                "会话状态": list(st.session_state.keys())
            })

# ---------- 6. 消息操作功能 ----------
def render_message_actions(message_id, message_content):
    """渲染消息操作按钮"""
    
    # 创建JavaScript复制功能
    copy_js = f"""
    <script>
        function copyToClipboard_{message_id}() {{
            const text = `{message_content.replace('`', '').replace("'", "").replace('"', '')}`;
            navigator.clipboard.writeText(text).then(function() {{
                // 可以添加成功提示
            }});
        }}
    </script>
    """
    st.markdown(copy_js, unsafe_allow_html=True)
    
    # 创建按钮布局
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 6])
    
    # 初始化消息反馈状态
    feedback_key = f"feedback_{message_id}"
    if feedback_key not in st.session_state:
        st.session_state[feedback_key] = {"liked": False, "disliked": False, "copied": False}
    
    with col1:
        # 复制按钮
        copy_text = "📋" if not st.session_state[feedback_key]["copied"] else "✅"
        if st.button(copy_text, key=f"copy_{message_id}", help="复制内容"):
            st.session_state[feedback_key]["copied"] = True
            # 显示复制成功的临时消息
            st.success("已复制到剪贴板！", icon="📋")
            # 使用JavaScript复制（备用方案）
            st.markdown(f"""
            <script>
                copyToClipboard_{message_id}();
            </script>
            """, unsafe_allow_html=True)
            st.rerun()
    
    with col2:
        # 点赞按钮
        like_style = "👍🏻" if st.session_state[feedback_key]["liked"] else "👍"
        if st.button(like_style, key=f"like_{message_id}", help="点赞这个回答"):
            st.session_state[feedback_key]["liked"] = not st.session_state[feedback_key]["liked"]
            st.session_state[feedback_key]["disliked"] = False
            if st.session_state[feedback_key]["liked"]:
                st.success("感谢您的点赞！", icon="👍")
            st.rerun()
    
    with col3:
        # 不点赞按钮
        dislike_style = "👎🏻" if st.session_state[feedback_key]["disliked"] else "👎"
        if st.button(dislike_style, key=f"dislike_{message_id}", help="这个回答不够好"):
            st.session_state[feedback_key]["disliked"] = not st.session_state[feedback_key]["disliked"]
            st.session_state[feedback_key]["liked"] = False
            if st.session_state[feedback_key]["disliked"]:
                st.warning("感谢您的反馈，我们会持续改进！", icon="💭")
            st.rerun()
    
    with col4:
        # 重新回答按钮
        if st.button("🔄", key=f"regenerate_{message_id}", help="重新生成回答"):
            return True
    
    # 显示反馈状态
    if st.session_state[feedback_key]["liked"]:
        st.markdown('<div style="color: #10b981; font-size: 0.8rem;">👍 您觉得这个回答很有帮助</div>', unsafe_allow_html=True)
    elif st.session_state[feedback_key]["disliked"]:
        st.markdown('<div style="color: #ef4444; font-size: 0.8rem;">👎 我们会改进这个回答</div>', unsafe_allow_html=True)
    
    return False

# ---------- 8. Streamlit 主界面 ----------
def add_copy_js():
    """添加全局复制JavaScript功能"""
    st.markdown("""
    <script>
        function copyText(text, buttonId) {
            navigator.clipboard.writeText(text).then(function() {
                // 更新按钮显示
                const button = document.getElementById(buttonId);
                if (button) {
                    const original = button.innerHTML;
                    button.innerHTML = '✅';
                    setTimeout(() => {
                        button.innerHTML = original;
                    }, 2000);
                }
            }).catch(function(err) {
                console.error('复制失败: ', err);
            });
        }
    </script>
    """, unsafe_allow_html=True)
def main():
    # 添加复制功能的JavaScript
    add_copy_js()
    
    # 页面标题
    st.markdown("""
    <div class="main-header">
        <h1>🦜🔗 智能文档问答助手</h1>
        <p style="color: rgba(255,255,255,0.9); text-align: center; margin: 0.5rem 0 0 0; font-size: 1.1rem;">
            基于大模型的智能检索与对话系统 ✨
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 设置侧边栏
    setup_sidebar()
    
    # 初始化会话状态
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "chain" not in st.session_state:
        st.session_state.chain = get_qa_chain_with_memory()
    
    # 主聊天区域
    st.markdown("### 💬 智能问答")
    
    # 聊天消息容器
    chat_container = st.container()
    
    # 显示聊天历史
    with chat_container:
        for i, (role, text) in enumerate(st.session_state.messages):
            with st.chat_message(role):
                st.write(text)
                
                # 为AI回答添加操作按钮
                if role == "assistant":
                    st.divider()
                    regenerate = render_message_actions(i, text)
                    
                    # 如果点击了重新回答按钮
                    if regenerate:
                        # 获取对应的用户问题
                        if i > 0 and st.session_state.messages[i-1][0] == "user":
                            user_question = st.session_state.messages[i-1][1]
                            
                            # 重新生成回答
                            with st.spinner("正在重新生成回答..."):
                                try:
                                    chain_input = {
                                        "question": user_question,
                                        "chat_history": st.session_state.chat_history[:i-1] if i > 1 else []
                                    }
                                    
                                    # 生成新回答
                                    new_response = ""
                                    for chunk in st.session_state.chain.stream(chain_input):
                                        new_response += chunk
                                    
                                    # 更新消息
                                    st.session_state.messages[i] = ("assistant", new_response)
                                    
                                    # 更新对话历史
                                    if len(st.session_state.chat_history) > i:
                                        st.session_state.chat_history[i] = AIMessage(content=new_response)
                                    
                                    st.success("重新生成完成！", icon="✨")
                                    st.rerun()
                                    
                                except Exception as e:
                                    st.error(f"重新生成失败: {str(e)}")
    
    # 用户输入
    if prompt := st.chat_input("请输入你的问题... 💭"):
        # 添加用户消息
        st.session_state.messages.append(("user", prompt))
        
        # 显示用户消息
        with st.chat_message("user"):
            st.write(prompt)
        
        # 生成AI回答
        with st.chat_message("assistant"):
            try:
                # 准备输入数据，包含对话历史
                chain_input = {
                    "question": prompt,
                    "chat_history": st.session_state.chat_history
                }
                
                # 显示处理状态
                with st.spinner("🤖 AI正在思考中..."):
                    # 流式输出回答
                    response = st.write_stream(st.session_state.chain.stream(chain_input))
                
                # 添加操作按钮
                st.divider()
                message_id = len(st.session_state.messages)
                render_message_actions(message_id, response)
                
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
                    
            except Exception as e:
                error_msg = f"生成回答时出错: {str(e)}"
                st.error(error_msg, icon="❌")
                st.session_state.messages.append(("assistant", "抱歉，生成回答时出现了错误。请检查网络连接和API密钥。"))
                # 显示详细错误信息供调试
                with st.expander("🔧 错误详情"):
                    st.code(str(e))
                    st.markdown("**可能的解决方案：**")
                    st.markdown("- 检查网络连接")
                    st.markdown("- 验证 OPENAI_API_KEY 是否正确设置")
                    st.markdown("- 确认上传的文档格式是否支持")
    
    # 底部统计信息
    st.markdown("---")
    
    # 创建美观的指标卡片
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #667eea; margin: 0;">💬</h3>
            <h2 style="margin: 0.5rem 0;">{}</h2>
            <p style="color: #64748b; margin: 0;">对话轮数</p>
        </div>
        """.format(len(st.session_state.messages) // 2), unsafe_allow_html=True)
    
    with col2:
        uploaded_count = len(st.session_state.get('uploaded_files_info', []))
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #667eea; margin: 0;">📁</h3>
            <h2 style="margin: 0.5rem 0;">{}</h2>
            <p style="color: #64748b; margin: 0;">已上传文件</p>
        </div>
        """.format(uploaded_count), unsafe_allow_html=True)
    
    with col3:
        memory_count = len(st.session_state.chat_history) // 2
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #667eea; margin: 0;">🧠</h3>
            <h2 style="margin: 0.5rem 0;">{}</h2>
            <p style="color: #64748b; margin: 0;">记忆对话数</p>
        </div>
        """.format(memory_count), unsafe_allow_html=True)
    
    # 添加使用提示
    if len(st.session_state.messages) == 0:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); border-radius: 15px; margin: 2rem 0;">
            <h3 style="color: #667eea; margin-bottom: 1rem;">🚀 开始使用</h3>
            <p style="color: #64748b; margin-bottom: 1rem;">上传文档并开始智能问答吧！</p>
            <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;">
                <span style="background: rgba(102, 126, 234, 0.1); padding: 0.5rem 1rem; border-radius: 20px; color: #667eea;">📚 支持PDF、Word、TXT等格式</span>
                <span style="background: rgba(102, 126, 234, 0.1); padding: 0.5rem 1rem; border-radius: 20px; color: #667eea;">🔍 智能检索相关内容</span>
                <span style="background: rgba(102, 126, 234, 0.1); padding: 0.5rem 1rem; border-radius: 20px; color: #667eea;">💭 具备对话记忆</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 添加页脚信息
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; color: #64748b; font-size: 0.9rem;">
        <p style="margin: 0;">🤖 智能文档问答助手 | 基于 LangChain + OpenAI + Streamlit 构建</p>
        <p style="margin: 0.5rem 0 0 0;">💡 让AI成为您的智能知识助手</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
