# 1. 在您的自定义CSS样式中替换以下部分：

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
    
    /* 消息按钮组样式 - 优化为左下角白框按钮 */
    .message-actions {
        position: fixed;
        bottom: 20px;
        left: 20px;
        display: flex;
        gap: 10px;
        z-index: 1000;
    }
    
    /* 统一的白框按钮样式 */
    .action-button {
        width: 50px;
        height: 50px;
        background: transparent;
        border: 2px solid rgba(255, 255, 255, 0.8);
        border-radius: 8px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }
    
    .action-button:hover {
        border-color: white;
        background: rgba(255, 255, 255, 0.1);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    .action-button:active {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    
    /* 复制按钮图标样式 */
    .copy-button::before {
        content: "📋";
        font-size: 20px;
        filter: grayscale(100%) brightness(0) invert(1);
    }
    
    /* 重新生成按钮图标样式 */
    .regenerate-button::before {
        content: "🔄";
        font-size: 20px;
        filter: grayscale(100%) brightness(0) invert(1);
        transition: transform 0.3s ease;
    }
    
    .regenerate-button:hover::before {
        transform: rotate(180deg);
    }
    
    /* 按钮点击波纹效果 */
    .action-button::after {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .action-button:active::after {
        width: 120px;
        height: 120px;
    }
    
    /* 成功状态样式 */
    .action-button.success {
        border-color: #4ade80 !important;
        animation: successPulse 0.3s ease;
    }
    
    @keyframes successPulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
    
    /* 响应式设计 */
    @media (max-width: 768px) {
        .message-actions {
            bottom: 15px;
            left: 15px;
            gap: 8px;
        }
        
        .action-button {
            width: 45px;
            height: 45px;
        }
        
        .copy-button::before,
        .regenerate-button::before {
            font-size: 18px;
        }
    }
    
    /* 隐藏原有的按钮文字 */
    .stButton > button {
        color: transparent !important;
        font-size: 0 !important;
    }
    
    /* 状态提示样式 */
    .status-message {
        position: fixed;
        bottom: 80px;
        left: 20px;
        background: rgba(0, 0, 0, 0.8);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 14px;
        opacity: 0;
        transform: translateY(10px);
        transition: all 0.3s ease;
        z-index: 1001;
    }
    
    .status-message.show {
        opacity: 1;
        transform: translateY(0);
    }
</style>
""", unsafe_allow_html=True)

# 2. 替换 create_copy_button_html 函数：
def create_copy_button_html(message_index, message_text):
    """创建优化的白框复制按钮HTML"""
    # 转义文本中的特殊字符
    escaped_text = message_text.replace('\\', '\\\\').replace('`', '\\`').replace("'", "\\'").replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')
    
    copy_html = f'''
    <div class="message-actions">
        <button onclick="copyToClipboard{message_index}()" 
                class="action-button copy-button"
                title="复制">
        </button>
        <button onclick="regenerateAnswer{message_index}()" 
                class="action-button regenerate-button"
                title="重新生成">
        </button>
    </div>
    
    <div id="status-message-{message_index}" class="status-message"></div>
    
    <script>
    function copyToClipboard{message_index}() {{
        const text = `{escaped_text}`;
        const button = document.querySelector('.copy-button');
        const statusElement = document.getElementById('status-message-{message_index}');
        
        if (navigator.clipboard && window.isSecureContext) {{
            navigator.clipboard.writeText(text).then(function() {{
                showStatus{message_index}('✅ 已复制', button);
            }}).catch(function(err) {{
                fallbackCopy{message_index}(text, button, statusElement);
            }});
        }} else {{
            fallbackCopy{message_index}(text, button, statusElement);
        }}
    }}
    
    function fallbackCopy{message_index}(text, button, statusElement) {{
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
                showStatus{message_index}('✅ 已复制', button);
            }} else {{
                showStatus{message_index}('❌ 复制失败', button);
            }}
        }} catch (err) {{
            showStatus{message_index}('❌ 复制失败', button);
        }}
        
        document.body.removeChild(textArea);
    }}
    
    function showStatus{message_index}(message, button) {{
        const statusElement = document.getElementById('status-message-{message_index}');
        statusElement.textContent = message;
        statusElement.classList.add('show');
        
        // 添加成功动画
        if (message.includes('已复制')) {{
            button.classList.add('success');
            setTimeout(() => button.classList.remove('success'), 300);
        }}
        
        // 3秒后隐藏状态消息
        setTimeout(() => {{
            statusElement.classList.remove('show');
        }}, 3000);
    }}
    
    function regenerateAnswer{message_index}() {{
        // 这里需要触发Streamlit的重新生成功能
        // 由于JavaScript无法直接调用Streamlit函数，这里只显示提示
        const button = document.querySelector('.regenerate-button');
        const statusElement = document.getElementById('status-message-{message_index}');
        showStatus{message_index}('请点击Streamlit重新生成按钮', button);
    }}
    </script>
    '''
    
    return copy_html

# 3. 修改按钮创建部分，将原来的列布局改为固定定位：
def create_message_actions_optimized(message_index, message_text, question=None):
    """创建优化的消息操作按钮组"""
    # 不再使用列布局，直接使用HTML和CSS固定定位
    copy_html = create_copy_button_html(message_index, message_text)
# 4. 在生成AI回答的函数中，替换按钮创建部分：
def generate_ai_response(prompt, msgs):
    """生成AI回答"""
    try:
        # ... 现有的生成逻辑保持不变 ...
        
        # 在函数最后，替换原来的按钮创建代码：
        message_index = len(st.session_state.messages) - 1
        
        # 使用优化的按钮（固定定位在左下角）
        copy_html = create_copy_button_html(message_index, response)
        st.components.v1.html(copy_html, height=0)
        
        # 保留Streamlit的重新生成按钮功能，但隐藏文字
        col1, col2, col3 = st.columns([1, 1, 8])
        with col2:
            if st.button("", key=f"regen_new_{message_index}", help="重新生成回答"):
                st.session_state.regenerate_question = prompt
                st.session_state.regenerate_index = message_index
                st.rerun()
                
    except Exception as e:
        # ... 错误处理保持不变 ...

# 5. 在主函数的聊天历史显示部分，也要相应修改：
# 在显示聊天历史的循环中，替换按钮创建部分：
for i, (role, text) in enumerate(st.session_state.messages):
    with msgs.chat_message(role):
        st.write(text)
        
        # 为AI回答添加操作按钮
        if role == "assistant":
            # 寻找对应的用户问题
            question = None
            if i > 0 and st.session_state.messages[i-1][0] == "user":
                question = st.session_state.messages[i-1][1]
            
            # 使用优化的按钮
            copy_html = create_copy_button_html(i, text)
            st.components.v1.html(copy_html, height=0)
            
            # 保留隐藏的重新生成按钮
            if question is not None:
                col1, col2, col3 = st.columns([1, 1, 8])
                with col2:
                    if st.button("", key=f"regen_history_{i}", help="重新生成回答"):
                        st.session_state.regenerate_question = question
                        st.session_state.regenerate_index = i
                        st.rerun()
