# 修改后的 create_message_actions_html 函数 - 统一使用HTML按钮，保持高度一致
def create_message_actions_html(message_index, message_text, question=None):
    """创建统一样式的消息操作按钮组HTML"""
    # 转义文本中的特殊字符
    escaped_text = message_text.replace('\\', '\\\\').replace('`', '\\`').replace("'", "\\'").replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')
    
    # 构建HTML，包含复制和重新生成按钮
    buttons_html = f'''
    <div style="display: flex; align-items: center; gap: 10px; margin: 5px 0; height: 40px;">
        <!-- 复制按钮 -->
        <button onclick="copyToClipboard{message_index}()"
                class="copy-button"
                style="background: white; border: 1px solid #ddd; border-radius: 6px; padding: 6px; cursor: pointer; font-size: 18px; color: #666; transition: all 0.2s ease; display: inline-flex; align-items: center; justify-content: center; width: 36px; height: 36px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); flex-shrink: 0;"
                onmouseover="this.style.background='#f8f9fa'; this.style.borderColor='#adb5bd'; this.style.color='#333'; this.style.transform='translateY(-1px)'; this.style.boxShadow='0 2px 6px rgba(0,0,0,0.15)';"
                onmouseout="this.style.background='white'; this.style.borderColor='#ddd'; this.style.color='#666'; this.style.transform='translateY(0px)'; this.style.boxShadow='0 1px 3px rgba(0,0,0,0.1)';">
            📋
        </button>
        
        <!-- 重新生成按钮 (只有当question不为None时才显示) -->
        {f'''
        <button onclick="regenerateAnswer{message_index}()"
                class="regenerate-button"
                style="background: white; border: 1px solid #ddd; border-radius: 6px; padding: 6px; cursor: pointer; font-size: 18px; color: #666; transition: all 0.2s ease; display: inline-flex; align-items: center; justify-content: center; width: 36px; height: 36px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); flex-shrink: 0;"
                onmouseover="this.style.background='#f8f9fa'; this.style.borderColor='#adb5bd'; this.style.color='#333'; this.style.transform='translateY(-1px)'; this.style.boxShadow='0 2px 6px rgba(0,0,0,0.15)';"
                onmouseout="this.style.background='white'; this.style.borderColor='#ddd'; this.style.color='#666'; this.style.transform='translateY(0px)'; this.style.boxShadow='0 1px 3px rgba(0,0,0,0.1)';">
            🔄
        </button>
        ''' if question is not None else ''}
        
        <span id="status-{message_index}" style="color: #28a745; font-size: 12px; line-height: 1;"></span>
    </div>

    <script>
    function copyToClipboard{message_index}() {{
        const text = `{escaped_text}`;
        const statusElement = document.getElementById('status-{message_index}');
        const button = event.target;

        if (navigator.clipboard && window.isSecureContext) {{
            navigator.clipboard.writeText(text).then(function() {{
                statusElement.textContent = '✅ 已复制';
                button.style.color = '#28a745';
                button.style.background = '#f0f8f0';
                button.style.borderColor = '#28a745';
                setTimeout(() => {{
                    statusElement.textContent = '';
                    button.style.color = '#666';
                    button.style.background = 'white';
                    button.style.borderColor = '#ddd';
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
                statusElement.textContent = '✅ 已复制';
                button.style.color = '#28a745';
                button.style.background = '#f0f8f0';
                button.style.borderColor = '#28a745';
            }} else {{
                statusElement.textContent = '❌ 复制失败';
            }}
        }} catch (err) {{
            statusElement.textContent = '❌ 复制失败';
        }}

        document.body.removeChild(textArea);
        setTimeout(() => {{
            statusElement.textContent = '';
            button.style.color = '#666';
            button.style.background = 'white';
            button.style.borderColor = '#ddd';
        }}, 2000);
    }}
    
    {f'''
    function regenerateAnswer{message_index}() {{
        const statusElement = document.getElementById('status-{message_index}');
        const button = event.target;
        
        // 显示加载状态
        statusElement.textContent = '🔄 正在重新生成...';
        button.style.color = '#ffc107';
        button.style.background = '#fff8e1';
        button.style.borderColor = '#ffc107';
        
        // 触发Streamlit重新运行
        // 这里需要通过设置session state来触发重新生成
        window.parent.postMessage({{
            type: 'streamlit:setComponentValue',
            key: 'regenerate_trigger_{message_index}',
            value: Math.random()
        }}, '*');
    }}
    ''' if question is not None else ''}
    </script>
    '''

    return buttons_html

# 修改显示聊天历史的部分
def display_chat_history_with_aligned_buttons(msgs):
    """显示聊天历史，使用对齐的按钮"""
    for i, (role, text) in enumerate(st.session_state.messages):
        with msgs.chat_message(role):
            st.write(text)
                        
            # 为AI回答添加操作按钮
            if role == "assistant":
                # 寻找对应的用户问题
                question = None
                if i > 0 and st.session_state.messages[i-1][0] == "user":
                    question = st.session_state.messages[i-1][1]
                                
                # 添加分隔线
                st.markdown("---")
                
                # 使用统一的HTML按钮组
                buttons_html = create_message_actions_html(i, text, question)
                st.components.v1.html(buttons_html, height=50)
                
                # 检查是否触发了重新生成
                regenerate_key = f'regenerate_trigger_{i}'
                if regenerate_key in st.session_state:
                    # 清除触发器
                    del st.session_state[regenerate_key]
                    # 设置重新生成请求
                    if question is not None:
                        st.session_state.regenerate_question = question
                        st.session_state.regenerate_index = i
                        st.rerun()

# 修改生成AI回答函数中的按钮部分
def generate_ai_response_with_aligned_buttons(prompt, msgs):
    """生成AI回答，使用对齐的按钮"""
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
                
        # 添加对齐的按钮组
        message_index = len(st.session_state.messages) - 1
                
        # 添加分隔线
        st.markdown("---")
        
        # 使用统一的HTML按钮组，包含复制和重新生成按钮
        buttons_html = create_message_actions_html(message_index, response, prompt)
        st.components.v1.html(buttons_html, height=50)
                    
    except Exception as e:
        error_msg = f"生成回答时出错: {str(e)}"
        st.error(error_msg)
        st.session_state.messages.append(("assistant", "抱歉，生成回答时出现了错误。请检查网络连接和API密钥。"))
        # 显示详细错误信息供调试
        with st.expander("错误详情"):
            st.code(str(e))
